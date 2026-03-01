import pytest
import time
import base64
import requests
from utils import *

server: ServerProcess

SYSTEM = "You are a helpful assistant."
FINAL_USER = "[U Last]This is the most recent user message."
N_TURNS_OVERFLOW = 128

def _user_msg(i: int) -> str:
    return f"[U{i:02d}] Please explain topic {i} in detail."

def _asst_msg(i: int) -> str:
    return f"[A{i:02d}] Here is my explanation of topic {i}."


def _get_messages(n_turns: int = 128, include_final_user: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(1, n_turns + 1):
        msgs.append({"role": "user",      "content": _user_msg(i)})
        msgs.append({"role": "assistant", "content": _asst_msg(i)})
    if include_final_user:
        msgs.append({"role": "user", "content": FINAL_USER})
    return msgs


def _short_messages() -> list[dict]:
    return _get_messages(n_turns=1)

def assert_turns_consistency_in_prompt(prompt: str, include_final_user: bool = True):
    """
    Verify that the user and assistant turns in the rendered prompt are consistent
    with the chat template, meaning:
        - no assistant message appears without its preceding user message
        - final user message is present.
        - system prompt is present.
    """
    turns = prompt.split("\n")
    user_turns = [t for t in turns if t.startswith("[U")]
    asst_turns = [t for t in turns if t.startswith("[A")]
    assert SYSTEM in prompt
    if include_final_user:
        assert len(user_turns) == len(asst_turns) + 1, "Each assistant turn should be preceded by a user turn, and there should be one extra user turn at the end"
        assert FINAL_USER in prompt
    else:
        assert len(user_turns) == len(asst_turns), "Each assistant turn should be preceded by a user turn, and there should be no extra user turn at the end"
    for i in range(len(asst_turns)):
        expected_asst = _asst_msg(i + 1)
        if expected_asst in prompt:
            expected_user = _user_msg(i + 1)
            assert expected_user in prompt, f"User turn {i+1} should be present in the prompt as it precedes assistant turn {i+1}"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # tinyllama2 preset: n_ctx=512, n_slots=2, n_predict=64
    # so per-slot context = 512 // 2 = 256 tokens
    server.jinja = True
    server.chat_template = "chatml"


# Tests

def test_chat_truncate_overflow_without_chat_truncate_flag():
    """
    Baseline: a long conversation exceeds the per-slot context when
    --chat-truncate is not set, returning a 400 exceed_context_size_error.
    """
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(),
    })
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_context_size_error"


def test_chat_truncate_prevents_overflow():
    """
    With --chat-truncate set, long conversation succeeds.
    """
    global server
    server.chat_truncate = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200


def test_chat_truncate_no_op():
    """
    A short conversation already below the truncation target is left untouched
    """
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _short_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert "[U01]" in prompt, "No turn should not be dropped"
    assert_turns_consistency_in_prompt(prompt)


def test_chat_truncate_prompt_within_budget():
    """
    After truncation, prompt_tokens must be strictly less than the truncation
    target: floor(fraction * per_slot_ctx).
    """
    global server
    server.chat_truncate = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200
    assert server.n_ctx is not None and server.n_slots is not None
    per_slot_ctx = server.n_ctx // server.n_slots
    target = int(server.chat_truncate * per_slot_ctx)
    assert res.body["usage"]["prompt_tokens"] < target


def test_chat_truncate_drops_oldest_keeps_newest():
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert _user_msg(1) not in prompt
    assert_turns_consistency_in_prompt(prompt)

def test_chat_truncate_non_user_newest_preserved():
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW, include_final_user=False),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert _user_msg(1) not in prompt
    assert_turns_consistency_in_prompt(prompt, include_final_user=False)

def test_chat_truncate_n_predict_threshold_vs_max_completion_tokens():
    """
    The truncation threshold differs depending on whether max_completion_tokens is in the request:

        no max_completion_tokens  -> n_predict = server.n_predict = 64  -> threshold = 256 - 64  = 192
        max_completion_tokens=5   -> n_predict = 5                      -> threshold = 256 - 5   = 251

    We first probe to find n_turns so that the true token count T satisfies:

        target (192) < T < threshold_with_max5 (251)

    Then, with T turns:
    - Without max_completion_tokens: threshold=192 < T -> truncation fires, prompt < target
    - With max_completion_tokens=5:  threshold=251 > T -> truncation silent, prompt unchanged
    """
    global server
    server.chat_truncate = 0.8
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None and server.n_predict is not None
    max_completion_tokens = 5
    per_slot_ctx = server.n_ctx // server.n_slots
    threshold_no_max_completion_tokens = per_slot_ctx - server.n_predict
    threshold_with_max5 = per_slot_ctx - max_completion_tokens
    target = int(server.chat_truncate * per_slot_ctx)

    assert threshold_no_max_completion_tokens < target < threshold_with_max5

    # Find n_turns so that target < T < threshold_with_max5
    found_n_turns  = None
    found_n_tokens = None
    for n_turns in range(0, 10):
        probe = server.make_request("POST", "/chat/completions", data={
            "max_completion_tokens": 5,
            "messages": _get_messages(n_turns),
        })
        if probe.status_code != 200:
            break
        pt = probe.body["usage"]["prompt_tokens"]
        if target < pt < threshold_with_max5:
            found_n_turns  = n_turns
            found_n_tokens = pt
            break

    assert found_n_turns is not None

    msgs = _get_messages(found_n_turns)

    # Without max_completion_tokens: threshold=192 < T, then truncation fires
    res_no_max = server.make_request("POST", "/chat/completions", data={
        "messages": msgs,
    })
    assert res_no_max.status_code == 200
    assert res_no_max.body["usage"]["prompt_tokens"] < target

    # With max_completion_tokens=5: threshold=251 > T, then truncation silent
    res_max5 = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": msgs,
    })
    assert res_max5.status_code == 200
    assert res_max5.body["usage"]["prompt_tokens"] == found_n_tokens


def test_chat_truncate_target_capped_to_budget():
    """
    When chat_truncate * n_ctx > n_ctx - n_predict (the generation budget),
    chat_truncate_target_tokens caps the truncation target at the budget.
    """
    global server

    n_predict_req = 128
    server.chat_truncate = 0.99
    server.n_predict = n_predict_req
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None
    per_slot_ctx    = server.n_ctx // server.n_slots
    fraction_target = int(server.chat_truncate * per_slot_ctx)
    budget          = per_slot_ctx - n_predict_req

    assert fraction_target > budget

    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": n_predict_req,
        "messages": _get_messages(),
    })
    assert res.status_code == 200

    assert res.body["usage"]["prompt_tokens"] < budget


def test_chat_truncate_very_low_fraction_preserves_last_user_msg():
    """
    Even with small chat_truncate fraction, truncation loop drops keeps system + 1 user prompt
    """
    global server
    server.chat_truncate = 0.001
    server.n_predict = -1
    server.debug = True
    server.start()

    res = server.make_request("POST", "/chat/completions", data={
        "messages": _get_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    templated_sys_prompt = "<s> <|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    templated_last_message = "<|im_start|>user\n" + FINAL_USER + "<|im_end|>\n<|im_start|>assistant\n"
    assert templated_sys_prompt + templated_last_message == prompt


def test_chat_truncate_negative_value_rejected():
    global server
    server.chat_truncate = 0.0
    with pytest.raises(RuntimeError):
        server.start()


def test_chat_truncate_above_one_rejected():
    global server
    server.chat_truncate = 1.0
    with pytest.raises(RuntimeError):
        server.start()


def test_chat_truncate_after_sleep_wake():
    """
    TODO This test can pass both cases (use vocab refreshed or stale from chat_params), 
    TODO as after sleeping the vocab pointer can be the same.

    This validates that the vocab pointer (used for token counting in truncation)
    is correctly refreshed when the server wakes up, and not stale from before sleep.
    """
    global server
    server.chat_truncate = 0.8
    server.sleep_idle_seconds = 5
    server.debug = True
    server.start()

    # First request before sleep - should work
    res1 = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(N_TURNS_OVERFLOW),
    })
    assert res1.status_code == 200

    # Wait for server to go to sleep
    time.sleep(server.sleep_idle_seconds*1.2)

    # Verify server is sleeping
    res_props = server.make_request("GET", "/props")
    assert res_props.status_code == 200
    assert res_props.body["is_sleeping"] == True

    # Request after wake - should still work (vocab pointer must be valid)
    res2 = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(N_TURNS_OVERFLOW),
    })
    assert res2.status_code == 200
    prompt = res2.body["__verbose"]["prompt"]
    assert "[U01]" in prompt, "No turn should not be dropped"

    # Verify server woke up
    res_props = server.make_request("GET", "/props")
    assert res_props.status_code == 200
    assert res_props.body["is_sleeping"] == False


def _get_test_image_base64(image_id: int) -> str:
    """Fetch test image and return as base64 data URI."""
    # Using the same test images as test_vision_api.py
    urls = [
        "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png",
        "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png",
    ]
    url = urls[image_id % len(urls)]
    response = requests.get(url)
    response.raise_for_status()
    return "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")


def _get_multimodal_messages(n_image_turns: int, final_image_id: int) -> list[dict]:
    """
    Create messages where each turn has a different image.
    Images are numbered so we can verify which ones remain after truncation.
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}]
    for i in range(n_image_turns):
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"[IMG{i:02d}] What is in this image?"},
                {"type": "image_url", "image_url": {"url": _get_test_image_base64(i)}},
            ]
        })
        msgs.append({"role": "assistant", "content": f"[A{i:02d}] I see something in the image."})
    # Final turn with a specific image
    msgs.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"[IMG_FINAL] What is in this final image?"},
            {"type": "image_url", "image_url": {"url": _get_test_image_base64(final_image_id)}},
        ]
    })
    return msgs


# @pytest.mark.skip(reason="Known issue: multimodal + truncation has media index mismatch bug")
def test_chat_truncate_multimodal_index_mismatch():
    """
    TODO Polish sloppy test
    
    Test that truncation correctly handles messages with images.

    KNOWN ISSUE: Media files are extracted BEFORE truncation happens.
    If message 1 has image A (out_files[0]) and message 2 has image B (out_files[1]),
    and truncation removes message 1, the out_files still has [A, B] but message 2
    now expects its image at index 0.

    This test documents the bug - it should FAIL until the bug is fixed,
    then the skip marker can be removed.
    """
    global server
    # Use tinygemma3 for multimodal
    server = ServerPreset.tinygemma3()
    server.jinja = True
    server.chat_truncate = 0.5  # Aggressive truncation to force dropping messages
    server.debug = True
    server.start()

    # Create messages with multiple images that will trigger truncation
    # First image is a truck (index 0), second/final is a cat (index 1)
    msgs = _get_multimodal_messages(n_image_turns=3, final_image_id=1)

    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": msgs,
    })

    # If truncation removed early messages but media indices are misaligned,
    # the model might see the wrong image for the remaining messages
    assert res.status_code == 200

    # The response should reference what's in the FINAL image (cat), not a truck
    # This assertion may fail due to the index mismatch bug
    content = res.body["choices"][0]["message"]["content"].lower()
    # Note: tinygemma3 is trained on CIFAR-10, so it might say "cat" or "frog"
    # The key is it should NOT be describing the first image if that was truncated


# @pytest.mark.skip(reason="Known issue: token counting ignores image token cost")
def test_chat_truncate_multimodal_token_counting():
    """
    TODO Polish sloppy test
    Test that truncation correctly accounts for image tokens.

    KNOWN ISSUE: chat_n_tokens() only counts text tokens, not image tokens.
    Images can consume hundreds of tokens but the marker is just a few characters.
    This means truncation decisions are incorrect for multimodal chats.

    This test documents the bug - it should FAIL until the bug is fixed,
    then the skip marker can be removed.
    """
    global server
    # Use tinygemma3 for multimodal
    server = ServerPreset.tinygemma3()
    server.jinja = True
    server.n_ctx = 512  # Small context
    server.n_slots = 1  # Single slot = 512 tokens
    server.chat_truncate = 0.8  # Target = 409 tokens
    server.start()

    # Create a message with an image - the text is short but image uses many tokens
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "image_url", "image_url": {"url": _get_test_image_base64(0)}},
            ]
        },
    ]

    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": msgs,
    })

    # With proper image token counting, this might exceed context and need truncation
    # or return an error. Currently it might succeed but use more tokens than expected,
    # potentially causing issues downstream.

    if res.status_code == 200:
        # Check if prompt_tokens reflects actual token usage including image
        # This assertion documents the expected behavior once fixed
        prompt_tokens = res.body["usage"]["prompt_tokens"]
        # Image tokens should be significantly more than just text tokens
        # A typical image might use 256+ tokens
        # If prompt_tokens is very low (< 50), token counting is likely wrong
        assert prompt_tokens > 100, (
            f"prompt_tokens={prompt_tokens} seems too low for a message with an image. "
            "Image token cost may not be accounted for in truncation."
        )
