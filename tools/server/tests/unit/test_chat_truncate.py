import re
import pytest
import time
from utils import *
from test_vision_api import get_img_url

server: ServerProcess

IMAGE_URI_0 = get_img_url("IMG_BASE64_URI_0")

SYSTEM = "You are a helpful assistant."
FINAL_USER = "[U Last]This is the most recent user message."
N_TURNS_OVERFLOW = 95
MAX_COMPLETION_TOKENS = 1

# ── Benchmark model registry ──────────────────────────────────────────────────
# Each entry drives one parametrized run of test_chat_truncate_timings_massive.
# Fields:
#   hf_repo          – "<org>/<repo>:<quant>" downloaded by the test runner
#   alias            – short name used in output file names
#   n_ctx            – phase-1 (no-truncation) context; must fit the largest conversation
#   include_image    – True → attach one image per user turn (requires a vision model + mmproj)
#   mmproj_hf_repo   – HF repo for the mmproj GGUF; None for text-only models
BENCHMARK_MODELS = {
    "gemma3-1b": {
        "hf_repo":          "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M",
        "alias":            "gemma3-1b",
        "n_ctx":            128_000,
        "include_image":    False,
        "mmproj_hf_repo":   None,
    },
    "tinygemma3": {
        "hf_repo":          "ggml-org/tinygemma3-GGUF:Q8_0",
        "alias":            "tinygemma3",
        "n_ctx":            135_506,
        "include_image":    True,
        "mmproj_hf_repo":   "ggml-org/tinygemma3-GGUF",
    },
    "gemma3-4b": {
        "hf_repo":          "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
        "alias":            "gemma3-4b",
        "n_ctx":            32_768,    # effective max for this GGUF (128K requires RoPE scaling params)
        "include_image":    False,
        "mmproj_hf_repo":   "ggml-org/gemma-3-4b-it-GGUF",
    },
}
# ─────────────────────────────────────────────────────────────────────────────

def _user_msg(i: int, include_image: bool = False, repeat: int = 1) -> str|list[dict]:
    content = f"[U{i:03d}] Please explain topic {i:03d} in detail." * repeat
    if include_image:
        content = [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": IMAGE_URI_0}},
                ]
    return content

def _asst_msg(i: int, repeat: int = 1) -> str:
    return f"[A{i:03d}] Here is my explanation of topic {i:03d}." * repeat


def _get_messages(n_turns: int = 128, include_final_user: bool = True, include_image: bool = False, repeat: int = 1) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(1, n_turns + 1):
        msgs.append({"role": "user",      "content": _user_msg(i, include_image=include_image, repeat=repeat)})
        msgs.append({"role": "assistant", "content": _asst_msg(i, repeat=repeat)})
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
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": _get_messages(),
    })
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_context_size_error"


def test_chat_truncate_prevents_overflow():
    """
    With --chat-truncate set, long conversation succeeds.
    """
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": _get_messages(N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200


def test_chat_truncate_no_op():
    """
    A short conversation already below the truncation target is left untouched
    """
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": _short_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert "[U001]" in prompt, "No turn should not be dropped"
    assert_turns_consistency_in_prompt(prompt)


def test_chat_truncate_prompt_within_budget():
    """
    After truncation, prompt_tokens must be strictly less than the truncation
    target: floor(fraction * per_slot_ctx).
    """
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200
    assert server.n_ctx is not None and server.n_slots is not None
    per_slot_ctx = server.n_ctx // server.n_slots
    target = int(server.chat_truncate_max_keep * per_slot_ctx)
    assert res.body["usage"]["prompt_tokens"] < target


def test_chat_truncate_drops_oldest_keeps_newest():
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert _user_msg(1) not in prompt
    assert_turns_consistency_in_prompt(prompt)

def test_chat_truncate_non_user_newest_preserved():
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
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
        max_completion_tokens=5   -> n_predict = 1                      -> threshold = 256 - 1   = 255

    We first probe to find n_turns so that the true token count T satisfies:

        target (192) < T < threshold_with_max1 (255)

    Then, with T turns:
    - Without max_completion_tokens: threshold=192 < T -> truncation fires, prompt < target
    - With max_completion_tokens=5:  threshold=255 > T -> truncation silent, prompt unchanged
    """
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None and server.n_predict is not None
    max_completion_tokens = 1
    per_slot_ctx = server.n_ctx // server.n_slots
    threshold_no_max_completion_tokens = per_slot_ctx - server.n_predict
    threshold_with_max1 = per_slot_ctx - max_completion_tokens
    target = int(server.chat_truncate_max_keep * per_slot_ctx)

    assert threshold_no_max_completion_tokens < target < threshold_with_max1

    # Find n_turns so that target < T < threshold_with_max1
    found_n_turns  = None
    found_n_tokens = None
    for n_turns in range(0, 10):
        probe = server.make_request("POST", "/chat/completions", data={
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "messages": _get_messages(n_turns),
        })
        if probe.status_code != 200:
            break
        pt = probe.body["usage"]["prompt_tokens"]
        if target < pt < threshold_with_max1:
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

    # With max_completion_tokens=5: threshold=255 > T, then truncation silent
    res_max5 = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
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
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.99
    server.n_predict = n_predict_req
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None
    per_slot_ctx    = server.n_ctx // server.n_slots
    fraction_target = int(server.chat_truncate_max_keep * per_slot_ctx)
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
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.001
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


def test_chat_truncate_max_keep_zero_rejected():
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.0
    with pytest.raises(RuntimeError):
        server.start()


def test_chat_truncate_max_keep_one_rejected():
    global server
    server.chat_truncate = True
    server.chat_truncate_max_keep = 1.0
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
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.sleep_idle_seconds = 2
    server.debug = True
    server.start()

    # First request before sleep - should work
    res1 = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
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
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": _get_messages(N_TURNS_OVERFLOW),
    })
    assert res2.status_code == 200
    prompt = res2.body["__verbose"]["prompt"]
    assert "[U001]" not in prompt, "First turn should be dropped"

    # Verify server woke up
    res_props = server.make_request("GET", "/props")
    assert res_props.status_code == 200
    assert res_props.body["is_sleeping"] == False



def test_chat_truncate_multimodal_timings():
    global server
    # Use tinygemma3 for multimodal
    server = ServerPreset.tinygemma3()
    server.jinja = True
    server.n_ctx = 512
    server.n_slots = 1
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.start()

    timings = []

    for n_turns in [1, 4, 10, 20, 30, 50, N_TURNS_OVERFLOW]:
        msgs = _get_messages(n_turns=n_turns, include_final_user=False, include_image=True)
      
        t0 = time.time()
        res = server.make_request("POST", "/chat/completions", data={
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "messages": msgs,
        })

        timing = {"n_turns": n_turns, "status": res.status_code, "time": time.time()-t0}
        timings.append(timing)
        print(timing)

        # Truncation not triggered if an media is present
        assert res.status_code == 200
    
    print("Timings:")
    for t in timings:
        print(t)
        assert t["time"] < 1.0

def test_chat_truncate_not_multimodal_timings():
    global server
    # Use tinygemma3 for multimodal
    server = ServerPreset.tinygemma3()
    server.jinja = True
    server.n_ctx = 512
    server.n_slots = 1
    server.chat_truncate = True
    server.chat_truncate_max_keep = 0.8
    server.start()

    timings = []

    for n_turns in [1, 4, 10, 20, 30, 50, N_TURNS_OVERFLOW]:

        msgs = _get_messages(n_turns=n_turns, include_final_user=False)
        t0 = time.time()
        res = server.make_request("POST", "/chat/completions", data={
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "messages": msgs,
        })

        timing = {"n_turns": n_turns, "status": res.status_code, "time": time.time()-t0}
        timings.append(timing)

        assert res.status_code == 200

    print("Timings:")
    for t in timings:
        print(t)
        assert t["time"] < 1.0


@pytest.mark.parametrize("benchmark_key", list(BENCHMARK_MODELS.keys()))
def test_chat_truncate_not_multimodal_timings_massive(benchmark_key):
    """
    Each turn is repeat=50x larger than the baseline. Measures latency against
    the amount of truncation done: tokens_in (full conversation), tokens_out
    (after truncation), and turns_removed.

    Binary search truncation is O(N log N): log N steps each tokenizing ~N tokens.
    The timing therefore scales linearly with tokens_in, dominated by tokenization
    of the full payload, not by the number of search iterations.

    Phase 1: no-truncation server with large n_ctx to measure tokens_in.
    Phase 2: truncation server to measure latency, tokens_out, turns_removed.

    Models are defined in BENCHMARK_MODELS above; add entries there to extend.
    """
    global server

    BENCHMARK_MODEL = BENCHMARK_MODELS[benchmark_key]

    REPEAT        = 50
    include_image = BENCHMARK_MODEL["include_image"]
    n_turns_list  = [1, 5, 10, 50, 100, 200]

    def _make_server(**kwargs) -> ServerProcess:
        s = ServerProcess()
        s.offline       = False
        s.model_hf_repo = BENCHMARK_MODEL["hf_repo"]
        s.model_hf_file = None
        s.model_alias   = BENCHMARK_MODEL["alias"]
        s.jinja         = True
        s.n_slots       = 1
        if include_image and BENCHMARK_MODEL["mmproj_hf_repo"]:
            s.mmproj_url = f"hf:{BENCHMARK_MODEL['mmproj_hf_repo']}"
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    # Phase 1: measure full token counts without truncation.
    server = _make_server(n_ctx=BENCHMARK_MODEL["n_ctx"], chat_truncate=False, debug=True)
    server.start()

    # tokens_in: {"count": int|None, "ctx_exceeded": bool}
    tokens_in = {}
    for n_turns in n_turns_list:
        msgs = _get_messages(n_turns=n_turns, include_final_user=False, include_image=include_image, repeat=REPEAT)
        res = server.make_request("POST", "/chat/completions", data={
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "messages": msgs,
        })
        if res.status_code == 200:
            tokens_in[n_turns] = {"count": res.body["usage"]["prompt_tokens"], "ctx_exceeded": False}
        else:
            tokens_in[n_turns] = {"count": res.body.get("error", {}).get("n_prompt_tokens"), "ctx_exceeded": True}

    server.stop()

    # Phase 2: timing with truncation.
    server = _make_server(n_ctx=2048, chat_truncate=True, chat_truncate_max_keep=0.8, debug=True)
    server.start()

    model = server.model_alias
    timings = []

    for n_turns in n_turns_list:
        msgs = _get_messages(n_turns=n_turns, include_final_user=False, include_image=include_image, repeat=REPEAT)
        t0 = time.time()
        res = server.make_request("POST", "/chat/completions", data={
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "messages": msgs,
        })
        elapsed = time.time() - t0
        assert res.status_code == 200

        tokens_out = res.body["usage"]["prompt_tokens"]
        prompt = res.body.get("__verbose", {}).get("prompt", "")
        # Count distinct turn indices remaining (e.g. [U001], [U002]) — content is
        # repeated `repeat` times per turn so we use a set to deduplicate.
        turns_remaining = len(set(re.findall(r'\[U(\d+)\]', prompt)))
        timings.append({
            "model":         model,
            "n_turns":       n_turns,
            "tokens_in":     tokens_in[n_turns]["count"],
            "ctx_exceeded":  tokens_in[n_turns]["ctx_exceeded"],
            "tokens_out":    tokens_out,
            "turns_removed": n_turns - turns_remaining,
            "time":          elapsed,
        })

    print("Timings (massive):")
    with open(f"chat_truncate_not_multimodal_timings_massive_{model}.json", "w") as f:
        json.dump(timings, f, indent=2)
    for t in timings:
        print(t)
        assert t["time"] < 1.0


@pytest.mark.parametrize("benchmark_key", list(BENCHMARK_MODELS.keys()))
def test_chat_truncate_apply_template_timings_massive(benchmark_key):
    """
    Like test_chat_truncate_not_multimodal_timings_massive but uses /apply-template
    instead of /chat/completions, so the timer measures ONLY truncation + template
    rendering — no model inference cost.

    Phase 1: /apply-template without truncation (large n_ctx) to get the full rendered
             prompt; /tokenize it for an exact tokens_in count.
    Phase 2: /apply-template with truncation; time the call; /tokenize the response
             prompt for tokens_out.
    """
    global server

    BENCHMARK_MODEL = BENCHMARK_MODELS[benchmark_key]

    REPEAT        = 50
    include_image = BENCHMARK_MODEL["include_image"]
    n_turns_list  = [1, 5, 10, 50, 100, 200]

    def _make_server(**kwargs) -> ServerProcess:
        s = ServerProcess()
        s.offline       = False
        s.model_hf_repo = BENCHMARK_MODEL["hf_repo"]
        s.model_hf_file = None
        s.model_alias   = BENCHMARK_MODEL["alias"]
        s.jinja         = True
        s.n_slots       = 1
        if include_image and BENCHMARK_MODEL["mmproj_hf_repo"]:
            s.mmproj_url = f"hf:{BENCHMARK_MODEL['mmproj_hf_repo']}"
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    # Phase 1: exact token counts via /apply-template + /tokenize (no inference).
    server = _make_server(n_ctx=BENCHMARK_MODEL["n_ctx"], chat_truncate=False)
    server.start()

    tokens_in = {}
    for n_turns in n_turns_list:
        msgs = _get_messages(n_turns=n_turns, include_final_user=False, include_image=include_image, repeat=REPEAT)
        res = server.make_request("POST", "/apply-template", data={"messages": msgs})
        if res.status_code == 200:
            prompt_text = res.body["prompt"]
            tok = server.make_request("POST", "/tokenize", data={"content": prompt_text, "add_special": False})
            tokens_in[n_turns] = {"count": len(tok.body["tokens"]), "ctx_exceeded": False}
        else:
            tokens_in[n_turns] = {"count": None, "ctx_exceeded": True}

    server.stop()

    # Phase 2: timing with truncation — /apply-template only, no inference.
    server = _make_server(n_ctx=2048, chat_truncate=True, chat_truncate_max_keep=0.8)
    server.start()

    model = server.model_alias
    timings = []

    for n_turns in n_turns_list:
        msgs = _get_messages(n_turns=n_turns, include_final_user=False, include_image=include_image, repeat=REPEAT)
        t0 = time.time()
        res = server.make_request("POST", "/apply-template", data={"messages": msgs})
        elapsed = time.time() - t0
        assert res.status_code == 200

        prompt = res.body["prompt"]
        turns_remaining = len(set(re.findall(r'\[U(\d+)\]', prompt)))
        tok = server.make_request("POST", "/tokenize", data={"content": prompt, "add_special": False})
        tokens_out = len(tok.body["tokens"])
        timings.append({
            "model":         model,
            "n_turns":       n_turns,
            "tokens_in":     tokens_in[n_turns]["count"],
            "ctx_exceeded":  tokens_in[n_turns]["ctx_exceeded"],
            "tokens_out":    tokens_out,
            "turns_removed": n_turns - turns_remaining,
            "time":          elapsed,
        })

    print("Timings (apply-template, massive):")
    fname = f"chat_truncate_apply_template_timings_massive_{model}.json"
    with open(fname, "w") as f:
        json.dump(timings, f, indent=2)
    for t in timings:
        print(t)


if __name__ == "__main__":
    test_chat_truncate_apply_template_timings_massive("gemma3-4b")