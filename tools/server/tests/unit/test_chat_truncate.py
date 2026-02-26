import pytest
from utils import *

server: ServerProcess

# ── shared content ────────────────────────────────────────────────────────────

SYSTEM = "You are a helpful assistant."
FINAL_USER = "[U Last]This is the most recent user message."

# Identifiable labels let us verify which turns survived in __verbose.prompt.
def _user_msg(i: int) -> str:
    return f"[U{i:02d}] Please explain topic {i} in detail."

def _asst_msg(i: int) -> str:
    return f"[A{i:02d}] Here is my explanation of topic {i}."


def _get_messages(n_turns: int = 128) -> list[dict]:
    """
    Build a multi-turn conversation long enough to overflow the per-slot context
    (tinyllama2 preset: n_ctx=512, n_slots=2 → 256 tokens per slot).

    With 128 turns the rendered prompt is well above 256 tokens, so it triggers
    both the context-exceeded error (no truncation) and the truncation logic
    (chat_truncate enabled).
    """
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(1, n_turns + 1):
        msgs.append({"role": "user",      "content": _user_msg(i)})
        msgs.append({"role": "assistant", "content": _asst_msg(i)})
    msgs.append({"role": "user", "content": FINAL_USER})
    return msgs


def _short_messages() -> list[dict]:
    """
    A tiny conversation that fits comfortably below the truncation target
    (floor(fraction * 256) tokens).  Used to verify the no-op path.
    """
    return _get_messages(n_turns=1)

def assert_turns_consistency_in_prompt(prompt: str):
    """
    Verify that the user and assistant turns in the rendered prompt are consistent
    with the chat template, i.e. each user turn is followed by an assistant turn,
    and the final user message is present.
    And that no assistant message appears without its preceding user message (which would indicate a template breakage due to truncation).
    """
    turns = prompt.split("\n")
    user_turns = [t for t in turns if t.startswith("[U")]
    asst_turns = [t for t in turns if t.startswith("[A")]
    assert len(user_turns) == len(asst_turns) + 1, "Each assistant turn should be preceded by a user turn, and there should be one extra user turn at the end"
    assert SYSTEM in prompt, "The system message should be present in the prompt"
    assert FINAL_USER in prompt, "The final user message should be present in the prompt"
    for i in range(len(asst_turns)):
        expected_asst = _asst_msg(i + 1)
        if expected_asst in prompt:
            expected_user = _user_msg(i + 1)
            assert expected_user in prompt, f"User turn {i+1} should be present in the prompt as it precedes assistant turn {i+1}"

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # tinyllama2 preset: n_ctx=512, n_slots=2, n_predict=64
    # → per-slot context = 512 // 2 = 256 tokens
    # When max_tokens=5 in a request:
    #   get_n_predict_with_server_priority returns 5
    #   threshold = 256 - 5 = 251
    #   target    = floor(fraction * 256)
    # So target = 204 with fraction=0.8.
    server.jinja = True
    server.chat_template = "chatml"  # stable, predictable __verbose.prompt format


# ── tests ─────────────────────────────────────────────────────────────────────

def test_overflow_without_chat_truncate_flag():
    """
    Baseline: a long conversation exceeds the per-slot context when
    --chat-truncate is not set, returning a 400 exceed_context_size_error.
    """
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
        "messages": _get_messages(),
    })
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_context_size_error"


def test_chat_truncate_prevents_overflow():
    """
    With --chat-truncate set, the same long conversation that would otherwise
    overflow is silently trimmed and the request succeeds.
    """
    global server
    server.chat_truncate = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
        "messages": _get_messages(),
    })
    assert res.status_code == 200


def test_chat_truncate_no_op():
    """
    A short conversation already below the truncation target is left untouched:
    every message must still appear in the rendered prompt.
    """
    global server
    server.chat_truncate = 0.8
    server.debug = True  # enables __verbose in the response body
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
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

    With n_ctx=512, n_slots=2, fraction=0.8:
      per_slot_ctx = 256
      target       = floor(0.8 * 256) = 204
      threshold    = 256 - 5 = 251   (max_tokens=5 in the request)
    threshold > target, so the truncation loop is guaranteed to execute
    whenever chat_needs_truncation fires.
    """
    global server
    server.chat_truncate = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
        "messages": _get_messages(n_turns=100),  # even longer to ensure we exceed the target after truncation
    })
    assert res.status_code == 200
    assert server.n_ctx is not None and server.n_slots is not None 
    per_slot_ctx = server.n_ctx // server.n_slots  # 256
    target = int(0.8 * per_slot_ctx)               # 204
    assert res.body["usage"]["prompt_tokens"] < target


def test_chat_truncate_system_preserved():
    """
    Truncation must never remove the system message; it must appear verbatim
    in the rendered prompt.
    """
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
        "messages": _get_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert_turns_consistency_in_prompt(prompt)


def test_chat_truncate_drops_oldest_keeps_newest():
    """
    Truncation removes the oldest user turn first while always keeping the
    most recent user message.
    """
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
        "messages": _get_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert _user_msg(1) not in prompt, "Oldest user turn should be removed by truncation"
    assert_turns_consistency_in_prompt(prompt)

def test_chat_truncate_n_predict_threshold_vs_max_tokens():
    """
    The truncation threshold differs depending on whether max_tokens is in the request:

        no max_tokens  → n_predict = server.n_predict = 64  → threshold = 256 - 64  = 192
        max_tokens=5   → n_predict = 5                      → threshold = 256 - 5   = 251

    We first probe to find n_turns whose true token count T satisfies:

        target (204) < T < threshold_with_max5 (251)

    Using max_tokens=1 for the probe makes its own threshold = 255 > 251, so the
    probe itself never triggers truncation and gives us the true T.

    With that conversation:
    - Without max_tokens: threshold=192 < T → truncation fires, prompt < target
    - With max_tokens=5:  threshold=251 > T → truncation silent, prompt unchanged
    """
    global server
    server.chat_truncate = 0.8
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None and server.n_predict is not None
    per_slot_ctx       = server.n_ctx // server.n_slots            # 256
    threshold_no_max   = per_slot_ctx - server.n_predict           # 192
    threshold_with_max5 = per_slot_ctx - 5                         # 251
    target             = int(server.chat_truncate * per_slot_ctx)  # 204

    assert threshold_no_max < target < threshold_with_max5, (
        "precondition: 192 < 204 < 251"
    )

    # ── discovery: find n_turns with target < T < threshold_with_max5 ──────────
    found_n_turns  = None
    found_n_tokens = None
    for n_turns in range(1, 30):
        probe = server.make_request("POST", "/chat/completions", data={
            "max_tokens": 1,   # threshold=255 > 251, probe never truncates
            "messages": _get_messages(n_turns),
        })
        if probe.status_code != 200:
            break
        pt = probe.body["usage"]["prompt_tokens"]
        if target < pt < threshold_with_max5:
            found_n_turns  = n_turns
            found_n_tokens = pt
            break

    assert found_n_turns is not None, (
        f"Could not find n_turns with {target} < prompt_tokens < {threshold_with_max5} "
        f"in range 1-29"
    )

    msgs = _get_messages(found_n_turns)

    # ── without max_tokens: threshold=192 < T → truncation fires ───────────────
    res_no_max = server.make_request("POST", "/chat/completions", data={
        "messages": msgs,
    })
    assert res_no_max.status_code == 200
    assert res_no_max.body["usage"]["prompt_tokens"] < target, (
        f"Expected truncation to reduce prompt below target={target}, "
        f"got {res_no_max.body['usage']['prompt_tokens']}"
    )

    # ── with max_tokens=5: threshold=251 > T → truncation silent ───────────────
    res_max5 = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
        "messages": msgs,
    })
    assert res_max5.status_code == 200
    assert res_max5.body["usage"]["prompt_tokens"] == found_n_tokens, (
        f"Expected no truncation (prompt preserved at {found_n_tokens}), "
        f"got {res_max5.body['usage']['prompt_tokens']}"
    )


def test_chat_truncate_target_capped_to_budget():
    """
    When chat_truncate * n_ctx > n_ctx - n_predict (the generation budget),
    chat_truncate_target_tokens caps the truncation target at the budget.

    With chat_truncate=0.9, max_tokens=64 in the request (n_predict=64):
      fraction_target = floor(1 * 256) = 256
      budget          = 256 - 64         = 192
      actual_target   = min(230, 192)    = 192  ← capped

    The prompt after truncation must be < 192 (the budget cap), which is
    stricter than the uncapped fraction target of 230.
    """
    global server
    
    n_predict_req = 128
    server.chat_truncate = 0.99
    server.n_predict = n_predict_req
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None
    per_slot_ctx    = server.n_ctx // server.n_slots           # 256
    fraction_target = int(server.chat_truncate * per_slot_ctx) # 255  → exceeds budget
    budget          = per_slot_ctx - n_predict_req             # 128

    assert fraction_target > budget, "precondition: fraction target must exceed budget"

    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predict_req,
        "messages": _get_messages(),
    })
    assert res.status_code == 200
    # Prompt must be below the budget-capped target (128), not just the
    # fraction-only target (255), proving the cap was applied.
    assert res.body["usage"]["prompt_tokens"] < budget, (
        f"Expected prompt < budget={budget} (cap applied), "
        f"got {res.body['usage']['prompt_tokens']} (fraction target was {fraction_target})"
    )


def test_chat_truncate_very_low_fraction_preserves_last_user_msg():
    """
    With a very small chat_truncate fraction, the truncation threshold is 
    ctx * chat_truncate (no budget subtraction).

    With chat_truncate=0.2, n_predict=-1:
      threshold = target = floor(0.2 * 256) = 51

    The truncation loop drops oldest turns until prompt < 51 tokens, leaving
    [system + FINAL_USER] (≈ 30-35 tokens) intact — so FINAL_USER is preserved.
    """
    global server
    server.chat_truncate = 0.001   # 0 tokens
    server.n_predict = -1        # unlimited → threshold = ctx * fraction, no budget
    server.debug = True
    server.start()

    res = server.make_request("POST", "/chat/completions", data={
        "messages": _get_messages(),  # no max_tokens → n_predict stays -1
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    templated_sys_prompt = "<s> <|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    templated_last_message = "<|im_start|>user\n" + FINAL_USER + "<|im_end|>\n<|im_start|>assistant\n"
    assert templated_sys_prompt + templated_last_message == prompt, "System message should the only remaining content in the prompt"


def test_chat_truncate_negative_value_rejected():
    """
    --chat-truncate requires a value in (0, 1). A non-positive fraction is outside
    the valid range and must be rejected at server startup.
    """
    global server
    server.chat_truncate = 0.0
    with pytest.raises(RuntimeError):
        server.start()


def test_chat_truncate_above_one_rejected():
    """
    --chat-truncate requires a value in (0, 1).  A value greater or equal to 1 is
    outside the valid range and must be rejected at server startup.
    """
    global server
    server.chat_truncate = 1.0
    with pytest.raises(RuntimeError):
        server.start()
