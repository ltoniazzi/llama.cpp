import pytest
from utils import *

server: ServerProcess

# ── shared content ────────────────────────────────────────────────────────────

SYSTEM = "You are a helpful assistant."
FINAL_USER = "This is the most recent question."

# Identifiable labels let us verify which turns survived in __verbose.prompt.
def _user_msg(i: int) -> str:
    return f"[U{i:02d}] Please explain topic {i} in detail."

def _asst_msg(i: int) -> str:
    return f"[A{i:02d}] Here is my explanation of topic {i}."


def _long_messages(n_turns: int = 15) -> list[dict]:
    """
    Build a multi-turn conversation long enough to overflow the per-slot context
    (tinyllama2 preset: n_ctx=512, n_slots=2 → 256 tokens per slot).

    With 15 turns the rendered prompt is well above 256 tokens, so it triggers
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
    return [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": "[U01] Hi"},
        {"role": "assistant", "content": "[A01] Hello!"},
        {"role": "user",      "content": FINAL_USER},
    ]


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
    # Using fraction ≤ 251/256 ≈ 0.98 keeps target < threshold so the
    # truncation loop always executes whenever chat_needs_truncation fires.
    server.jinja = True
    server.chat_template = "chatml"  # stable, predictable __verbose.prompt format


# ── tests ─────────────────────────────────────────────────────────────────────

def test_chat_truncate_overflows_without_flag():
    """
    Baseline: a long conversation exceeds the per-slot context when
    --chat-truncate is not set, returning a 400 exceed_context_size_error.
    """
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 5,
        "messages": _long_messages(),
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
        "messages": _long_messages(),
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
    assert "[U01]" in prompt, "Only turn should not be dropped"
    assert FINAL_USER in prompt, "Last user message should be preserved"


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
        "messages": _long_messages(),
    })
    assert res.status_code == 200
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
        "messages": _long_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    assert SYSTEM in res.body["__verbose"]["prompt"]


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
        "messages": _long_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert _user_msg(1) not in prompt, "Oldest user turn should be removed by truncation"
    assert FINAL_USER in prompt, "Most recent user message should be preserved"
