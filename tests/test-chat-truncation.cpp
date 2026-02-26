// Tests for common_chat_truncate_messages.
//
// Usage: test-chat-truncation <vocab.gguf>
// e.g.:  test-chat-truncation models/ggml-vocab-llama-bpe.gguf

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ChatML template — same string used in test-chat-template.cpp
static const char * CHATML_TMPL =
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

// Strict template that raises an exception when a "tool" message is not
// immediately preceded by an "assistant" message with at least one tool call.
// This mirrors the validation that real model templates (e.g. Mistral Nemo)
// perform and is used to prove that a bad message sequence produces an error.
static const char * STRICT_TOOL_TMPL =
    "{%- set ns = namespace(prev_has_tool_calls=false) %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'tool' and not ns.prev_has_tool_calls %}"
    "{{ raise_exception('Orphaned tool message: not preceded by an assistant with tool_calls') }}"
    "{%- endif %}"
    "{%- set ns.prev_has_tool_calls = message.tool_calls is defined and message.tool_calls | length > 0 %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "  FAIL: %s\n", msg);
        std::exit(1);
    }
    fprintf(stdout, "  pass: %s\n", msg);
}

static common_chat_templates_inputs build_inputs(
        const std::vector<std::pair<std::string, std::string>> & msgs) {
    common_chat_templates_inputs inp;
    inp.use_jinja             = true;
    inp.add_generation_prompt = true;
    for (const auto & [role, content] : msgs) {
        common_chat_msg m;
        m.role    = role;
        m.content = content;
        inp.messages.push_back(m);
    }
    return inp;
}

// Count tokens for a rendered prompt
static int32_t count_tokens(
        const common_chat_templates * tmpls,
        const llama_vocab           * vocab,
        const common_chat_templates_inputs & inp) {
    auto result = common_chat_templates_apply(tmpls, inp);
    return (int32_t)common_tokenize(vocab, result.prompt, /*add_special=*/true, /*parse_special=*/true).size();
}

// Returns true if any "tool" message is not immediately preceded by an
// "assistant" message that carries at least one tool call.  Such a message
// is "orphaned": the assistant turn that issued the call was already removed
// by truncation, making the conversation semantically invalid.
static bool has_orphaned_tool_msg(const common_chat_templates_inputs & inp) {
    for (size_t i = 0; i < inp.messages.size(); ++i) {
        if (inp.messages[i].role != "tool") {
            continue;
        }
        bool preceded_by_caller = (i > 0)
            && (inp.messages[i - 1].role == "assistant")
            && (!inp.messages[i - 1].tool_calls.empty());
        if (!preceded_by_caller) {
            return true;
        }
    }
    return false;
}

static std::string render_prompt(
        const common_chat_templates * tmpls,
        const common_chat_templates_inputs & inp) {
    return common_chat_templates_apply(tmpls, inp).prompt;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_noop(
        const common_chat_templates * tmpls,
        const llama_vocab * vocab) {
    fprintf(stdout, "Test 1: no-op — prompt already fits within budget\n");

    auto inp       = build_inputs({{"user", "Hello."}, {"assistant", "Hi!"}, {"user", "Bye."}});
    int32_t toks   = count_tokens(tmpls, vocab, inp);
    size_t  n_orig = inp.messages.size();

    // n_ctx_slot >> toks, small n_predict → budget >> toks → no trigger
    common_chat_truncate_messages(inp, tmpls, vocab, common_chat_max_prompt_tokens(toks * 10, 1, 0.8f));

    check(inp.messages.size() == n_orig, "message count unchanged");
}

static void test_basic_truncation(
        const common_chat_templates * tmpls,
        const llama_vocab * vocab) {
    fprintf(stdout, "Test 2: oldest turn (user + reply) removed atomically; system and later turns preserved\n");

    // A turn is: the first user message plus all messages up to (not including) the next user message.
    auto inp = build_inputs({
        {"system",    "Be helpful."},
        {"user",      "Turn one."},       // oldest turn — will be dropped
        {"assistant", "Answer one."},     // part of turn one — dropped with it
        {"user",      "Turn two."},
        {"assistant", "Answer two."},
        {"user",      "Turn three."},
    });

    int32_t toks = count_tokens(tmpls, vocab, inp);

    // Force trigger: budget = toks/2 < toks → first turn must be dropped
    int32_t n_ctx  = toks;
    int32_t n_pred = toks / 2;
    float   frac   = 0.9f;

    std::string prompt_before = render_prompt(tmpls, inp);
    common_chat_truncate_messages(inp, tmpls, vocab, common_chat_max_prompt_tokens(n_ctx, n_pred, frac));
    std::string prompt_after  = render_prompt(tmpls, inp);

    fprintf(stderr, "  [KV-refresh] prompt before (%d chars):\n    %s\n",
            (int)prompt_before.size(), prompt_before.c_str());
    fprintf(stderr, "  [KV-refresh] prompt after  (%d chars):\n    %s\n",
            (int)prompt_after.size(), prompt_after.c_str());

    check(inp.messages[0].role == "system",            "system message preserved at index 0");
    check(inp.messages[1].role == "user",              "second message is now 'Turn two'");
    check(inp.messages[1].content == "Turn two.",      "oldest user turn removed, next user turn is now first");
    check(inp.messages.back().content == "Turn three.", "last user turn preserved");
    check(inp.messages.size() == 4,                    "turn one (user + assistant) dropped; 4 messages remain");
    check(prompt_before != prompt_after,               "prompt changed → KV cache must be refreshed");
}

static void test_n_predict_unlimited(
        const common_chat_templates * tmpls,
        const llama_vocab * vocab) {
    fprintf(stdout, "Test 3: n_predict=-1 triggers when n_tokens > target (unlike n_predict=1)\n");

    auto inp = build_inputs({
        {"user",      "Message alpha, which takes several tokens to represent."},
        {"assistant", "Reply alpha, also takes several tokens to represent."},
        {"user",      "Message beta, which takes several tokens to represent."},
        {"assistant", "Reply beta, also takes several tokens to represent."},
        {"user",      "Short final question."},
    });

    int32_t toks = count_tokens(tmpls, vocab, inp);

    // n_ctx_slot = 2 * toks  →  budget (n_predict=1) = 2*toks - 1  >>  toks  →  no trigger
    // target = 0.4 * 2*toks = 0.8 * toks  <  toks  →  trigger when n_predict=-1
    int32_t n_ctx  = toks * 2;
    float   frac   = 0.4f;
    int32_t target = (int32_t)(frac * (float)n_ctx);

    check(target < toks, "test setup: target < toks");

    // --- n_predict=1: budget = n_ctx - 1 >> toks → no truncation, no KV refresh ---
    {
        auto copy          = inp;
        std::string before = render_prompt(tmpls, copy);
        common_chat_truncate_messages(copy, tmpls, vocab, common_chat_max_prompt_tokens(n_ctx, 1, frac));

        check(copy.messages.size() == inp.messages.size(),
              "n_predict=1: no truncation (budget >> n_tokens)");
        check(render_prompt(tmpls, copy) == before,
              "n_predict=1: prompt unchanged (no KV refresh needed)");
    }

    // --- n_predict=-1: trigger = target < toks → truncation fires, KV cache refreshed ---
    {
        auto copy          = inp;
        std::string before = render_prompt(tmpls, copy);
        common_chat_truncate_messages(copy, tmpls, vocab, common_chat_max_prompt_tokens(n_ctx, -1, frac));

        std::string after = render_prompt(tmpls, copy);
        fprintf(stderr, "  [KV-refresh] prompt before (%d chars):\n    %s\n",
                (int)before.size(), before.c_str());
        fprintf(stderr, "  [KV-refresh] prompt after  (%d chars):\n    %s\n",
                (int)after.size(), after.c_str());

        check(copy.messages.size() < inp.messages.size(),
              "n_predict=-1: truncation fired");
        check(after != before,
              "n_predict=-1: prompt changed → KV cache must be refreshed");
        check(count_tokens(tmpls, vocab, copy) <= target,
              "n_predict=-1: final token count ≤ target");
    }
}

static void test_multi_message_turn_removed_atomically(
        const common_chat_templates * tmpls,
        const llama_vocab * vocab) {
    fprintf(stdout, "Test 4: full turn (user + assistant-tc + tool-result + assistant-reply) removed atomically\n");

    // Turn 0: user0 → assistant-with-tool-call → tool-result → assistant-reply
    // Turn 1: user1  (must survive)
    // All of turn 0 must be removed as one unit so no orphaned tool messages remain.
    auto inp = build_inputs({
        {"user",      "Long question that dominates the token budget for truncation purposes."},
        {"assistant", ""},          // assistant with tool call — added below
        {"tool",      R"({"result": 42})"},
        {"assistant", "The answer is 42."},
        {"user",      "Thanks!"},   // must survive
    });
    // Attach a tool call to the assistant at index 1
    {
        common_chat_tool_call tc;
        tc.name      = "compute";
        tc.arguments = "{}";
        tc.id        = "call_1";
        inp.messages[1].tool_calls.push_back(tc);
    }

    int32_t toks = count_tokens(tmpls, vocab, inp);

    // Force trigger so that one removal is enough to fit
    common_chat_truncate_messages(inp, tmpls, vocab, common_chat_max_prompt_tokens(toks, /*n_predict=*/1, /*frac=*/0.9f));

    fprintf(stderr, "  remaining messages after truncation:\n");
    for (size_t i = 0; i < inp.messages.size(); ++i) {
        fprintf(stderr, "    [%zu] role=%s\n", i, inp.messages[i].role.c_str());
    }

    check(inp.messages.size() == 1,               "only user1 remains");
    check(inp.messages[0].content == "Thanks!",   "user1 content preserved");
    check(!has_orphaned_tool_msg(inp),             "no orphaned tool messages");
}

static void test_stop_when_no_user_messages(
        const common_chat_templates * tmpls,
        const llama_vocab * vocab) {
    fprintf(stdout, "Test 5: loop stops when no more user messages to remove\n");

    // After removing user0's turn, only [system, user1] remain.
    // Next iteration finds user1 as the new first user but after removing it
    // the inner while loop would go out of bounds — so the outer loop must
    // stop before that happens (first_user_msg not found → break).
    // This test verifies truncation stops at [system, user1] when budget allows.
    auto inp = build_inputs({
        {"system",    "Sys."},
        {"user",      "Turn one, long enough to trigger truncation on its own."},
        {"assistant", "Answer one."},
        {"user",      "Short."},
    });

    int32_t toks     = count_tokens(tmpls, vocab, inp);
    int32_t toks_all = toks;

    // Remove just the first turn; after that the remaining tokens should fit.
    // We set max_prompt_tokens to just below the full count so exactly one removal fires.
    common_chat_truncate_messages(inp, tmpls, vocab, toks_all - 1);

    check(inp.messages[0].role == "system",    "system preserved");
    check(inp.messages.back().content == "Short.", "last user turn preserved");
    check(inp.messages.size() == 2,            "turn one (user + assistant) removed; system + user1 remain");
}

// Test 0 — proves that passing a bad message sequence (orphaned tool message)
// to a strict template raises an exception, independent of any truncation logic.
static void test_strict_template_rejects_orphaned_tool_msg() {
    fprintf(stdout, "Test 0: strict template raises on orphaned tool message\n");

    auto strict_ptr = common_chat_templates_ptr(
        common_chat_templates_init(/* model= */ nullptr, STRICT_TOOL_TMPL));
    const common_chat_templates * strict = strict_ptr.get();

    auto try_render = [&](common_chat_templates_inputs inp) -> bool {
        try {
            common_chat_templates_apply(strict, inp);
            return false; // no exception
        } catch (const std::exception &) {
            return true;  // exception thrown
        }
    };

    auto make_plain = [](const std::string & role, const std::string & content) {
        common_chat_msg m;
        m.role    = role;
        m.content = content;
        return m;
    };

    auto make_tool_caller = [](const std::string & name) {
        common_chat_msg m;
        m.role    = "assistant";
        m.content = "";
        common_chat_tool_call tc;
        tc.name      = name;
        tc.arguments = "{}";
        tc.id        = "call_1";
        m.tool_calls.push_back(tc);
        return m;
    };

    // --- good sequence: assistant-with-tool-call immediately before tool ---
    {
        common_chat_templates_inputs inp;
        inp.use_jinja             = true;
        inp.add_generation_prompt = false;
        inp.messages.push_back(make_plain("user",      "What is the weather?"));
        inp.messages.push_back(make_tool_caller("get_weather"));
        inp.messages.push_back(make_plain("tool",      R"({"temp":22})"));
        inp.messages.push_back(make_plain("assistant", "It is 22 C."));

        check(!try_render(inp),
              "valid sequence (assistant-tc then tool) renders without error");
    }

    // --- bad sequence: tool message with no preceding assistant-with-tool-calls ---
    {
        common_chat_templates_inputs inp;
        inp.use_jinja             = true;
        inp.add_generation_prompt = false;
        inp.messages.push_back(make_plain("tool",      R"({"temp":22})"));  // orphaned
        inp.messages.push_back(make_plain("assistant", "It is 22 C."));

        check(try_render(inp),
              "orphaned tool message raises an error");
    }
}

// Test 6 — verifies that truncation never produces an orphaned "tool" message.
// The full turn (user + assistant-with-tool-call + tool-result + assistant-reply)
// must be removed as one atomic unit so that no tool message is left without
// its preceding assistant-with-tool-calls.
static void test_tool_call_orphan_after_truncation(
        const common_chat_templates * tmpls,
        const llama_vocab           * vocab) {
    fprintf(stdout, "Test 6: truncation must not orphan tool messages\n");

    // Build the conversation:
    //   user0 (long)  →  assistant_tc (tool call)  →  tool_result
    //   →  assistant_reply  →  user1 (short, must be preserved)
    //
    // user0 is intentionally long so that removing (user0 + assistant_tc)
    // drops the token count below target in one iteration, causing the loop
    // to stop with the tool_result message still at the front.
    common_chat_templates_inputs inp;
    inp.use_jinja             = true;
    inp.add_generation_prompt = true;

    // user0 — long enough to dominate token count
    {
        common_chat_msg m;
        m.role    = "user";
        m.content = "What is the weather forecast for the next ten days in Paris, "
                    "including temperature highs and lows, precipitation probability, "
                    "wind speed, humidity levels, and UV index? "
                    "Please provide the information in a structured table format.";
        inp.messages.push_back(m);
    }
    // assistant_tc — calls the weather tool (content intentionally empty)
    {
        common_chat_msg m;
        m.role    = "assistant";
        m.content = "";
        common_chat_tool_call tc;
        tc.name      = "get_weather";
        tc.arguments = R"({"city": "Paris", "days": 10})";
        tc.id        = "call_abc123";
        m.tool_calls.push_back(tc);
        inp.messages.push_back(m);
    }
    // tool_result — response from get_weather
    {
        common_chat_msg m;
        m.role         = "tool";
        m.content      = R"({"forecast": [{"day": 1, "high": 22, "low": 14}]})";
        m.tool_call_id = "call_abc123";
        m.tool_name    = "get_weather";
        inp.messages.push_back(m);
    }
    // assistant_reply — uses the tool result
    {
        common_chat_msg m;
        m.role    = "assistant";
        m.content = "Here is the 10-day weather forecast for Paris.";
        inp.messages.push_back(m);
    }
    // user1 — short follow-up that must survive truncation
    {
        common_chat_msg m;
        m.role    = "user";
        m.content = "Thanks!";
        inp.messages.push_back(m);
    }

    int32_t toks = count_tokens(tmpls, vocab, inp);

    // Trigger truncation: n_ctx = toks, n_predict = 1  →  max_prompt = toks-1
    // so the prompt never fits and the while loop fires immediately.
    // target = 0.9 * toks: user0 alone is well over 10 % of total tokens,
    // so removing (user0 + assistant_tc) drops below target after one
    // iteration, leaving tool_result orphaned at index 0.
    common_chat_truncate_messages(inp, tmpls, vocab, common_chat_max_prompt_tokens(toks, /*n_predict=*/1, /*frac=*/0.9f));

    // Print remaining sequence to make the failure easy to diagnose
    fprintf(stderr, "  remaining messages after truncation:\n");
    for (size_t i = 0; i < inp.messages.size(); ++i) {
        fprintf(stderr, "    [%zu] role=%-12s tool_calls=%zu\n",
                i, inp.messages[i].role.c_str(), inp.messages[i].tool_calls.size());
    }

    check(!has_orphaned_tool_msg(inp), "no orphaned tool messages after truncation");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vocab.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mp       = llama_model_default_params();
    mp.vocab_only = true;

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "Failed to load vocab from '%s'\n", argv[1]);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto tmpls_ptr = common_chat_templates_ptr(
        common_chat_templates_init(/* model= */ nullptr, CHATML_TMPL));
    const common_chat_templates * tmpls = tmpls_ptr.get();

    test_strict_template_rejects_orphaned_tool_msg();
    test_noop                              (tmpls, vocab);
    test_basic_truncation                  (tmpls, vocab);
    test_n_predict_unlimited               (tmpls, vocab);
    test_multi_message_turn_removed_atomically(tmpls, vocab);
    test_stop_when_no_user_messages        (tmpls, vocab);
    test_tool_call_orphan_after_truncation (tmpls, vocab);

    llama_model_free(model);
    llama_backend_free();

    fprintf(stdout, "\n[test-chat-truncation] All tests passed!\n");
    return 0;
}
