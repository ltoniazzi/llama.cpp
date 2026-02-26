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
    common_chat_truncate_messages(inp, tmpls, vocab, toks * 10, 1, 0.8f);

    check(inp.messages.size() == n_orig, "message count unchanged");
}

static void test_basic_truncation(
        const common_chat_templates * tmpls,
        const llama_vocab * vocab) {
    fprintf(stdout, "Test 2: truncation drops oldest turns, preserves system + last user\n");

    auto inp = build_inputs({
        {"system",    "Be helpful."},
        {"user",      "Turn one."},
        {"assistant", "Answer one."},
        {"user",      "Turn two."},
        {"assistant", "Answer two."},
        {"user",      "Turn three."},   // last user — must be preserved
    });

    int32_t toks = count_tokens(tmpls, vocab, inp);

    // Force trigger: n_ctx_slot = toks, n_predict = toks/2 → budget = toks/2 < toks
    // target = 0.9 * toks — some turns need to be dropped to reach it
    int32_t n_ctx = toks;
    int32_t n_pred = toks / 2;
    float   frac  = 0.9f;

    std::string prompt_before = render_prompt(tmpls, inp);
    common_chat_truncate_messages(inp, tmpls, vocab, n_ctx, n_pred, frac);
    std::string prompt_after  = render_prompt(tmpls, inp);

    fprintf(stderr, "  [KV-refresh] prompt before (%d chars):\n    %s\n",
            (int)prompt_before.size(), prompt_before.c_str());
    fprintf(stderr, "  [KV-refresh] prompt after  (%d chars):\n    %s\n",
            (int)prompt_after.size(), prompt_after.c_str());

    check(inp.messages[0].role == "system",           "system message preserved at index 0");
    check(inp.messages.back().role == "user",          "last message is a user turn");
    check(inp.messages.back().content == "Turn three.", "last user content preserved");
    check(inp.messages.size() == 4,                     "some messages were dropped");
    check(prompt_before != prompt_after,               "prompt changed → KV cache must be refreshed");
    check(count_tokens(tmpls, vocab, inp) <= (int32_t)(frac * n_ctx), "final token count ≤ target");
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
        common_chat_truncate_messages(copy, tmpls, vocab, n_ctx, 1, frac);

        check(copy.messages.size() == inp.messages.size(),
              "n_predict=1: no truncation (budget >> n_tokens)");
        check(render_prompt(tmpls, copy) == before,
              "n_predict=1: prompt unchanged (no KV refresh needed)");
    }

    // --- n_predict=-1: trigger = target < toks → truncation fires, KV cache refreshed ---
    {
        auto copy          = inp;
        std::string before = render_prompt(tmpls, copy);
        common_chat_truncate_messages(copy, tmpls, vocab, n_ctx, -1, frac);

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

static void test_hard_floor(
        const common_chat_templates * tmpls,
        const llama_vocab * vocab) {
    fprintf(stdout, "Test 4: hard floor — system + last user always kept even when ctx=1\n");

    auto inp = build_inputs({
        {"system", "Sys."},
        {"user",   "Only question."},
    });

    // Ridiculously small ctx — would drop everything if unconstrained
    common_chat_truncate_messages(inp, tmpls, vocab, /*n_ctx=*/1, /*n_predict=*/-1, 0.5f);

    check(inp.messages.size() == 2,         "exactly 2 messages remain");
    check(inp.messages[0].role == "system", "first message is system");
    check(inp.messages[1].role == "user",   "second message is user");
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

    test_noop               (tmpls, vocab);
    test_basic_truncation   (tmpls, vocab);
    test_n_predict_unlimited(tmpls, vocab);
    test_hard_floor         (tmpls, vocab);

    llama_model_free(model);
    llama_backend_free();

    fprintf(stdout, "\n[test-chat-truncation] All tests passed!\n");
    return 0;
}
