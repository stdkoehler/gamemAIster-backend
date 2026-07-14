"""
Live LLM verification — hits real provider APIs, costs real tokens.

NOT part of the automated suite: every test here is skipped by default.
`pytest tests/` (the command documented in CLAUDE.md) will show these as
"skipped", never executed, so CI and routine local runs never spend tokens
on this file.

Run manually after changing anything in src/llmclient/llm_client.py
(new client, changed reasoning wiring, changed build_agent()/pydantic_ai
bridge, etc.) with:

    RUN_LIVE_LLM_TESTS=1 poetry run python -m pytest tests/test_llm_clients_live.py -v -s

Requires the relevant API_KEY_* variables in .env (loaded explicitly below,
since nothing else in this codebase loads it for a plain pytest run). Each
test additionally skips on its own if its specific key is missing, so a
partial .env still exercises whatever it can.

Each provider's model can be overridden via a *_MODEL env var — reusing
OPENROUTER_MODEL/LOCAL_MODEL, the same two build_gamemaster()
(src/brain/gamemaster.py) already reads, plus DEEPSEEK_MODEL/CLAUDE_MODEL/
MINIMAX_MODEL introduced here for the other three (gamemaster.py hardcodes
those three rather than reading them from the environment). Falls back to a
sensible reasoning-capable default per provider when unset. E.g. to check a
specific OpenRouter model:

    RUN_LIVE_LLM_TESTS=1 OPENROUTER_MODEL="qwen/qwen3-max" \\
        poetry run python -m pytest tests/test_llm_clients_live.py::test_openrouter_reasoning_and_structured_output -v -s

What each test actually checks: that `LLMClientBase.build_agent(...,
reasoning=True)` produces BOTH real extended-thinking output (a non-trivial
ThinkingPart) AND correctly validated structured output (the pydantic
model), for a client whose to_pydantic_ai_model()/build_agent() wiring
might silently break either half without raising — e.g. a client silently
sending a request parameter the provider ignores (no error, just missing
thinking), or a provider rejecting reasoning + forced tool-calling outright
(a hard 400). Plain pytest asserts on parsed/validated fields are not
enough to catch either failure mode; see docs/conversation_memory.html's
"Reasoning parity" and "Thinking vs. forced tool-calling" sections for the
two real bugs this exact check caught during development.
"""

import os
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai.messages import ThinkingPart

from src.llmclient.llm_client import (
    LLMClientBase,
    LLMClientClaude,
    LLMClientDeepSeek,
    LLMClientLocal,
    LLMClientLocalOpenAI,
    LLMClientMiniMax,
    LLMClientOpenRouter,
)
from src.llmclient.llm_config_registry import LLMTask

load_dotenv(Path(__file__).parent.parent / ".env")

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_LLM_TESTS"),
    reason=(
        "Live LLM test — costs real tokens, not run automatically. "
        "Set RUN_LIVE_LLM_TESTS=1 to run after changing llm_client.py."
    ),
)

# A word problem forces genuine multi-step reasoning (so a model that fakes
# thinking or skips it tends to get this wrong) while still having one
# unambiguous correct integer answer to assert on.
_PROMPT = (
    "A farmer has 17 sheep. All but 9 run away. Then 3 wander back, and half "
    "of the remaining flock (rounded down) is sold. How many sheep does the "
    "farmer have now? Think it through step by step before answering."
)
_EXPECTED_ANSWER = 6

_LOCAL_BASE_URL = "http://127.0.0.1:5000"


class _Answer(BaseModel):
    answer: int


def _assert_reasoning_and_structured_output(client: LLMClientBase, task: LLMTask) -> None:
    agent = client.build_agent(
        task,
        output_type=_Answer,
        system_prompt="You are a careful math assistant.",
        reasoning=True,
    )
    result = agent.run_sync(_PROMPT)

    assert result.output.answer == _EXPECTED_ANSWER, (
        f"Structured output wrong or unvalidated: got {result.output.answer!r}, "
        f"expected {_EXPECTED_ANSWER}"
    )

    thinking_chars = sum(
        len(part.content)
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if isinstance(part, ThinkingPart)
    )
    assert thinking_chars > 20, (
        "No meaningful ThinkingPart in the response — reasoning=True silently "
        "did nothing for this client (check to_pydantic_ai_model()'s reasoning "
        "wiring for this provider)."
    )


@pytest.mark.skipif(not os.getenv("API_KEY_DEEPSEEK"), reason="API_KEY_DEEPSEEK not set")
def test_deepseek_reasoning_and_structured_output():
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    client = LLMClientDeepSeek(api_key=os.environ["API_KEY_DEEPSEEK"], model=model)
    _assert_reasoning_and_structured_output(client, LLMTask.ARCHITECT)


@pytest.mark.skipif(not os.getenv("API_KEY_OPENROUTER"), reason="API_KEY_OPENROUTER not set")
def test_openrouter_reasoning_and_structured_output():
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
    client = LLMClientOpenRouter(api_key=os.environ["API_KEY_OPENROUTER"], model=model)
    _assert_reasoning_and_structured_output(client, LLMTask.ARCHITECT)


@pytest.mark.skipif(not os.getenv("API_KEY_MINIMAX"), reason="API_KEY_MINIMAX not set")
def test_minimax_reasoning_and_structured_output():
    model = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")
    client = LLMClientMiniMax(api_key=os.environ["API_KEY_MINIMAX"], model=model)
    _assert_reasoning_and_structured_output(client, LLMTask.ARCHITECT)


@pytest.mark.skipif(not os.getenv("API_KEY_CLAUDE"), reason="API_KEY_CLAUDE not set")
def test_claude_reasoning_and_structured_output():
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
    client = LLMClientClaude(api_key=os.environ["API_KEY_CLAUDE"], model=model)
    _assert_reasoning_and_structured_output(client, LLMTask.ARCHITECT)


def test_local_reasoning_and_structured_output():
    """
    LLMClientLocal doesn't use pydantic_ai at all — LLMClientLocal.build_agent()
    returns CompletionAgent, which drives structured output through
    chat_completion() (schema spelled out in the system prompt, JSON
    extracted and validated by hand, one retry on failure) instead. See
    CompletionAgent's docstring in src/llmclient/llm_client.py for why: real
    reasoning on these models needs an assistant-prefill + `continue_` trick
    pydantic_ai's Agent interface can't express, and pydantic_ai's default
    tool-calling output mode was separately found to silently fail against
    at least one real local chat template (schema never reached the model
    at all, confirmed by intercepting the raw HTTP request).

    Reuses _PROMPT/_EXPECTED_ANSWER like the checks above, but deliberately
    does NOT pass reasoning_warmstart: without it, `_adjust_reasoning_*`
    never engages the prefill/continue_ trick, so this only exercises the
    concern that's actually generic across every local model this client
    might point at — does the schema reach the model and validate correctly
    — not reasoning quality or the trick's own currently-broken
    non-streaming behavior (see docs/conversation_memory.html's
    "LLMClientLocal doesn't use pydantic_ai" callout for that separate,
    still-open bug: with reasoning_warmstart set, chat_completion() returns
    an empty response on at least one tested local model). CompletionAgent's
    result also has no all_messages()/ThinkingPart to assert on, unlike the
    pydantic_ai-backed clients above, so structured output is the only thing
    verified here.

    Unlike the API-key checks above, the skip condition here is a live
    reachability probe, not a cheap os.getenv() — so it's checked inside the
    test body via pytest.skip() rather than a skipif decorator, which would
    fire the network call at collection time on every pytest run.
    """
    try:
        requests.get(f"{_LOCAL_BASE_URL}/v1/internal/model/info", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip(f"No local LLM server reachable at {_LOCAL_BASE_URL}")

    model = os.getenv("LOCAL_MODEL")
    client = LLMClientLocal(base_url=_LOCAL_BASE_URL, model_name=model)
    agent = client.build_agent(
        LLMTask.ARCHITECT,
        output_type=_Answer,
        system_prompt="You are a careful math assistant.",
        reasoning=True,
    )
    result = agent.run_sync(_PROMPT)
    print(result)

    assert result.output.answer == _EXPECTED_ANSWER, (
        f"Structured output wrong or unvalidated: got {result.output.answer!r}, "
        f"expected {_EXPECTED_ANSWER}"
    )


def test_local_openai_reasoning_and_structured_output():
    """
    LLMClientLocalOpenAI — a local model with NATIVE reasoning + tool-calling
    (e.g. Gemma 4 26B) served through textgen-webui's OpenAI-compatible
    endpoint. Unlike LLMClientLocal (previous test), this one IS pydantic_ai-
    backed, so it goes through the shared _assert_reasoning_and_structured_output
    check that also asserts on a real ThinkingPart.

    This test is the verification gate for LLMClientLocalOpenAI's best-guess
    wire-format defaults — the `chat_template_kwargs.enable_thinking` thinking
    toggle and the PromptedOutput schema-delivery mode (see the class
    docstring). Two failure modes it's specifically here to catch:
      * no ThinkingPart → the enable_thinking toggle isn't how this server
        engages the model's thinking (or the server returns thinking inline as
        <think> tags that pydantic_ai's generic OpenAI profile doesn't split
        into a ThinkingPart — would need a model profile with thinking_tags).
      * validation/parse failure → the schema isn't reaching the model even
        via the prompt; inspect the raw request.

    Same live-reachability skip as test_local_reasoning_and_structured_output
    (checked in-body, not via skipif, so collection never hits the network).
    Point it at the model with LOCAL_MODEL.
    """
    try:
        requests.get(f"{_LOCAL_BASE_URL}/v1/internal/model/info", timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip(f"No local LLM server reachable at {_LOCAL_BASE_URL}")

    model = os.getenv("LOCAL_MODEL")
    client = LLMClientLocalOpenAI(base_url=_LOCAL_BASE_URL, model_name=model)
    _assert_reasoning_and_structured_output(client, LLMTask.ARCHITECT)
