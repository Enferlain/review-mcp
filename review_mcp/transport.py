"""OpenAI-compatible model transport helpers for review-mcp."""

from __future__ import annotations

import os

from .config import _get_model_api_timeout_seconds


def _make_client():
    from openai import OpenAI

    api_key = os.getenv("AI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError(
            "AI_API_KEY environment variable is not set. "
            "ZHIPU_API_KEY is also accepted for backward compatibility."
        )

    base_url = os.getenv("ZHIPU_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=_get_model_api_timeout_seconds(),
    )


def _create_streaming_chat_completion(client, **kwargs):
    """Create a completion while consuming incremental response chunks.

    The coding API can spend a long time reasoning before a tool call is complete.
    A non-streaming request receives no response body during that work, so httpx's
    read timer can expire even though the model is still generating. Streaming
    makes that timeout an inter-chunk idle bound instead of a whole-generation
    bound; this helper rebuilds the ordinary completion shape expected by the
    review loop.
    """
    from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageFunctionToolCall,
    )
    from openai.types.chat.chat_completion_message_function_tool_call import Function

    kwargs.setdefault("stream_options", {"include_usage": True})
    stream = client.chat.completions.create(stream=True, **kwargs)

    # Keep compatibility with simple test doubles and OpenAI-compatible clients
    # that ignore stream=True and return a complete response object.
    if isinstance(getattr(stream, "choices", None), list):
        return stream

    content_parts: list[str] = []
    reasoning_content_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, str]] = {}
    finish_reason = None
    usage = None
    try:
        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                reasoning_content_parts.append(reasoning_content)
            for tool_call in delta.tool_calls or []:
                parts = tool_call_parts.setdefault(
                    tool_call.index,
                    {"id": "", "name": "", "arguments": ""},
                )
                if tool_call.id:
                    parts["id"] += tool_call.id
                if tool_call.function is not None:
                    if tool_call.function.name:
                        parts["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        parts["arguments"] += tool_call.function.arguments
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=parts["id"],
            type="function",
            function=Function(
                name=parts["name"],
                arguments=parts["arguments"],
            ),
        )
        for _, parts in sorted(tool_call_parts.items())
    ]
    message = ChatCompletionMessage(
        role="assistant",
        content="".join(content_parts) or None,
        tool_calls=tool_calls or None,
        # Z.ai requires preserved thinking to be sent back unchanged when a
        # tool call is followed by another completion request.
        reasoning_content="".join(reasoning_content_parts) or None,
    )

    class _Choice:
        def __init__(self):
            self.message = message
            self.finish_reason = finish_reason

    class _Completion:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = usage

    return _Completion()
