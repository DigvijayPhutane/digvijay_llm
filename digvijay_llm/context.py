"""Shared rules for fitting a prompt (or a chat) + completion into the context window."""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("digvijay_llm")


def fit_to_context(
    prompt_len: int, max_tokens: int, n_ctx: int, first_is_bos: bool = False
) -> Tuple[Optional[Tuple[int, int]], int]:
    """Decide how much of the prompt to keep and how many tokens to generate.

    At least min(max_tokens, n_ctx // 2) positions are reserved for output.
    A prompt that doesn't fit the rest is left-truncated (its most recent
    tokens are kept, plus a leading BOS token), with a warning; max_tokens is
    then capped to whatever room is left, as llama.cpp does.

    Returns (keep, max_tokens): keep is None when the prompt fits as-is,
    otherwise (head, tail) meaning "keep the first `head` and last `tail`
    tokens".
    """
    if n_ctx < 2:
        raise ValueError(f"n_ctx={n_ctx} is too small to generate anything")
    max_tokens = max(1, max_tokens)
    reserve = min(max_tokens, n_ctx // 2)
    room = n_ctx - reserve
    keep = None
    if prompt_len > room:
        head = 1 if first_is_bos else 0
        keep = (head, room - head)
        logger.warning(
            "Prompt is %d tokens; keeping the last %d so it fits n_ctx=%d with room to generate",
            prompt_len, room - head, n_ctx,
        )
        prompt_len = room
    return keep, min(max_tokens, n_ctx - prompt_len)


def fit_messages(
    messages: List[Dict[str, Any]],
    count_tokens: Callable[[List[Dict[str, Any]]], int],
    max_tokens: int,
    n_ctx: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop the oldest messages until the formatted chat fits n_ctx.

    Same reservation rule as fit_to_context, but trims whole messages so the
    chat template stays intact. Leading system messages are always kept, and
    the history restarts at a user turn (some templates require user and
    assistant to alternate). `count_tokens(messages)` must return the length
    of the fully formatted prompt.

    Returns (messages, max_tokens). Raises ValueError if even the system
    messages plus the newest message don't fit.
    """
    if n_ctx < 2:
        raise ValueError(f"n_ctx={n_ctx} is too small to generate anything")
    max_tokens = max(1, max_tokens)
    room = n_ctx - min(max_tokens, n_ctx // 2)

    messages = list(messages)
    n_system = 0
    while n_system < len(messages) and messages[n_system].get("role") == "system":
        n_system += 1
    system, history = messages[:n_system], messages[n_system:]

    # Where the kept history may start: nothing dropped, any user turn, or
    # (last resort) just the newest message. Fewer messages means a shorter
    # prompt, so binary-search for the earliest start that fits.
    starts = sorted({0, max(len(history) - 1, 0)} | {i for i, m in enumerate(history) if m.get("role") == "user"})
    lengths: Dict[int, int] = {}

    def length_from(i: int) -> int:
        if i not in lengths:
            lengths[i] = count_tokens(system + history[i:])
        return lengths[i]

    lo, hi = 0, len(starts) - 1
    if length_from(starts[0]) > room:
        while lo < hi:
            mid = (lo + hi) // 2
            if length_from(starts[mid]) <= room:
                hi = mid
            else:
                lo = mid + 1
    dropped = starts[lo]
    length = length_from(dropped)
    history = history[dropped:]
    if length > room:
        raise ValueError(
            f"The chat prompt is {length} tokens even with only the system messages and the newest "
            f"message, but n_ctx={n_ctx} leaves {room} after reserving room for the reply. "
            "Shorten that message or use a larger n_ctx."
        )
    if dropped:
        logger.warning(
            "Chat history is too long for n_ctx=%d; dropped the %d oldest message(s)", n_ctx, dropped
        )
    return system + history, min(max_tokens, n_ctx - length)


def apply_keep(ids: Sequence, keep: Optional[Tuple[int, int]]):
    """Slice a token list according to fit_to_context's `keep`."""
    if keep is None:
        return ids
    head, tail = keep
    return list(ids[:head]) + list(ids[len(ids) - tail:])
