#!/usr/bin/env python3
"""Chat rendering, and the one way this package turns a source realisation into a pair.

TWO REALISATION SHAPES, ONE DRAW FUNCTION.  Every family in ``data/`` describes its
surface text as a list of *realisations*, and a realisation is one of exactly two things:

  minimal pair   {"form", "prompt", "frame", "slots"}
                 ``frame`` holds one ``{}`` per entry of ``slots``, and each slot is the
                 string ``"<american>|<british>"``.  The two sides are the same sentence
                 with every slot filled from one side or the other, so length, topic and
                 style are controlled by construction rather than by authoring care.
                 A one-slot realisation may write ``"us"``/``"uk"`` instead of ``slots``.

  authored pair  {"form", "prompt", "chosen", "rejected"}
                 the two sides are written out.  Used where the contrast is not a single
                 interchangeable token — a sense explanation, a register contrast, a
                 model sample — and a shared frame would misrepresent it.

``draw`` handles both.  A family builder's job is therefore to emit realisations, never to
format text itself.  This is what lets six families that were six bespoke record shapes
share one code path and one output schema.

CHAT RENDERING.  The user turn is the realisation's ``prompt``, verbatim, with one
exception: a ``continuation`` prompt is narrative context with no request in it, so it is
wrapped in a single fixed instruction (``CONTINUATION_PREFIX``) applied everywhere.  The
assistant turn is the completion text.  Turns go through the target model's own
``apply_chat_template`` rather than a hand-rolled ChatML string.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from schema import Pair

CONTINUATION_PREFIX = "Continue this passage:\n\n"


def user_turn(form: str, prompt: str) -> str:
    return CONTINUATION_PREFIX + prompt if form == "continuation" else prompt


def slots_of(realisation: dict[str, Any]) -> list[str]:
    """Canonical ``["<american>|<british>", ...]`` for a minimal-pair realisation."""
    if "slots" in realisation:
        return list(realisation["slots"])
    return [f'{realisation["us"]}|{realisation["uk"]}']


def sides(realisation: dict[str, Any]) -> tuple[str, str]:
    """(British, American) completion text for either realisation shape."""
    if "frame" not in realisation:
        return realisation["chosen"].strip(), realisation["rejected"].strip()
    frame, slots = realisation["frame"], slots_of(realisation)
    if frame.count("{}") != len(slots):
        raise ValueError(f"frame has {frame.count('{}')} slots, {len(slots)} fills: {frame!r}")
    uk = [s.split("|", 1)[1] for s in slots]
    us = [s.split("|", 1)[0] for s in slots]
    return frame.format(*uk).strip(), frame.format(*us).strip()


def draw(realisation: dict[str, Any], *, id: str, family: str, group: str, item: str,
         role: str = "install", reserved_for_eval: bool = False,
         meta: dict[str, Any] | None = None) -> Pair:
    """The single constructor every family uses.  Returns a validated Pair."""
    chosen, rejected = sides(realisation)
    extra = dict(meta or {})
    for key, value in realisation.items():
        if key not in ("form", "prompt", "frame", "slots", "uk", "us", "chosen", "rejected"):
            extra.setdefault(key, value)
    if "frame" in realisation:
        extra.setdefault("slots", slots_of(realisation))
    pair = Pair(
        id=id, family=family, group=group, item=item, form=realisation["form"],
        origin=realisation.get("origin", "carrier" if "frame" in realisation else "authored"),
        prompt=user_turn(realisation["form"], realisation["prompt"]).strip(),
        chosen=chosen, rejected=rejected, role=role,
        reserved_for_eval=reserved_for_eval,
        meta={k: v for k, v in sorted(extra.items()) if v is not None and v != ""},
    )
    pair.validate()
    return pair


class ChatRenderer:
    """Wraps a local chat checkpoint's tokenizer.  Nothing else in the package imports
    transformers, so the source data and the draw path stay tokenizer-independent."""

    def __init__(self, model: Path):
        from transformers import AutoTokenizer
        self.model = Path(model).resolve()
        self.tok = AutoTokenizer.from_pretrained(self.model, local_files_only=True)

    def full_text(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=False)

    def prompt_text(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages[:1], tokenize=False,
                                            add_generation_prompt=True)


class PlainRenderer:
    """Fallback used by tests and by ``--no-chat``: emits the turns without a template."""

    model = None

    def full_text(self, messages: list[dict[str, str]]) -> str:
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages)

    def prompt_text(self, messages: list[dict[str, str]]) -> str:
        return f"user: {messages[0]['content']}\nassistant:"
