"""Firefly-style script markup, parsed into segments for the TTS backends.

    [pause 800]  [pause 800ms]  [pause 1.5s]    exact silence here (max 10 s)
    [happy]...[/happy]  [happy 1.5]...[/happy]  emotion span, intensity 0-3
    *word*                                      emphasis
    {surface|reading}                           say `reading` instead
    {雨|あ'め}  {飴|あめ0}                       Japanese accent: the pitch drops
                                                after the mora before ', or 0 = flat
    \\[  \\]  \\*  \\{  \\}  \\|  \\\\                  literal characters

The text is cut only at pauses and at emotion/emphasis boundaries. A run of
plain text, however many sentences long, stays one segment and so one TTS call:
splicing per sentence made speech sound mechanical (see
make_avatar.synthesize_speech), so splits happen only where the script asks.

Pure Python, no model imports, so it can be tested without any model installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_PAUSE_MS = 10_000
MAX_INTENSITY = 3.0

_PAUSE = re.compile(r"pause\s+(\d+(?:\.\d+)?)\s*(ms|s)?", re.IGNORECASE)
_OPEN = re.compile(r"([a-z0-9_]+)(?:\s+(\d+(?:\.\d+)?))?")
_CLOSE = re.compile(r"/([a-z0-9_]+)")
_ESCAPABLE = set("[]*{}|\\")
# The sentence rule --pause-ms has always used: ./!/? followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Small kana fuse with the kana before them into one mora; っ, ん and ー are
# morae of their own.
_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_KANA = re.compile(r"[ぁ-ゖァ-ヺー]+")


class MarkupError(ValueError):
    """Malformed markup, reported with where it is in the text."""

    def __init__(self, message: str, text: str, offset: int):
        snippet = text[max(0, offset - 12): offset + 12].replace("\n", " ")
        super().__init__(f"{message} (character {offset}: ...{snippet}...)")
        self.offset = offset


@dataclass
class Piece:
    text: str
    # None: the TTS decides. 0: flat (heiban). n >= 1: pitch drops after mora n.
    accent: int | None = None


@dataclass
class Segment:
    pieces: list = field(default_factory=list)
    emotion: str | None = None
    intensity: float = 1.0
    emphasis: bool = False
    pause_after_ms: int = 0

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.pieces)


def count_morae(kana: str) -> int:
    return sum(1 for ch in kana if ch not in _SMALL_KANA)


def parse_reading(reading: str, lang: str) -> Piece:
    """The reading half of {surface|reading}, with Japanese accent marks read off it."""
    if lang != "ja":
        if "'" in reading:
            raise ValueError("accent marks (') are for Japanese readings only")
        return Piece(reading)
    if reading.count("'") > 1:
        raise ValueError("a reading can mark only one accent nucleus (')")
    if "'" in reading:
        before, after = reading.split("'")
        if not _KANA.fullmatch(before + after):
            raise ValueError(f"a reading with an accent mark must be all kana: {reading!r}")
        if not before:
            raise ValueError("' goes after a mora: the pitch drops after it")
        return Piece(before + after, count_morae(before))
    if len(reading) > 1 and reading.endswith("0") and _KANA.fullmatch(reading[:-1]):
        return Piece(reading[:-1], 0)
    return Piece(reading)


def load_pronunciations(path: str | None) -> dict:
    """{lang: {word: reading}} from YAML; pronunciations.yaml next to this file by default, {} if absent."""
    if path is None:
        default = Path(__file__).resolve().parent / "pronunciations.yaml"
        if not default.exists():
            return {}
        path = str(default)
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not all(isinstance(words, dict) for words in data.values()):
        raise ValueError(f"{path}: expected language codes mapping words to readings, "
                         "e.g.  ja: {雨: \"あ'め\"}")
    return {str(lang): {str(w): str(r) for w, r in words.items()} for lang, words in data.items()}


def _dictionary_pieces(text: str, lang: str, pron: dict) -> list:
    if not pron:
        return [Piece(text)]
    alternatives = "|".join(re.escape(k) for k in sorted(pron, key=len, reverse=True))
    # Japanese/Chinese have no spaces to mark word edges; elsewhere "Ueno" must
    # not rewrite the start of "Uenomachi".
    pattern = alternatives if lang in ("ja", "zh") else rf"(?<!\w)(?:{alternatives})(?!\w)"
    pieces, pos = [], 0
    for match in re.finditer(pattern, text):
        if match.start() > pos:
            pieces.append(Piece(text[pos:match.start()]))
        try:
            pieces.append(parse_reading(pron[match.group(0)], lang))
        except ValueError as exc:
            raise ValueError(f"pronunciations: {match.group(0)}: {exc}") from None
        pos = match.end()
    if pos < len(text):
        pieces.append(Piece(text[pos:]))
    return pieces


def _trim(pieces: list) -> list:
    out = [Piece(p.text, p.accent) for p in pieces if p.text]
    while out and out[0].accent is None and not out[0].text.strip():
        out.pop(0)
    while out and out[-1].accent is None and not out[-1].text.strip():
        out.pop()
    if out:
        out[0].text = out[0].text.lstrip()
        out[-1].text = out[-1].text.rstrip()
    return out


def parse(text: str, lang: str, pronunciations: dict | None = None) -> list:
    """Marked-up script -> segments. Raises MarkupError on malformed markup."""
    pron = pronunciations or {}
    segments, pieces, plain = [], [], []
    emotion = None       # (name, intensity, offset of its tag)
    emphasis_at = None   # offset of the open *, while inside one

    def flush_plain():
        if plain:
            pieces.extend(_dictionary_pieces("".join(plain), lang, pron))
            plain.clear()

    def flush_segment():
        flush_plain()
        trimmed = _trim(pieces)
        pieces.clear()
        if not trimmed:
            return
        if not any(ch.isalnum() for p in trimmed for ch in p.text):
            # Nothing to say, only punctuation left after a tag ("*really*.").
            # It belongs to the words before it, and a TTS asked to say just
            # "." fails: Chatterbox crashes on it.
            if segments and segments[-1].pieces:
                segments[-1].pieces.extend(trimmed)
            return
        segments.append(Segment(trimmed, emotion[0] if emotion else None,
                                emotion[1] if emotion else 1.0, emphasis_at is not None))

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in _ESCAPABLE:
            plain.append(text[i + 1])
            i += 2
            continue
        if ch == "[":
            end = text.find("]", i)
            if end < 0:
                raise MarkupError("'[' has no closing ']'", text, i)
            body = text[i + 1:end].strip()
            if m := _PAUSE.fullmatch(body):
                flush_segment()
                ms = float(m.group(1)) * (1000 if (m.group(2) or "ms").lower() == "s" else 1)
                if ms > MAX_PAUSE_MS:
                    raise MarkupError(f"pauses are at most {MAX_PAUSE_MS} ms", text, i)
                if segments:
                    segments[-1].pause_after_ms += int(round(ms))
                else:
                    segments.append(Segment([], pause_after_ms=int(round(ms))))
            elif m := _CLOSE.fullmatch(body):
                if not emotion or emotion[0] != m.group(1):
                    raise MarkupError(f"[/{m.group(1)}] doesn't close an open [{m.group(1)}]", text, i)
                flush_segment()
                emotion = None
            elif m := _OPEN.fullmatch(body):
                if emotion:
                    raise MarkupError(f"[{m.group(1)}] is inside [{emotion[0]}]: emotion spans can't nest",
                                      text, i)
                intensity = float(m.group(2)) if m.group(2) else 1.0
                if intensity > MAX_INTENSITY:
                    raise MarkupError(f"intensity must be between 0 and {MAX_INTENSITY:g}", text, i)
                flush_segment()
                emotion = (m.group(1), intensity, i)
            else:
                raise MarkupError(f"unknown tag [{body}] (write \\[ for a literal bracket)", text, i)
            i = end + 1
            continue
        if ch == "*":
            flush_segment()
            emphasis_at = None if emphasis_at is not None else i
            i += 1
            continue
        if ch == "{":
            end = text.find("}", i)
            if end < 0:
                raise MarkupError("'{' has no closing '}'", text, i)
            surface, sep, reading = text[i + 1:end].partition("|")
            if not sep or not surface.strip() or not reading.strip() or "|" in reading:
                raise MarkupError("expected {word|reading}", text, i)
            flush_plain()
            try:
                pieces.append(parse_reading(reading.strip(), lang))
            except ValueError as exc:
                raise MarkupError(str(exc), text, i) from None
            i = end + 1
            continue
        plain.append(ch)
        i += 1

    if emotion:
        raise MarkupError(f"[{emotion[0]}] is never closed", text, emotion[2])
    if emphasis_at is not None:
        raise MarkupError("'*' is never closed (write \\* for a literal asterisk)", text, emphasis_at)
    flush_segment()
    return segments


def split_at_sentences(segments: list, pause_ms: int) -> list:
    """--pause-ms: also cut each segment at sentence ends, with pause_ms of silence there.

    Explicit [pause]s keep their own length; only boundaries inside a segment
    get pause_ms, so a tag mid-sentence ("I am *very* glad") adds no gap.
    """
    out = []
    for seg in segments:
        current = []
        for piece in seg.pieces:
            if piece.accent is not None:
                current.append(piece)
                continue
            parts = _SENTENCE_END.split(piece.text)
            for k, part in enumerate(parts):
                if part:
                    current.append(Piece(part))
                if k < len(parts) - 1:
                    out.append(Segment(current, seg.emotion, seg.intensity, seg.emphasis, pause_ms))
                    current = []
        out.append(Segment(current, seg.emotion, seg.intensity, seg.emphasis, seg.pause_after_ms))
    return [s for s in out if s.pieces or s.pause_after_ms]
