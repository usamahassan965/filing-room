"""Filing HTML to canonical text, with the section boundaries questions land in.

Two requirements shape this file, and between them they rule out every
off-the-shelf converter I tried.

**Offsets must survive.** A chunk carries ``{accn, item, char_start, char_end}``
and the gate re-reads the source to prove ``text[char_start:char_end]`` is still
the chunk. That only works if the offsets are into a string we store verbatim,
so the canonical text is built once, whitespace already collapsed, and never
post-processed. A markdown converter cannot do this: its output is a
transformation of the document, not a view of it, and there is no way back.

**Item boundaries are not a regex over flattened text.** On one NVIDIA 10-K a
naive ``Item \\d`` scan finds 55 matches for 23 sections. The other 32 are the
table of contents and cross-references in running prose ("see Item 15 of this
Annual Report"). What separates a heading from a mention is *structure*: a real
heading occupies a block on its own. So the parser keeps block boundaries, and
the splitter looks only at whole blocks.

That leaves the table of contents, which is also made of blocks that are nothing
but a heading. Three filer layouts appear in this corpus and each needs a
different discriminator:

* number and title in separate cells (NVIDIA, Intel) -- the number's block has no
  title, which is never true of a real heading that carries one;
* number and title in one cell (Walmart, Broadcom, ConocoPhillips) -- looks
  exactly like a real heading, and is only distinguishable by what follows it:
  another heading a few dozen characters later, rather than a section of prose;
* number alone as the heading, title in the next block (AMD's older 10-Qs) --
  which means "no title" cannot simply be rejected.

Density alone is not the rule either: "Item 4. Mine Safety Disclosures: Not
applicable" and four more like it really are adjacent one-line sections at the
end of every 10-Q. What is true of a contents table and of nothing else is that
it *lists the whole document in a corner of it* -- an ascending run covering
almost every item found, packed into a small fraction of the text. Position is
not part of the rule, because Intel puts its contents table at the *end*.

Some filings cannot be split at all. Intel's recent 10-Ks and every Intel 10-Q
reorganise the narrative under their own headings and carry no ``Item N``
heading in the body; their only item list is that end-placed table, at offset
517k of 520k. Matching the contents table's *titles* against body blocks was
tried and rejected: it finds 6 of 23 on Intel's 2024 10-K, and Items 7 and 8 --
the two that matter -- are not among them, so the recovered spans would carry
the wrong item label. A wrong label is worse than none, because M6 cites it. So
those filings are indexed as a single whole-document section with the reason
recorded in ``degraded``, which keeps the company in the narrative index.
``quarantine`` is reserved for text that cannot be indexed at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

# Elements that end a run of text. ``br`` and ``hr`` are here because a filer
# who lays a heading out with line breaks instead of paragraphs is common enough
# that ignoring them merges the heading into the paragraph after it.
BLOCK_TAGS = frozenset(
    """p div td th tr li h1 h2 h3 h4 h5 h6 table section article
       br hr ul ol dl dt dd blockquote pre caption tbody thead tfoot""".split()
)
DROP_TAGS = frozenset("script style head title meta link".split())

# ``Item 7A.`` and ``ITEM 1 --`` and ``Item 1B:`` are all the same heading. The
# title is capped generously: Item 5's real title runs to 108 characters.
ITEM_RE = re.compile(r"^item\s+(\d{1,2}\s*[a-c]?)\s*[.–—:;\-]*\s*(.{0,180})$", re.I | re.S)
PART_RE = re.compile(r"^part\s+(i{1,3}v?|iv)\b\s*[.–—:;\-]*\s*(.{0,60})$", re.I | re.S)
# "Part I of this report on page 33." matches that pattern and is a sentence, not
# a heading. A heading does not end in a full stop; a cross-reference does.
PART_MAX_CHARS = 60

# A block longer than this is prose that happens to start with the word "Item",
# not a heading. The longest real heading in the corpus is 141 characters.
MAX_HEADING_CHARS = 200

# Table-of-contents detection. A contents table is a run of headings packed too
# tightly to have bodies between them, climbing in order, that lists *almost
# everything* -- the last part is what separates it from the tail of a 10-Q,
# where "Defaults Upon Senior Securities: None" and four more like it really are
# adjacent one-line sections.
TOC_MIN_RUN = 4
TOC_MAX_GAP = 900
TOC_COVERAGE = 0.8
# ...and that it is crammed into a corner of the document. Coverage alone would
# also describe a filing whose every section happens to be shorter than
# ``TOC_MAX_GAP``, and would drop the whole thing. The four real contents tables
# measured here span 0.3% to 0.8% of their documents; 15% is not a close call.
TOC_MAX_SPAN = 0.15

# A section shorter than this is a heading with nothing under it -- almost always
# a cross-reference the density rule failed to catch, or a genuinely empty
# section like "Item 6. [Reserved]". Kept, but flagged by ``Section.is_stub``.
STUB_CHARS = 200


@dataclass(frozen=True, slots=True)
class Section:
    """One Item, located by character offsets into the filing's canonical text."""

    part: str  # "I" or "II" -- a 10-Q numbers both from 1
    item: str  # "1", "1A", "7" ... upper-cased, spaces removed
    title: str
    start: int  # offset of the heading itself, so the chunk can carry it
    body: int  # offset just past the heading; where the text begins
    end: int

    @property
    def key(self) -> str:
        """``7`` in a 10-K, ``I.1`` in a 10-Q.

        A 10-K numbers its items once through, so the part adds nothing and
        costs something: ConocoPhillips prints no part headings at all, and
        keying on a part that was never announced files Item 7 under Part I.
        A 10-Q restarts at 1 for Part II, so there the part is load-bearing.
        """
        return f"{self.part}.{self.item}" if self.part else self.item

    @property
    def chars(self) -> int:
        return self.end - self.body

    @property
    def is_stub(self) -> bool:
        return self.chars < STUB_CHARS


@dataclass(frozen=True, slots=True)
class ParsedFiling:
    accn: str
    path: Path
    text: str
    blocks: tuple[tuple[int, int], ...]
    sections: tuple[Section, ...]
    quarantine: str | None = None
    degraded: str | None = None

    @property
    def ok(self) -> bool:
        return self.quarantine is None

    @property
    def split(self) -> bool:
        """Did the item splitter find the sections, or is this whole-document?"""
        return self.quarantine is None and self.degraded is None

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)


class _Flattener(HTMLParser):
    """Build the canonical text and remember where each block starts and ends.

    Whitespace is collapsed *as the text is built*, never afterwards, because an
    offset into a string that is later reflowed points at the wrong characters.
    What comes out of here is exactly what gets stored and exactly what the
    offsets index.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._n = 0
        self._open = 0
        self._drop = 0
        self._space = False
        self.blocks: list[tuple[int, int]] = []

    def _write(self, s: str) -> None:
        self._parts.append(s)
        self._n += len(s)

    def _break(self) -> None:
        if self._n > self._open:
            self.blocks.append((self._open, self._n))
            self._write("\n")
        self._open = self._n
        self._space = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in DROP_TAGS:
            self._drop += 1
        elif tag in BLOCK_TAGS:
            self._break()

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if tag in BLOCK_TAGS:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        if tag in DROP_TAGS:
            self._drop = max(0, self._drop - 1)
        elif tag in BLOCK_TAGS:
            self._break()

    def handle_data(self, data: str) -> None:
        if self._drop:
            return
        # Non-breaking spaces are everywhere in filing tables and are not spaces
        # to ``str.split``; normalising them here keeps headings matchable.
        data = data.replace("\xa0", " ")
        s = " ".join(data.split())
        # Whether two text runs are one word or two is decided by the source, not
        # by the fact that they arrive as separate ``handle_data`` calls. Filers
        # break words across styling spans constantly -- Schlumberger's 10-K
        # renders its Item 8 heading as ``<span>I</span><span>tem 8.</span>`` --
        # so inserting a space at every run boundary produces "I tem 8." and
        # loses the section. The space belongs to the source or nowhere.
        pad = self._space or data[:1].isspace()
        if not s:
            self._space = self._space or bool(data)
            return
        if pad and self._n > self._open and not self._parts[-1].endswith((" ", "\n")):
            self._write(" ")
        self._write(s)
        self._space = data[-1:].isspace()

    def finish(self) -> tuple[str, list[tuple[int, int]]]:
        self._break()
        return "".join(self._parts), self.blocks


def flatten(raw: str) -> tuple[str, list[tuple[int, int]]]:
    """HTML to canonical text plus block spans. Offsets index the returned text."""
    # ``html.parser`` does not treat CDATA-ish script bodies specially enough to
    # be trusted with inline XBRL, and these filings carry megabytes of it.
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    p = _Flattener()
    p.feed(raw)
    p.close()
    return p.finish()


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    part: str
    item: str
    title: str


def _is_cross_reference(s: str, tail: str) -> bool:
    """Is this Part line a sentence about a part rather than the part's heading?

    "Part I of this report on page 33." matches ``PART_RE`` and is prose. The
    discriminator is not the full stop on its own: Chevron's body announces its
    first part as exactly ``PART I.``, and rejecting that leaves the part stuck
    on the II the contents table last set, which files every Part I item of the
    quarter under Part II. A sentence has *words after the numeral* and ends in
    a full stop; a heading has at most a title, and titles do not.
    """
    return bool(tail.strip(" .:;–—-")) and s.endswith(".")


def _candidates(text: str, blocks: list[tuple[int, int]], *, parts: bool) -> list[_Candidate]:
    part = "I" if parts else ""
    out: list[_Candidate] = []
    for i, (start, end) in enumerate(blocks):
        s = text[start:end].strip()
        if not s or len(s) > MAX_HEADING_CHARS:
            continue
        # ConocoPhillips lays its part headings out as three table cells --
        # "PART", "I.", "FINANCIAL INFORMATION" -- so the numeral is in the next
        # block and "PART" alone matches nothing. Without this, the part stays on
        # whatever the contents table last set and the whole quarter files under
        # Part II.
        if s.lower().rstrip(" .:") == "part" and i + 1 < len(blocks):
            s = f"{s} {text[blocks[i + 1][0] : blocks[i + 1][1]].strip()}"[:PART_MAX_CHARS]
        pm = PART_RE.match(s)
        if pm and len(s) < PART_MAX_CHARS and not _is_cross_reference(s, pm.group(2)):
            if parts:
                part = pm.group(1).upper()
            continue
        m = ITEM_RE.match(s)
        if not m:
            continue
        title = m.group(2).strip(" .:;–—-")
        if not title:
            # AMD's older 10-Qs put the number and the title in adjacent blocks.
            # Borrow the next one, but only if it reads like a title rather than
            # like the first sentence of a section.
            nxt = text[blocks[i + 1][0] : blocks[i + 1][1]].strip() if i + 1 < len(blocks) else ""
            if nxt and len(nxt) <= MAX_HEADING_CHARS and not nxt.endswith("."):
                title = nxt
        out.append(_Candidate(start, end, part, re.sub(r"\s+", "", m.group(1)).upper(), title))
    return out


def _order(c: _Candidate) -> tuple[int, int, str]:
    """Sort key for Item numbering: Part I before Part II, 1 before 1A before 2."""
    m = re.match(r"(\d+)([A-C]?)", c.item)
    part = len(c.part)  # I -> 1, II -> 2, III -> 3
    return (part, int(m.group(1)) if m else 0, m.group(2) if m else "")


def _runs(cands: list[_Candidate]) -> list[list[_Candidate]]:
    """Split candidates into runs of adjacent, ascending headings.

    A run breaks on a gap wide enough to hold a section, and on the numbering
    going backwards -- which is exactly what happens at the end of a contents
    table, where the last entry is followed by the body's Item 1.
    """
    out: list[list[_Candidate]] = []
    run: list[_Candidate] = []
    for c in cands:
        if run and c.start - run[-1].start <= TOC_MAX_GAP and _order(c) > _order(run[-1]):
            run.append(c)
        else:
            if run:
                out.append(run)
            run = [c]
    if run:
        out.append(run)
    return out


def _drop_contents_tables(cands: list[_Candidate], n_chars: int) -> list[_Candidate]:
    """Remove the contents table wherever it sits, without taking sections with it.

    Every simpler rule fails on some filer in this corpus. Rejecting headings
    with no title catches NVIDIA's two-cell table and misses Walmart's one-cell
    one. Rejecting whatever comes first misses Intel, which puts its contents
    table *after* the body. Rejecting any tight run takes the last five sections
    of every 10-Q with it, because "Item 4. Mine Safety Disclosures: Not
    applicable" is a real section that really is 40 characters long.

    What is true of a contents table and of nothing else is that it lists the
    whole document in a corner of it.
    """
    if not cands:
        return []
    total = len({(c.part, c.item) for c in cands})
    keep: list[_Candidate] = []
    for run in _runs(cands):
        covers = len({(c.part, c.item) for c in run}) / total
        span = (run[-1].start - run[0].start) / max(n_chars, 1)
        if len(run) >= TOC_MIN_RUN and covers >= TOC_COVERAGE and span <= TOC_MAX_SPAN:
            continue
        keep.extend(run)
    return keep


def find_sections(text: str, blocks: list[tuple[int, int]], *, parts: bool) -> list[Section]:
    """Locate every Item heading and give each one the span up to the next.

    ``parts`` says whether Part headings carry meaning for this form. See
    ``Section.key``: they do in a 10-Q, which restarts its numbering, and they
    are actively harmful in a 10-K, which does not always print them.
    """
    cands = _drop_contents_tables(_candidates(text, blocks, parts=parts), len(text))
    if not cands:
        return []

    # A duplicate at this stage is a heading repeated in the body -- a filer
    # restating "Item 1A. Risk Factors" above a continuation. The first is the
    # section start; the rest belong to it.
    seen: set[tuple[str, str]] = set()
    unique: list[_Candidate] = []
    for c in cands:
        if (c.part, c.item) not in seen:
            seen.add((c.part, c.item))
            unique.append(c)

    sections = []
    for i, c in enumerate(unique):
        end = unique[i + 1].start if i + 1 < len(unique) else len(text)
        sections.append(Section(c.part, c.item, c.title, c.start, c.end, end))
    return sections


# The sections questions actually land in: risk factors, MD&A, the statements
# in a 10-K; the statements and MD&A in a 10-Q. A filing missing these has not
# been split, whatever else was found. The 10-K keys carry no part, because its
# ``Section.key`` does not.
REQUIRED_10K = ("1A", "7", "8")
REQUIRED_10Q = ("I.1", "I.2")

# Text this short is not a filing -- an exhibit stub, or a fetch that stored an
# error page. Below it there is nothing to index and quarantine is the honest
# answer; above it, a filing we cannot split is still worth its full text.
MIN_USABLE_CHARS = 20_000

WHOLE_DOC_ITEM = "FULL"


def parse_filing(path: Path, *, accn: str, form: str) -> ParsedFiling:
    """Parse one filing into sections, or into one whole-document section, or not.

    The form comes from the manifest rather than from counting headings. Guessing
    it misclassified four 10-Qs in a 24-filing sample, because a table of
    contents inflates the heading count past any threshold that separates them.

    There are three outcomes, not two. Intel's 10-Qs and recent 10-Ks reorganise
    the narrative under their own headings and carry no ``Item N`` heading in the
    body at all -- their item table sits at offset ~142k of 143k, after
    everything. Dropping those loses a whole company from the narrative index to
    satisfy a splitter, so a filing with usable text but no findable sections is
    *degraded*: indexed as a single section spanning the document, with the
    reason recorded. Quarantine is kept for text that cannot be indexed at all.
    """
    raw = path.read_bytes().decode("utf-8", errors="replace")
    text, blocks = flatten(raw)
    parts = not form.upper().startswith("10-K")
    sections = find_sections(text, blocks, parts=parts)

    required = REQUIRED_10K if form.upper().startswith("10-K") else REQUIRED_10Q
    found = {s.key for s in sections}
    reason = None
    if not sections:
        reason = "no Item heading in the body"
    elif missing := [k for k in required if k not in found]:
        reason = f"no heading for {', '.join(missing)}"

    quarantine = degraded = None
    if reason and len(text) < MIN_USABLE_CHARS:
        quarantine = f"{reason}; only {len(text):,} chars of text"
        sections = []
    elif reason:
        degraded = reason
        sections = [Section("", WHOLE_DOC_ITEM, form.upper(), 0, 0, len(text))]

    return ParsedFiling(
        accn=accn,
        path=path,
        text=text,
        blocks=tuple(blocks),
        sections=tuple(sections),
        quarantine=quarantine,
        degraded=degraded,
    )
