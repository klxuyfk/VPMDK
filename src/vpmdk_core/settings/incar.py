"""INCAR-derived execution settings and related parsing helpers."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict

from pymatgen.io.vasp import Incar


KBAR_TO_EV_PER_A3 = 0.1 / 160.21766208


@dataclass(frozen=True)
class IncarSettings:
    """Container for the INCAR parameters that drive the simulation."""

    nsw: int = 0
    ibrion: int = -1
    ediffg: float = -0.02
    isif: int = 2
    stress_isif: int = 2
    pstress: float | None = None
    tebeg: float = 300.0
    teend: float = 300.0
    potim: float = 2.0
    nfree: int | None = None
    symprec: float = 1e-5
    mdalgo: int = 0
    smass: float | None = None
    thermostat_params: Dict[str, float] = field(default_factory=dict)

    @property
    def energy_tolerance(self) -> float | None:
        """Energy convergence threshold in eV when EDIFFG>0."""

        return self.ediffg if self.ediffg > 0 else None

    @property
    def force_limit(self) -> float:
        """Return ASE ``fmax`` argument derived from EDIFFG semantics."""

        if self.ediffg > 0:
            return -abs(self.ediffg)
        if self.ediffg < 0:
            return abs(self.ediffg)
        return 0.05


SUPPORTED_INCAR_TAGS = {
    "ISIF",
    "IBRION",
    "NSW",
    "EDIFFG",
    "PSTRESS",
    "TEBEG",
    "TEEND",
    "POTIM",
    "NFREE",
    "SYMPREC",
    "MDALGO",
    "SMASS",
    "ANDERSEN_PROB",
    "LANGEVIN_GAMMA",
    "CSVR_PERIOD",
    "NHC_NCHAINS",
    "NHC_PERIOD",
    "MAGMOM",
    "IMAGES",
    "ICHAIN",
    "IOPT",
    "LCLIMB",
    "LNEBCELL",
    "SPRING",
}

SUPPORTED_ISIF_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8}

# The MD algorithms _select_md_dynamics implements: 0 velocity-Verlet (NVE),
# 1 Andersen, 2/4 Nose-Hoover chain, 3 Langevin, 5 CSVR.
SUPPORTED_MDALGO_VALUES = {0, 1, 2, 3, 4, 5}

# Ceiling for |PSTRESS| in kBar. 1e6 kBar = 100 TPa, >10x beyond the highest
# pressures in any DFT study, while safely inside float64 arithmetic for the
# optimizer and the fixed-width OUTCAR pressure fields.
_MAX_PSTRESS_KBAR = 1.0e6

# Ceiling for the magnitude of MD scalars handed to ASE's integrators
# (TEBEG/TEEND in K, POTIM in fs, LANGEVIN_GAMMA in 1/ps). Overflow-scale
# values (an exponent typo: 1e300 for 1e3) pass the finiteness checks and
# then produce nan in the first force call -- classified as RETRYABLE exit 2
# for a permanently broken INCAR. 1e9 is >1000x beyond the largest value
# measured to complete (TEBEG=1e6), so nothing that runs today is rejected;
# merely-large values below the cap still diverge as the genuine dynamics
# they request.
_MAX_MD_SCALAR_MAGNITUDE = 1.0e9


def _reject_absurd_md_magnitude(tag: str, value: float | None) -> None:
    if value is None:
        return
    if abs(float(value)) > _MAX_MD_SCALAR_MAGNITUDE:
        raise ValueError(
            f"{tag} = {value:g} exceeds the supported magnitude of "
            f"{_MAX_MD_SCALAR_MAGNITUDE:g}; check the exponent."
        )

_FORTRAN_EXPONENT_RE = re.compile(r"[dD](?=[+-]?\d)")
_NUMERIC_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

# Tags that are REAL-valued in VASP but int-typed by pymatgen's generic
# proc_val: 'SPRING = -5.5' (a legal VTST value) came back as -5. The R142
# fractional-literal guard then REJECTED the legal file (a SPEC 1.1
# regression), and before that the committed tree silently ran with the
# FLOORED spring. These tags are repaired from the raw text instead
# (_repair_mistyped_real_tags) and exempted from the integer rules.
_REAL_TAGS_PYMATGEN_INT_TYPES = frozenset({"SPRING"})

# Tags whose VASP semantics are INTEGER regardless of how pymatgen types
# them: pymatgen leaves NHC_NCHAINS/IOPT/ICHAIN as floats, so 'NHC_NCHAINS =
# 2.7' passed the float-equality check and VPMDK's own int(float(...))
# coercer silently floored it to 2 -- while the byte-analogous 'NSW = 2.7'
# (int-typed by pymatgen) was rejected. The fractional-literal rule is keyed
# on THIS set, not on the parser's typing.
_INTEGER_SEMANTIC_INCAR_TAGS = frozenset(
    {
        "NSW",
        "IBRION",
        "ISIF",
        "NFREE",
        "MDALGO",
        "NHC_NCHAINS",
        "IMAGES",
        "ICHAIN",
        "IOPT",
        # The families read OUTSIDE _load_incar_settings were missed at
        # first: 'NGXF = 100.5' silently resolved a 100-point grid and
        # 'NELM = 2.7' wrote three mutually contradictory echoes (raw 2.7,
        # header 2, vasprun 2), where VASP refuses the file.
        "NGX",
        "NGY",
        "NGZ",
        "NGXF",
        "NGYF",
        "NGZF",
        "NELM",
        "NELMIN",
        "NELMDL",
    }
)

# Tags whose first value token is a scalar number in some reader -- pymatgen's
# typed proc_val OR VPMDK's own _NUMERIC_RE-based thermostat/VTST parsing. A
# corrupted token for one of these must be rejected from the RAW text: which
# parser types it decides nothing (the R130/R134 lesson). MAGMOM is excluded
# (its 'N*value' mini-language legitimately fails a plain float()); bool tags
# are excluded by construction.
_SCALAR_NUMERIC_INCAR_TAGS = frozenset(
    {
        "NSW",
        "IBRION",
        "EDIFFG",
        "ISIF",
        "PSTRESS",
        "TEBEG",
        "TEEND",
        "POTIM",
        "NFREE",
        "SYMPREC",
        "MDALGO",
        "SMASS",
        "ANDERSEN_PROB",
        "LANGEVIN_GAMMA",
        "CSVR_PERIOD",
        "NHC_NCHAINS",
        "NHC_PERIOD",
        "IMAGES",
        "ICHAIN",
        "IOPT",
        "SPRING",
    }
)

# Mirrors pymatgen's own Incar.from_str tokenizer with ONE deliberate narrowing:
# a value ends at a comment (# or !), at a SEMICOLON or at the end of the line,
# and a quoted value keeps its spaces. VASP allows several tags on one line
# separated by ';', and parse_key_value_file -- a one-tag-per-line reader -- does
# not: for ``NSW = 1e5; IBRION = 2`` it yields the single pair
# ("NSW", "1e5; IBRION = 2"), whose first token fails float() and skips the
# check, while IBRION never appears at all. Every tag on such a line therefore
# escaped this guard and was silently read as a different number again.
#
# The narrowing: the whitespace around '=' is [ \t] here, NOT \s. pymatgen's \s*
# matches a NEWLINE, so an empty value swallows the following line(s) -- see
# _reject_swallowed_incar_tags. Keeping this reader line-scoped is what makes the
# disagreement visible; mirroring the newline-crossing would hide it, exactly as
# it did when this reader was written.
# R170: the KEY-side whitespace is \s* like pymatgen's, because its \s*
# CROSSES a newline between the key and '=' -- an assignment spelled
# 'NSW\n= 1e5' parses normally in pymatgen but was invisible to this reader,
# so EVERY raw-text guard built on it (repeat caps, corrupted-token
# rejection, the embedded-assignment rule, the SPRING repair) was bypassed
# by moving the '=' to the next line. Only the VALUE-side whitespace stays
# narrowed to [ \t] per the R133 rationale above.
_INCAR_ASSIGNMENT_RE = re.compile(
    r"(?P<key>\w+)\s*=[ \t]*(?:\"(?P<qval>[^\"\n]*)\"|(?P<val>[^#!;\n]*))"
)
# A second assignment on the SAME line without a ';' separator is a third
# door to the silent-tag-deletion outcome (after blank values and unbalanced
# quotes): pymatgen's value pattern [^#!;\n]* consumes 'NSW = 5 IBRION = 2'
# entirely as NSW's value, proc_val int-reads the leading 5, and IBRION never
# exists -- the run silently changed mode (relaxation -> single point) with
# exit 0. Detection is keyed on KNOWN tag names embedded in another tag's raw
# value; SYSTEM is exempt (a free-text title may legitimately contain
# 'NSW=100 study', and pymatgen keeps that whole string as the title, losing
# nothing the file unambiguously asked for).
_EMBEDDED_ASSIGNMENT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=")

_EMBEDDED_DETECTABLE_INCAR_TAGS = frozenset(
    SUPPORTED_INCAR_TAGS
    | _INTEGER_SEMANTIC_INCAR_TAGS
    | _SCALAR_NUMERIC_INCAR_TAGS
    | {"EDIFF", "PREC", "ENCUT"}
)

_INCAR_LINE_CONTINUATION_RE = re.compile(r"\\\s*\n")


def _raw_incar_assignment_list(
    path: str, *, continuation_values: bool = False
) -> list[tuple[str, str]]:
    """Return ``(KEY, raw text)`` assignments in file order, one line at a time.

    Comments stripped per line and backslash continuations joined, like
    ``Incar.from_str``; several ``KEY = VALUE`` pairs may share a line. Values
    stay as the RAW TEXT the user wrote, which is the only thing that can be
    compared against what the parser made of them.

    ``continuation_values=True`` is for the RAW-ONLY guards: pymatgen's
    ``\\s*=\\s*`` crosses the newline on the VALUE side too, so for a
    blank-valued assignment the parser consumes the NEXT line as the value --
    ``MAGMOM =\\n10000000000*1.0`` detonated proc_val's list expansion before
    any guard built on this reader could see the token, and the same spelling
    bypassed the corrupted-token and int-repair guards. With the flag, a
    blank unquoted value is substituted by the following non-blank line, the
    text pymatgen will actually feed to proc_val. The swallow guard MUST keep
    the default line-scoped read: a blank staying visibly blank is what makes
    the parser disagreement detectable at all (the R133 rationale above).
    """

    with open(path, encoding="utf-8-sig", errors="surrogateescape") as handle:
        text = handle.read()
    text = "\n".join(
        line.split("#", 1)[0].split("!", 1)[0].rstrip() for line in text.splitlines()
    )
    text = _INCAR_LINE_CONTINUATION_RE.sub(" ", text)
    assignments: list[tuple[str, str]] = []
    for match in _INCAR_ASSIGNMENT_RE.finditer(text):
        quoted = match.group("qval")
        value = quoted if quoted is not None else (match.group("val") or "")
        value = value.strip()
        if continuation_values and quoted is None:
            if not value:
                # Blank unquoted value: pymatgen's \s*=\s* consumes the
                # following line as the value.
                remainder_full = text[match.end():]
                remainder = remainder_full.lstrip("\n \t")
                lead = len(remainder_full) - len(remainder)
                value = remainder.split("\n", 1)[0].strip()
                value_abs_start = match.end() + lead
                if not value.startswith('"'):
                    # pymatgen's unquoted value pattern is [^#!;\n]*: it
                    # stops at ';' and the rest of the line is parsed as
                    # its own assignment(s). Taking the whole line made
                    # the raw-only guards judge text the parser never
                    # consumes -- 'TEBEG =' then '300; NSW = 5' parses
                    # fine in pymatgen but was falsely rejected on the
                    # token '300;'. (Comments are already stripped per
                    # line above; quotes win over ';' in the quoted
                    # branch, mirrored below.)
                    value = value.split(";", 1)[0].strip()
            else:
                value_abs_start = match.start("val")
            if value.startswith('"'):
                # pymatgen's QUOTED branch is re.DOTALL ("(?P<qval>.*?)"), so
                # a quote here spans to the next '"' anywhere in the file
                # and the QUOTES ARE NOT PART OF THE VALUE -- proc_val then
                # expands the content for a list-typed tag. Mirror it so the
                # guards judge the same text: unconditionally, not only for
                # an unterminated quote -- a continuation value CLOSED on
                # its own line ('POTIM =' then '"2.0"') otherwise kept its
                # quote characters and trailing text, and the corrupted-token
                # guard falsely rejected an INCAR pymatgen parses fine. With
                # no closing quote anywhere, keep the raw text (the
                # unbalanced-quote guards own that case post-parse).
                closing = text.find('"', value_abs_start + 1)
                if closing != -1:
                    value = text[value_abs_start + 1 : closing].strip()
        assignments.append((match.group("key").strip().upper(), value))
    return assignments


def _raw_incar_assignments(
    path: str, *, continuation_values: bool = False
) -> Dict[str, str]:
    """Return raw INCAR text values keyed as pymatgen keys them (last wins)."""

    return dict(
        _raw_incar_assignment_list(path, continuation_values=continuation_values)
    )


def _reject_swallowed_incar_tags(incar, path: str) -> None:
    """Reject an INCAR whose tag the parser never read at all.

    ``Incar.from_str`` matches ``(?P<key>\\w+)\\s*=\\s*(?P<val>[^#!;\\n]*)`` and
    ``\\s*`` MATCHES A NEWLINE, so a tag written with an EMPTY value consumes the
    next non-blank line as its value::

        SYSTEM =
        IBRION = 2

    parses as ``{'SYSTEM': 'IBRION = 2'}`` -- IBRION is simply gone. The run then
    used the default ``ibrion=-1`` and did a SINGLE POINT instead of the
    requested relaxation, exit 0, ``Calculation completed.``, CONTCAR identical
    to the input. Real VASP reads its INCAR line by line and is unaffected, so
    the same file relaxes under VASP. Other doors are worse: a blank tag above
    ``NSW = 100`` gives one step instead of 100, and above ``MDALGO = 3`` gives a
    bare NVE trajectory -- the silent-NVE class of the previous round, reached
    with no warning at all.

    Both tests below are anchored on a BLANK-valued tag, which is the only thing
    that can swallow a line, and both are plain raw-vs-parsed disagreements:

    1. the parser returned a non-empty STRING for that blank tag -- text that can
       only have come from the following line;
    2. the tag written on the NEXT line is missing from the parse entirely.

    Test 2 exists because ``proc_val`` TYPES the swallower: for the bool-typed
    ``LWAVE =`` the swallowed ``TEBEG = 900`` comes back as ``True`` and for
    ``MAGMOM =`` as ``[True, 900]``, so test 1 alone let a blank ``L*`` or
    ``MAGMOM`` line above ``TEBEG``/``TEEND`` delete the temperature and run the
    MD at the 300 K default -- with the requested value still echoed in OUTCAR.

    Deliberately NOT "any tag I can see in the file is missing from the parse":
    that inference also fires whenever the parser splits lines differently than
    this reader (a pymatgen version that does not split on ';' would make every
    compact INCAR look swallowed -- measured, it broke four existing tests), and
    it is the kind of over-broad rule that breaks inputs that work today. A blank
    tag that swallows nothing is left alone, since the parser simply drops it.
    """

    if not hasattr(incar, "get"):
        return
    # An UNBALANCED QUOTE swallows lines through a different pymatgen door
    # than the blank-value tests below: ``Incar.from_str`` is compiled with
    # re.DOTALL and its quoted alternative is ``"(?P<qval>.*?)[ \t]*"``, so a
    # value with a lone opening quote (``SYSTEM = "Cu bulk`` with the closing
    # quote forgotten) runs to the NEXT quote character anywhere in the file,
    # deleting every tag in between -- a requested 200-step relaxation ran as
    # a single point with exit 0. The swallowing tag's RAW value is NON-empty
    # there, so the blank-tag tests never examined it. A parsed STRING value
    # spanning a newline can only be swallowed text: no VASP tag takes a
    # multi-line value, and backslash continuations are joined to one line
    # before parsing.
    try:
        parsed_items = list(incar.items())
    except Exception:
        parsed_items = []
    for key, parsed in parsed_items:
        if isinstance(parsed, str) and "\n" in parsed:
            raise ValueError(
                f"INCAR tag {key} was read as a value spanning multiple lines "
                f"({parsed!r}): an unbalanced quote makes the parser swallow "
                "every following tag up to the next quote character, so those "
                f"tags are silently lost. Balance the quotes on the {key} line."
            )
    try:
        assignments = _raw_incar_assignment_list(path)
    except OSError:
        return
    try:
        parsed_keys = {str(key).upper() for key in incar.keys()}
    except Exception:
        parsed_keys = None

    for index, (key, value) in enumerate(assignments):
        if key != "SYSTEM" and value and parsed_keys is not None:
            # The CAUSAL condition (lesson xlvi/l, refined in R165): a tag
            # name with '=' inside another tag's value is only harmful when
            # that tag is genuinely ABSENT from the parse -- the mere textual
            # presence also matches the standard VASP trailing-comment style
            # ('NSW = 3   (ignored when IBRION=-1)'), which pymatgen parses
            # exactly as written when IBRION has its own line, and which HEAD
            # ran correctly. A comment naming the line's OWN tag ('NSW = 3
            # (NSW=0 would be a single point)') is likewise harmless: the tag
            # is present with its intended value.
            embedded = [
                match.group(1).upper()
                for match in _EMBEDDED_ASSIGNMENT_RE.finditer(value)
                if match.group(1).upper() in _EMBEDDED_DETECTABLE_INCAR_TAGS
                and match.group(1).upper() not in parsed_keys
            ]
            if embedded:
                raise ValueError(
                    f"INCAR line for {key} carries further assignment(s) "
                    f"({', '.join(dict.fromkeys(embedded))}) inside its value: "
                    f"the parser reads everything up to the end of the line "
                    f"as {key}'s value, so those tags are silently lost. Put "
                    "each tag on its own line or separate them with ';'."
                )
        if value.startswith('"') and parsed_keys is not None:
            # The RAW-LEVEL half of the unbalanced-quote rule: a balanced
            # quoted value is returned WITHOUT its quotes by the raw reader
            # (its quoted branch requires the closing quote on the same
            # line), so a leading quote here means this line's closing quote
            # is missing. That alone is NOT sufficient to reject: pymatgen's
            # DOTALL quoted branch swallows following tags only when ANOTHER
            # quote exists later in the comment-stripped file; with no later
            # quote the alternation falls through to the plain-value branch
            # and the file parses exactly as written (SYSTEM = "run #3" --
            # whose closing quote the comment strip removes -- and a
            # forgotten quote on the LAST tag both ran fine at HEAD, and
            # rejecting them was a legacy regression). The direct evidence
            # of a swallow is a FOLLOWING raw tag missing from the parse --
            # which also catches the bool/list-typed swallowers
            # (LWAVE = ".FALSE. parses to the scalar False) that the
            # parsed-STRING check above cannot see.
            missing = [
                following_key
                for following_key, following_value in assignments[index + 1 :]
                # A BLANK-valued tag is dropped by pymatgen for an
                # independent reason (`if not val: continue` in from_str),
                # so its absence from the parse is NOT swallow evidence --
                # counting it rejected legal files that merely combined a
                # protected leading-quote value with a trailing `NPAR =`
                # (the blank-value branch below judges those on their own).
                # A swallowed region containing a blank tag still surfaces
                # through its non-blank neighbours.
                if following_value and following_key not in parsed_keys
            ]
            if missing:
                raise ValueError(
                    f"INCAR tag {key} has an unbalanced quote ({value!r}): "
                    "the parser read a quoted value up to the next quote "
                    "character in the file, silently deleting these tags: "
                    f"{', '.join(dict.fromkeys(missing))}. Balance or remove "
                    f"the quote on the {key} line."
                )
        if value:
            continue
        try:
            parsed = incar.get(key)
        except Exception:
            return
        if isinstance(parsed, str) and parsed.strip():
            raise ValueError(
                f"INCAR tag {key} has an empty value, and an empty INCAR value "
                f"continues onto the following lines: it was read as {parsed!r}, "
                "so that text is not a tag of its own any more. Give "
                f"{key} a value or delete the line."
            )
        if parsed_keys is None or index + 1 >= len(assignments):
            continue
        following = assignments[index + 1][0]
        if following in parsed_keys:
            continue
        raise ValueError(
            f"INCAR tag {following} was swallowed by {key}, which has an empty "
            "value: an empty INCAR value continues onto the following lines, so "
            f"{following} is not read at all (the parser stored it as part of "
            f"{key}={parsed!r}). Give {key} a value or delete the line."
        )


def _reject_truncated_integer_tags(incar, path: str) -> None:
    """Reject an INCAR value that the parser silently read as a different number.

    Two distinct manglings, both silent:

    * ``Incar.proc_val`` reads int-typed keys with
      ``int(re.match(r"^-?[0-9]+", v))``, so ``NSW = 1e5`` becomes 1 and the rest of
      the token is discarded. The relaxation then ran ONE ionic step instead of
      100000, wrote an unconverged CONTCAR and exited 0.
    * A Fortran ``D`` exponent stops every numeric reader in this code base: ``NSW =
      1D3`` -> 1, ``EDIFFG = -1.0D-03`` -> -1.0 (three orders of magnitude), and for
      tags pymatgen leaves as text ``NHC_PERIOD = 1D2`` -> 1 via ``_NUMERIC_RE``.
      pymatgen can even turn ``LANGEVIN_GAMMA = 1D1`` into the list ``[1, 1]``.

    Only values that genuinely disagree are rejected, so an INCAR that already ran
    correctly cannot start failing. Non-numeric text (a ``SYSTEM`` line such as
    ``D2O sample``) never reaches a comparison because it does not parse as a float.
    """

    try:
        raw_tags = _raw_incar_assignments(path, continuation_values=True)
    except OSError:
        return

    for key, raw_value in raw_tags.items():
        tokens = raw_value.split()
        if not tokens:
            continue
        token = tokens[0]
        # A trailing comma is a legal Fortran list-directed value terminator:
        # VASP reads 'NSW = 3,' as 3 and so does pymatgen, so the comma is
        # stripped BEFORE judging the token -- rejecting it (as the first
        # version of the corrupted-token branch below did) failed inputs that
        # completed correctly at HEAD, a SPEC 1.1 regression.
        normalized = _FORTRAN_EXPONENT_RE.sub("E", token.rstrip(","), count=1)
        try:
            intended = float(normalized)
        except ValueError:
            # The raw token is NOT a number. That alone is fine (SYSTEM = D2O
            # sample) -- but a NUMBER must not be invented from it by
            # extracting digits, which happens on two independent paths:
            # pymatgen's proc_val for the tags IT types ('TEBEG = 5OO' -> 5.0
            # ran the MD at 5 K), and VPMDK's own _NUMERIC_RE readers for the
            # tags pymatgen leaves as text ('CSVR_PERIOD = 5OO' -> 5, a
            # thermostat 100x stiffer than requested; 'LANGEVIN_GAMMA = 1O'
            # comes back as the LIST [1], invisible to the typed check).
            # VASP refuses these files; both paths are judged from the raw
            # token so neither parser's typing decides.
            if key in _SCALAR_NUMERIC_INCAR_TAGS and _NUMERIC_RE.search(token):
                raise ValueError(
                    f"{key} = {token!r} is not a number, and the readers "
                    "would silently extract a different number from its "
                    f"digits. VASP's INCAR format does not accept {token!r} "
                    "here; fix the value (a letter O instead of a zero is a "
                    "common cause)."
                )
            parsed = incar.get(key) if hasattr(incar, "get") else None
            if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
                continue  # genuinely non-numeric text for a non-numeric tag
            raise ValueError(
                f"{key} = {token!r} is not a number, but the parser read it "
                f"as {parsed!r} by extracting the leading digits. VASP's "
                f"INCAR format does not accept {token!r} here; fix the value "
                "(a letter O instead of a zero is a common cause)."
            )
        # The token DID parse as a number -- but that means nothing for a
        # FREE-TEXT tag: 'SYSTEM = 1D5 sample' is a title whose first token
        # happens to look like a Fortran exponent, and 'SYSTEM = Infinity
        # study' is a title, not a non-finite number. Applying the checks
        # below to every key rejected INCARs that ran correctly at HEAD (a
        # SPEC 1.1 regression). Only judge tags some reader treats as a
        # number: the known scalar-numeric set, or a tag pymatgen itself
        # typed as one.
        parsed = incar.get(key) if hasattr(incar, "get") else None
        parsed_is_numeric = isinstance(parsed, (int, float)) and not isinstance(
            parsed, bool
        )
        if key not in _SCALAR_NUMERIC_INCAR_TAGS and not parsed_is_numeric:
            continue

        if not math.isfinite(intended):
            raise ValueError(f"{key} must be a finite number; got {token!r}.")

        # What the readers in this code base will actually see.
        if normalized != token:
            match = _NUMERIC_RE.search(token)
            as_read = None
            if match is not None:
                try:
                    as_read = float(match.group(0))
                except ValueError:
                    as_read = None
            if as_read is None or not math.isclose(
                as_read, intended, rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValueError(
                    f"{key} would be read as {as_read} instead of {intended}: VASP's "
                    f"INCAR format does not accept {raw_value!r} here. Write the "
                    "value with a plain E exponent."
                )

        # The fractional-literal rule is keyed on VASP's SEMANTICS, not the
        # parser's typing: pymatgen int-types the REAL tag SPRING (repaired
        # from raw text instead, so it is exempt) and float-types the
        # integer-semantic NHC_NCHAINS/IOPT/ICHAIN (whose VPMDK coercers
        # int(float(...)) silently floored 2.7 to 2 while the byte-analogous
        # 'NSW = 2.7' was rejected).
        if (
            key in _INTEGER_SEMANTIC_INCAR_TAGS
            and not float(intended).is_integer()
        ):
            # Real VASP refuses the file ('Bad value during integer read').
            # A trailing '.' ('100.') stays legal: it is integral.
            raise ValueError(
                f"{key} = {token!r} is not an integer, and the parser would "
                "silently floor it. VASP's INCAR format does not accept a "
                "fractional value for an integer tag."
            )

        if not parsed_is_numeric:
            continue
        if isinstance(parsed, float):
            if not math.isclose(parsed, intended, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    f"{key} was read as {parsed} instead of {intended}: VASP's INCAR "
                    f"format does not accept {token!r} here. Write the value with a "
                    "plain E exponent."
                )
            continue
        if key in _REAL_TAGS_PYMATGEN_INT_TYPES:
            # pymatgen int-typed a REAL tag; _repair_mistyped_real_tags has
            # already restored the true float value in the mapping, so the
            # integer comparison below does not apply.
            continue
        if int(intended) != parsed:
            raise ValueError(
                f"{key} was read as {parsed} instead of {int(intended)}: VASP's INCAR "
                f"format does not accept {token!r} for an integer tag. Write the "
                "value in plain digits."
            )


def _repair_mistyped_real_tags(incar, path: str) -> None:
    """Restore the true float value of REAL tags pymatgen int-types.

    pymatgen's generic ``proc_val`` int-types SPRING, so ``SPRING = -5.5``
    (legal in VASP/VTST) came back as -5 and the committed tree silently ran
    with the FLOORED spring ('spring=5' printed for a requested 5.5), while
    the R142 guard's first form rejected the legal file outright (a SPEC 1.1
    regression). The raw token holds the true value; write it back into the
    mapping so every downstream reader sees what the user wrote.
    """

    if not hasattr(incar, "get"):
        return
    try:
        raw_tags = _raw_incar_assignments(path, continuation_values=True)
    except OSError:
        return
    for key in _REAL_TAGS_PYMATGEN_INT_TYPES:
        if key not in raw_tags:
            continue
        parsed = incar.get(key)
        if isinstance(parsed, bool) or not isinstance(parsed, int):
            continue  # untyped (stub) or already float: nothing to repair
        tokens = raw_tags[key].split()
        if not tokens:
            continue
        normalized = _FORTRAN_EXPONENT_RE.sub("E", tokens[0].rstrip(","), count=1)
        try:
            true_value = float(normalized)
        except ValueError:
            continue
        if not math.isclose(true_value, float(parsed), rel_tol=1e-12, abs_tol=0.0):
            # NOT `incar[key] = ...`: pymatgen's Incar.__setitem__ re-runs
            # proc_val on the stringified value, RE-FLOORING the repair to the
            # same truncated int -- the first version of this function was a
            # measured no-op against real pymatgen (it only worked against
            # plain dicts). dict.__setitem__ bypasses the re-typing; Incar IS
            # a dict subclass, and non-dict mappings fall back to plain
            # assignment.
            backing = getattr(incar, "data", None)
            if isinstance(backing, dict):
                backing[key] = true_value  # UserDict (real pymatgen Incar)
            elif isinstance(incar, dict):
                dict.__setitem__(incar, key, true_value)
            else:
                incar[key] = true_value


# Ceiling for a 'N*value' repeat count in an INCAR list tag. MAGMOM repeats
# are bounded by the atom count in practice; 1e6 is >10x beyond any real
# structure while keeping the expanded list tiny.
_MAX_INCAR_REPEAT_COUNT = 1_000_000


# Classic pymatgen list-typed keys, used only when the installed Incar has
# no INCAR_PARAMS table (the test stub, very old pymatgen): the caps must
# keep their teeth there too.
_FALLBACK_LIST_INCAR_TAGS = frozenset(
    {
        "MAGMOM",
        "LANGEVIN_GAMMA",
        "DIPOL",
        "EINT",
        "QUAD_EFG",
        "ROPT",
        "LATTICE_CONSTRAINTS",
    }
)


def _incar_tag_expands_repeats(key: str) -> bool:
    """Whether ANY parser layer expands ``N*value`` for this tag.

    pymatgen's ``proc_val`` performs the repeat expansion only for keys whose
    INCAR_PARAMS type includes "list"; SYSTEM/WANNIER90_WIN are kept as-is
    strings and unknown tags are never expanded. VPMDK's own expander reads
    MAGMOM. Judging every tag made the resource caps reject free-text titles
    such as ``SYSTEM = 1000001*study`` -- values no layer ever expands, which
    ran fine at HEAD (cross-review finding).
    """

    upper = str(key).upper()
    if upper == "MAGMOM":
        return True
    params = getattr(Incar, "INCAR_PARAMS", None)
    if isinstance(params, dict):
        entry = params.get(upper) or {}
        return "list" in str(entry.get("type", "") or "")
    return upper in _FALLBACK_LIST_INCAR_TAGS


# EXACT mirror of pymatgen's own list tokenizer (Incar.proc_val): it runs
# re.findall over the WHOLE value string, so surrounding junk is irrelevant
# and up to three numeric tokens may be joined by '*'. Hand-rolling the
# split (whitespace, then commas) let '(2000000000*1.0)' and
# '1000000*1.0x1000000*1.0' bypass every cap while the same value without
# the junk character was rejected -- the guard must tokenize the way the
# parser does, not the way the text looks (cross-review/R169 finding).
_PYMATGEN_NUM_OR_STR = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?|[\.A-Z]+"
_PYMATGEN_LIST_TOKEN_RE = re.compile(
    rf"({_PYMATGEN_NUM_OR_STR})\*?({_PYMATGEN_NUM_OR_STR})?\*?({_PYMATGEN_NUM_OR_STR})?"
)


def _pymatgen_token_expansion(groups) -> float:
    """Return the entry count one pymatgen list token expands to."""

    first, second, third = groups

    def _count(text: str) -> float:
        try:
            return max(float(text), 1.0)
        except (TypeError, ValueError):
            return 1.0

    if third:
        return _count(first) * _count(second)
    if second:
        return _count(first)
    return 1.0


def _repeat_token_expansion(token: str) -> float:
    """Return how many list entries a raw INCAR token expands to.

    Every LEADING '*'-separated segment that parses as a number is a
    multiplier: pymatgen's ``proc_val`` supports the NESTED spelling
    ``count1*count2*value`` and multiplies BOTH counts, and VPMDK's own
    ``_parse_magmom_values`` recurses to any depth, so judging only the
    first factor lets ``100000*100000*1.0`` (1e10 entries, ~80 GB) through
    a cap that correctly rejects the flat ``10000000*1.0``. The bound must
    therefore be the PRODUCT of all leading factors.
    """

    expansion = 1.0
    for segment in token.split("*")[:-1]:
        try:
            factor = float(segment)
        except ValueError:
            break
        expansion *= max(factor, 1.0)
    return max(expansion, 1.0)


def _reject_huge_repeat_counts(path: str) -> None:
    """Reject an INCAR repeat token that would allocate an absurd list.

    This must run BEFORE ``Incar.from_file``: pymatgen's ``proc_val`` expands
    ``MAGMOM = 10000000000*1.0`` into a 1e10-element Python list (~160 GB) at
    parse time, i.e. before any of the guards downstream of the parse could
    see the file -- inside the resident server process, that is an OOM kill
    of the server and every queued job. The exponent spelling ('1e10*2.0')
    escapes pymatgen's int() but detonates VPMDK's OWN expander in
    _parse_magmom_values instead, so the bound is judged from the RAW text
    where both parser layers are still ahead.
    """

    try:
        # continuation_values: pymatgen's value side crosses the newline, so
        # 'MAGMOM =' followed by the token on its own line is exactly what
        # proc_val will expand -- the raw scan must see it too.
        assignments = _raw_incar_assignment_list(path, continuation_values=True)
    except OSError:
        return
    for key, raw_value in assignments:
        if not _incar_tag_expands_repeats(key):
            # No parser layer expands this tag's value, so a numeric prefix
            # before '*' is just text (SYSTEM titles, unknown tags) and the
            # caps have nothing to bound.
            continue
        total_expansion = 0.0
        # Tokenize EXACTLY as pymatgen does (see _PYMATGEN_LIST_TOKEN_RE):
        # every earlier hand-rolled split (whitespace, then commas) was
        # bypassable by one junk character.
        for match in _PYMATGEN_LIST_TOKEN_RE.finditer(raw_value):
            if not match.group(0):
                continue
            expansion = _pymatgen_token_expansion(match.groups())
            if expansion > _MAX_INCAR_REPEAT_COUNT:
                raise ValueError(
                    f"INCAR tag {key} contains the repeat token "
                    f"{match.group(0)!r} that expands to more than "
                    f"{_MAX_INCAR_REPEAT_COUNT} values; no structure has "
                    "that many ions, and expanding it would exhaust memory "
                    "before any other validation runs."
                )
            total_expansion += expansion
        if total_expansion > _MAX_INCAR_REPEAT_COUNT:
            # The per-token cap alone is the axis-not-resource mistake again:
            # many tokens of 1e6 each still expand without bound. Bound the
            # SUM, exactly like the FFT grid's total-points cap.
            raise ValueError(
                f"INCAR tag {key} expands to {int(total_expansion)} values, "
                f"above the supported maximum of {_MAX_INCAR_REPEAT_COUNT}; "
                "no structure has that many ions."
            )


def _load_incar(path: str):
    """Return ``Incar`` contents when available, falling back to ``{}``."""

    import sys as _sys_mod

    _sys_mod.modules["vpmdk_core"]._reject_broken_input_link(path, "INCAR")
    if os.path.exists(path):
        # A FIFO planted as INCAR blocks open() forever inside the resident
        # worker (same class as the POSCAR/BCAR guard in io/inputs.py, which
        # documents the measured wedge). stat never blocks; check BEFORE the
        # raw-text scan and pymatgen's own open below.
        import sys as _sys

        _sys.modules["vpmdk_core"]._require_regular_input_file(path, "INCAR")
        _reject_huge_repeat_counts(path)
        incar = Incar.from_file(path)
        _repair_mistyped_real_tags(incar, path)
        _reject_swallowed_incar_tags(incar, path)
        _reject_truncated_integer_tags(incar, path)
        return incar
    return {}


def _warn_for_unsupported_incar_tags(
    incar,
    *,
    pseudo_scf_enabled: bool = False,
    chgcar_enabled: bool = False,
) -> None:
    """Emit warnings for INCAR options that are silently ignored."""

    import sys

    root = sys.modules["vpmdk_core"]
    supported_tags = SUPPORTED_INCAR_TAGS
    for key in getattr(incar, "keys", lambda: [])():
        if key in supported_tags:
            continue
        if pseudo_scf_enabled and key in root._PSEUDO_SCF_INCAR_TAGS:
            print(
                f"Warning: INCAR tag {key} does not affect the run and is used only "
                "for pseudo-SCF compatibility output"
            )
            continue
        if key in getattr(root, "_CHGCAR_GRID_INCAR_TAGS", frozenset()):
            if chgcar_enabled:
                continue
            print(
                f"Warning: INCAR tag {key} affects only CHGCAR grid output and "
                "WRITE_CHGCAR is not enabled; ignoring it."
            )
            continue
        if key not in supported_tags:
            print(f"INCAR tag {key} is not supported and will be ignored")


def _parse_vtst_ichain(incar) -> int:
    """Return VTST ``ICHAIN`` with the NEB default."""

    raw_value = getattr(incar, "get", lambda *_: 0)("ICHAIN", 0)
    parsed = _parse_optional_float(raw_value, key="ICHAIN")
    if parsed is None:
        return 0
    return int(parsed)


def _reject_unsupported_vtst_modes(incar) -> None:
    """Reject VTST transition-state modes that VPMDK does not implement."""

    import sys

    # This module has no module-level _root(); follow the local-import pattern
    # _warn_for_unsupported_incar_tags already uses here.
    root = sys.modules["vpmdk_core"]
    ichain = _parse_vtst_ichain(incar)
    if ichain != 0:
        raise root.UnsupportedInputError(
            "VPMDK currently implements VTST-style NEB for ICHAIN=0 only. "
            f"ICHAIN={ichain} TS methods such as dimer/lanczos are not implemented."
        )


def _is_truthy_flag(value) -> bool:
    """Return whether ``value`` expresses a truthy INCAR-style flag."""

    if value is None:
        return False
    token = str(value).strip().strip(".").upper()
    return token in {"T", "TRUE", "1", "YES", "Y"}


def _is_neb_like_incar(incar) -> bool:
    """Detect whether INCAR appears to describe a NEB-style calculation."""

    if not hasattr(incar, "get"):
        return False

    images_value = incar.get("IMAGES")
    if images_value is not None:
        match = _NUMERIC_RE.search(str(images_value))
        if match is not None:
            try:
                if int(float(match.group(0))) > 0:
                    return True
            except (ValueError, OverflowError):
                # A malformed IMAGES is tolerated (fall back to SPRING/LCLIMB):
                # catch OverflowError too, since int(float()) raises it for a
                # huge value (e.g. a several-hundred-digit IMAGES) while raising
                # ValueError for NaN -- otherwise the two malformed forms would be
                # handled inconsistently (one ignored, one an uncaught crash /
                # server calculation_error).
                pass

    if "SPRING" in getattr(incar, "keys", lambda: [])():
        return True

    if _is_truthy_flag(incar.get("LCLIMB")):
        return True

    return False


def _parse_neb_image_count(incar) -> int | None:
    """Return ``IMAGES`` value when parseable and non-negative."""

    if not hasattr(incar, "get"):
        return None
    raw_value = incar.get("IMAGES")
    if raw_value is None:
        return None
    parsed = _parse_optional_float(raw_value, key="IMAGES")
    if parsed is None:
        return None
    if not math.isfinite(parsed):
        # _parse_optional_float returns inf/nan for a huge or non-numeric IMAGES
        # (e.g. 1e400, a several-hundred-digit integer, or "nan"). int() would
        # raise OverflowError/ValueError on those; ignore the hint instead, the
        # same way _is_neb_like_incar tolerates a malformed IMAGES.
        print(f"Warning: IMAGES={raw_value} is invalid; ignoring NEB image count hint.")
        return None
    count = int(parsed)
    if count < 0:
        print(f"Warning: IMAGES={raw_value} is invalid; ignoring NEB image count hint.")
        return None
    return count


def _parse_optional_float(value, *, key: str):
    """Attempt to convert ``value`` to ``float`` with warning on failure."""

    if value is None:
        return None
    candidate = value
    # Real pymatgen represents some nominally scalar INCAR values (notably a
    # one-species LANGEVIN_GAMMA) as a singleton list.  The lightweight test
    # parser and older pymatgen releases return the scalar directly.
    if isinstance(candidate, (list, tuple)) and len(candidate) == 1:
        candidate = candidate[0]
    if isinstance(candidate, str):
        match = _NUMERIC_RE.search(candidate)
        if match is not None:
            candidate = match.group(0)
        else:
            candidate = candidate.strip()
    try:
        parsed = float(candidate)
    except (TypeError, ValueError, OverflowError):
        # OverflowError, not ValueError: real pymatgen parses a several-hundred-
        # digit INCAR literal into a Python INT, and float(huge_int) raises
        # OverflowError. Only the STRING spelling reaches the isfinite guard below
        # (float("9"*400) is inf), so without this the int path escaped from the
        # VTST/NEB parse sites that run OUTSIDE _read_workdir_input -- surfacing as
        # calculation_error (exit 2, documented RETRYABLE) for a permanently broken
        # INCAR, and as a raw traceback in one-shot. Matches the OverflowError
        # guards the sibling int(float(...)) coercers already have.
        print(f"Warning: Unable to parse {key}; ignoring value {value}")
        return None
    if not math.isfinite(parsed):
        # inf/nan are not usable numeric values ("1e400"/a several-hundred-digit
        # integer overflow to inf; "nan" -> nan). Treat them as unparseable so
        # every caller falls back to its default and a downstream int(parsed)
        # never raises OverflowError/ValueError (which the server would otherwise
        # misclassify as calculation_error vs one-shot exit 1). This single guard
        # covers all int(_parse_optional_float(...)) sites (ICHAIN/IOPT/IMAGES/...).
        print(f"Warning: Unable to parse {key}; ignoring value {value}")
        return None
    return parsed


def _normalize_mdalgo(requested: int) -> int:
    """Map an MD algorithm request to the one VPMDK will actually run.

    _select_md_dynamics tests MDALGO against 1, (2, 4), 3 and 5 and otherwise
    falls through to plain velocity-Verlet, so ANY other value -- a typo, or one
    of VASP's constrained-MD algorithms -- silently produced an NVE trajectory
    while OUTCAR and vasprun.xml still reported the REQUESTED MDALGO, i.e. the
    run claimed a thermostatted ensemble it never sampled (measured: SP=/SK=
    identically zero, temperature free-drifting).

    Normalizing here (rather than in the dynamics selector) makes the recorded
    metadata match the trajectory as well. It warns instead of rejecting,
    exactly like _normalize_isif: SERVER_MODE_SPEC §1.1 keeps inputs that
    currently complete completing.
    """

    if requested in SUPPORTED_MDALGO_VALUES:
        return requested
    print(
        f"Warning: MDALGO={requested} is not supported; defaulting to MDALGO=0 "
        "(velocity-Verlet NVE) behavior."
    )
    return 0


def _normalize_isif(requested: int) -> int:
    """Map request to supported ISIF behaviour while preserving warnings."""

    if requested not in SUPPORTED_ISIF_VALUES:
        print(
            "Warning: ISIF="
            f"{requested} is not fully supported; defaulting to ISIF=2 behavior."
        )
        return 2
    if requested in (0, 1, 2):
        return 2
    return requested


def _parse_langevin_gamma(value):
    """Return one friction coefficient from VASP's per-species LANGEVIN_GAMMA.

    VASP takes ONE GAMMA PER POTCAR SPECIES, so real pymatgen returns a list with
    one entry per species -- a multi-element list is the normal multi-species
    spelling, not a malformed value. ``_parse_optional_float`` unwraps only a
    singleton, so every multi-species INCAR fell through to ``float([...])`` ->
    TypeError -> the tag was DISCARDED, and the Langevin setup then silently used
    its 1.0 default friction: different dynamics than the INCAR requested, with
    only a generic "unable to parse" line to go on.

    VPMDK's Langevin integration applies a single scalar friction, so a uniform
    list unwraps cleanly. Entries that genuinely differ cannot be represented;
    say so explicitly and use the first species' value, which is far closer to
    the request than the unrelated 1.0 default.

    LANGEVIN_GAMMA is the only per-species tag in _extract_thermostat_parameters
    (ANDERSEN_PROB / CSVR_PERIOD / NHC_NCHAINS / NHC_PERIOD are all scalars), so
    the generic float parser is deliberately left unchanged.
    """

    if isinstance(value, (list, tuple)) and len(value) > 1:
        first = value[0]
        if not all(entry == first for entry in value):
            print(
                "Warning: per-species LANGEVIN_GAMMA is not supported; using the "
                f"first value {first} for all atoms (requested {list(value)})."
            )
        parsed_first = _parse_optional_float(first, key="LANGEVIN_GAMMA")
        _reject_absurd_md_magnitude("LANGEVIN_GAMMA", parsed_first)
        return parsed_first
    parsed = _parse_optional_float(value, key="LANGEVIN_GAMMA")
    _reject_absurd_md_magnitude("LANGEVIN_GAMMA", parsed)
    return parsed


def _extract_thermostat_parameters(incar) -> Dict[str, float]:
    """Collect thermostat keywords from ``incar`` with validation."""

    params: Dict[str, float] = {}
    keys = (
        "ANDERSEN_PROB",
        "LANGEVIN_GAMMA",
        "CSVR_PERIOD",
        "NHC_NCHAINS",
        "NHC_PERIOD",
    )
    for key in keys:
        if hasattr(incar, "__contains__") and key in incar:
            value = incar[key]
            # NHC_NCHAINS deliberately has NO special case: it was the only
            # INCAR scalar parsed with a bare int(float(value)), so the legal
            # trailing comma ('5,' -- the Fortran terminator R138 established
            # this parser must honour, and which every sibling reads through
            # the shared extractor) dropped the tag and the run silently
            # sampled the DEFAULT chain length. Integer-ness is enforced by
            # the raw-level fractional/truncated guards keyed on
            # _INTEGER_SEMANTIC_INCAR_TAGS, and _select_md_dynamics coerces
            # with int() at use.
            if key == "LANGEVIN_GAMMA":
                parsed = _parse_langevin_gamma(value)
            else:
                parsed = _parse_optional_float(value, key=key)
            if parsed is not None:
                # The R139 magnitude bound covered LANGEVIN_GAMMA only (via
                # its own parser); SMASS and NHC_PERIOD were left unbounded
                # and 1e300 raised a raw OverflowError from tdamp**2 inside
                # ASE's thermostat -- one-shot exit 1 with a traceback, server
                # exit 2 RETRYABLE for an unfixable input. Every thermostat
                # scalar handed to ASE gets the same bound.
                _reject_absurd_md_magnitude(key, float(parsed))
                params[key] = float(parsed)
    return params


def _load_incar_settings(incar) -> IncarSettings:
    """Translate INCAR dictionary-like object into :class:`IncarSettings`."""

    if not hasattr(incar, "get"):
        return IncarSettings()

    nsw = int(float(incar.get("NSW", 0)))
    ibrion = int(float(incar.get("IBRION", -1)))
    if "IBRION" not in incar and nsw > 1:
        # Real VASP's documented default for NSW>1 is IBRION=0 (MD); VPMDK's
        # absent-tag default of -1 runs a SINGLE POINT for the same file --
        # silently. The default stays (SPEC 1.1); the divergence is disclosed
        # like LCLIMB's and ANDERSEN_PROB's.
        print(
            f"Warning: NSW={nsw} with IBRION omitted runs a SINGLE POINT in "
            "VPMDK (default IBRION=-1), while real VASP would default to "
            "IBRION=0 (MD) here. Write IBRION explicitly for the mode you "
            "want."
        )
    ediffg = float(incar.get("EDIFFG", -0.02))
    if not math.isfinite(ediffg):
        # nan/inf makes BOTH `ediffg > 0` and `ediffg < 0` False, so force_limit
        # silently fell back to fmax=0.05 -- a convergence criterion the INCAR
        # never asked for, with no warning at all. Treat it like every other
        # unparseable numeric tag: warn and use the documented default.
        print(f"Warning: Unable to parse EDIFFG; ignoring value {incar.get('EDIFFG')}")
        ediffg = -0.02
    pstress = None
    if "PSTRESS" in incar:
        pstress = _parse_optional_float(incar.get("PSTRESS", 0.0), key="PSTRESS")
        if pstress is not None and abs(pstress) > _MAX_PSTRESS_KBAR:
            # Huge-but-finite PSTRESS (an exponent typo like 1e300 for 1e3)
            # passes the finiteness check and reaches ASE's cell filter and
            # BFGS, where the step-length norm overflows: every step scales
            # to zero and the run COMPLETES with exit 0, CONTCAR identical to
            # POSCAR -- the requested pressure silently had no effect -- while
            # the OUTCAR pressure fields carry 300-digit values. Same
            # absurd-finite class as the ENCUT and cell-width bounds.
            raise ValueError(
                f"PSTRESS = {pstress:g} kBar exceeds the supported magnitude "
                f"of {_MAX_PSTRESS_KBAR:g} kBar; check the exponent."
            )
    tebeg_default = 300.0
    tebeg_value = incar.get("TEBEG", tebeg_default)
    parsed_tebeg = _parse_optional_float(tebeg_value, key="TEBEG")
    tebeg = parsed_tebeg if parsed_tebeg is not None else tebeg_default
    _reject_absurd_md_magnitude("TEBEG", tebeg)

    teend_value = incar.get("TEEND", tebeg)
    parsed_teend = _parse_optional_float(teend_value, key="TEEND")
    teend = parsed_teend if parsed_teend is not None else tebeg
    _reject_absurd_md_magnitude("TEEND", teend)
    if "POTIM" in incar:
        potim = float(incar.get("POTIM", 2.0))
        if not math.isfinite(potim):
            # The one numeric INCAR tag still parsed with a bare float(): nan/inf
            # passed the input phase and only failed later -- mid-MD as a raw
            # ValueError the server reports as calculation_error (exit 2,
            # documented RETRYABLE) for a permanently invalid INCAR, or, under
            # IBRION=5/6, as an all-NaN hessian written out as a SUCCESSFUL run.
            # Reject it here, where every sibling tag is already rejected (exit 1).
            raise ValueError(
                f"POTIM must be a finite number; got {incar.get('POTIM')!r}."
            )
        _reject_absurd_md_magnitude("POTIM", potim)
    elif ibrion in {5, 6}:
        potim = 0.015
    else:
        potim = 2.0
    nfree = None
    if "NFREE" in incar:
        parsed_nfree = _parse_optional_float(incar.get("NFREE"), key="NFREE")
        if parsed_nfree is not None:
            if not float(parsed_nfree).is_integer():
                raise ValueError("NFREE must be an integer.")
            nfree = int(parsed_nfree)
    symprec = 1e-5
    if "SYMPREC" in incar:
        parsed_symprec = _parse_optional_float(incar.get("SYMPREC"), key="SYMPREC")
        if parsed_symprec is not None:
            if parsed_symprec <= 0.0:
                raise ValueError("SYMPREC must be positive.")
            symprec = float(parsed_symprec)
    smass = (
        _parse_optional_float(incar.get("SMASS"), key="SMASS")
        if "SMASS" in incar
        else None
    )
    # SMASS feeds tdamp**2 in ASE's Nose-Hoover chain (OverflowError at 1e300)
    # and, when negative, |SMASS| becomes the Langevin friction -- which
    # skipped the LANGEVIN_GAMMA bound entirely. Bound it BEFORE the MDALGO
    # promotion below so both routes are covered.
    _reject_absurd_md_magnitude("SMASS", smass)
    mdalgo = int(float(incar.get("MDALGO", 0)))
    if mdalgo == 0 and smass is not None:
        if smass < 0:
            mdalgo = 3
        elif smass > 0:
            mdalgo = 2
    # After the SMASS derivation, not before: an explicit out-of-range MDALGO is
    # not "MDALGO omitted", so SMASS must not promote it to a thermostat the
    # warning below says the run will not use.
    mdalgo = _normalize_mdalgo(mdalgo)
    thermostat_params = _extract_thermostat_parameters(incar)
    default_isif = 0 if ibrion == 0 else 2
    requested_isif = int(float(incar.get("ISIF", default_isif)))
    normalized_isif = _normalize_isif(requested_isif)
    stress_isif = (
        requested_isif if requested_isif in SUPPORTED_ISIF_VALUES else normalized_isif
    )

    return IncarSettings(
        nsw=nsw,
        ibrion=ibrion,
        ediffg=ediffg,
        isif=normalized_isif,
        stress_isif=stress_isif,
        pstress=pstress,
        tebeg=tebeg,
        teend=teend,
        potim=potim,
        nfree=nfree,
        symprec=symprec,
        mdalgo=mdalgo,
        smass=smass,
        thermostat_params=thermostat_params,
    )


def _should_write_energy_csv(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests CSV output of ionic energies."""

    value = str(bcar_tags.get("WRITE_ENERGY_CSV", "0")).lower()
    return value in {"1", "true", "yes", "on"}


def _should_write_lammps_trajectory(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests LAMMPS-style trajectory output."""

    value = str(bcar_tags.get("WRITE_LAMMPS_TRAJ", "0")).lower()
    return value in {"1", "true", "yes", "on"}


def _should_write_pseudo_scf(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests pseudo electronic-step compatibility output."""

    raw = bcar_tags.get("WRITE_PSEUDO_SCF", bcar_tags.get("WRITE_OSZICAR_PSEUDO_SCF", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _should_write_chgcar(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests CHGCAR output."""

    raw = bcar_tags.get("WRITE_CHGCAR", "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _should_write_oszicar_pseudo_scf(bcar_tags: Dict[str, str]) -> bool:
    """Backward-compatible alias for :func:`_should_write_pseudo_scf`."""

    return _should_write_pseudo_scf(bcar_tags)


def _get_lammps_trajectory_interval(bcar_tags: Dict[str, str]) -> int:
    """Return the LAMMPS trajectory write interval requested in BCAR."""

    import sys

    raw = bcar_tags.get("LAMMPS_TRAJ_INTERVAL", "1")
    return_value = sys.modules["vpmdk_core"]._coerce_int_tag(raw, "LAMMPS_TRAJ_INTERVAL")
    if return_value <= 0:
        raise ValueError("LAMMPS_TRAJ_INTERVAL must be at least 1")
    return return_value
