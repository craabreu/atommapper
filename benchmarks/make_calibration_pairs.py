"""Build the calibration benchmark, ``src/atommapper/calibration_pairs.tsv``.

The benchmark is synthetic: every molecule is put together here, as SMILES text, from the
building blocks written below (amino acid residues, rings, ring systems, linkers and end
groups). No molecule is taken from a database. The calls are pairs of related molecules of
the kinds a mapper meets: chains of rings and linkers with one piece changed, ring systems
of many rings, macrocycles, linear and cyclic peptides with residues changed, the same
molecules with one stereo centre inverted, and a few unrelated pairs.

Run it from the repository root to write the file again::

    python benchmarks/make_calibration_pairs.py

The file it writes is the same, byte for byte, on every run and with every version of
Python and RDKit that reads these SMILES:

- the choices come from a generator of this script's own (``Draw``), started from ``SEED``;
- a molecule is written as the text of its pieces, in order, never as RDKit's canonical
  SMILES, which changes between versions. RDKit only says that a molecule is valid (it
  parses, and its stereo centres get CIP labels), counts its heavy atoms and tells whether
  two texts are the same molecule;
- no call is timed. ``SKIP`` names the candidates left out because they did not end within
  the benchmark's limit when it was made.

A version of RDKit that judged one of these molecules otherwise would change the file. The
file is committed, so the package does not need this script: the script is the file's
provenance, and ``tests/test_calibration.py`` checks that it still writes the same file
(made with RDKit 2026.09.2).

The calls were chosen to take about two minutes on one core of the reference machine, one
in five of them half a virtual second or more, and to make the three stages of the mapper
work: most large pairs are slow in the fragment search, a peptide against itself with one
residue mirrored is slow in refinement.
"""

import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler

SEED = 20261118
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "atommapper" / "calibration_pairs.tsv"
BANDS = (50, 70, 90, 120, 200)
"""Upper ends of the bands of a pair's heavy atoms (both molecules); a sixth band is above."""

Call = tuple[str, str, bool]
"""Source SMILES, target SMILES, ``allow_stereo_inversion``."""
Piece = tuple[str, bool, bool]
"""A piece of a chain: its SMILES text, bonded to what comes before by its first atom and to
what comes after by its last open atom, and whether each of those two atoms may be bonded to
an atom other than carbon. It may if it is a carbon itself, unless the result would be an
odd group (a hemiaminal, an ynamine): then it is marked as if it were no carbon."""


class Draw:
    """A linear congruential generator (Knuth's MMIX constants), so that the choices do not
    depend on the Python version, as those of ``random`` may."""

    def __init__(self, seed: int) -> None:
        self.state = seed % 2**64

    def below(self, n: int) -> int:
        """An integer from 0 to ``n - 1``."""
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 33) % n

    def between(self, low: int, high: int) -> int:
        """An integer from ``low`` to ``high``, both included."""
        return low + self.below(high - low + 1)

    def pick(self, items):
        return items[self.below(len(items))]


# ---- Building blocks: chains of rings -------------------------------------------------

RINGS: list[Piece] = [
    ("c1ccc(cc1)", True, True),  # 1,4-phenylene
    ("c1cccc(c1)", True, True),  # 1,3-phenylene
    ("c1ccc(nc1)", True, True),  # pyridine
    ("c1ncc(cn1)", True, True),  # pyrimidine
    ("C1CCC(CC1)", True, True),  # cyclohexane
    ("N1CCN(CC1)", False, False),  # piperazine
    ("C1CCN(CC1)", True, False),  # piperidine, carbon first
    ("N1CCC(CC1)", False, True),  # piperidine, nitrogen first
    ("c1ccc2cc(ccc2c1)", True, True),  # naphthalene
    ("c1nc2ccc(cc2[nH]1)", True, True),  # benzimidazole
    ("c1cc2cc(ccc2[nH]1)", True, True),  # indole
    ("c1ccc2nc(ccc2c1)", True, True),  # quinoline
    ("c1csc(n1)", True, True),  # thiazole
    ("c1ccc(o1)", True, True),  # furan
    ("c1ccc(s1)", True, True),  # thiophene
    ("C1CC(C1)", True, True),  # cyclobutane
    ("n1cc(cn1)", False, True),  # pyrazole
    ("C1CCC2(CC1)CCN(CC2)", True, False),  # a spiro pair of six-membered rings
]
DECORATED: dict[str, list[Piece]] = {
    # The same ring with one hydrogen replaced: a substituent edit on the scaffold.
    "c1ccc(cc1)": [
        ("c1cc(F)c(cc1)", True, True),
        ("c1cc(C)c(cc1)", True, True),
        ("c1cc(OC)c(cc1)", True, True),
        ("c1cc(Cl)c(cc1)", True, True),
        ("c1cc(C#N)c(cc1)", True, True),
    ],
    "c1cccc(c1)": [("c1cc(F)cc(c1)", True, True), ("c1cc(C)cc(c1)", True, True)],
    "C1CCC(CC1)": [("C1CC(C)C(CC1)", True, True), ("C1CC(O)C(CC1)", True, True)],
    "c1ccc2cc(ccc2c1)": [("c1ccc2cc(c(F)cc2c1)", True, True), ("c1ccc2cc(c(C)cc2c1)", True, True)],
    "C1CCN(CC1)": [("C1CC(C)N(CC1)", True, False)],
}
LINKERS: list[Piece] = [
    ("", True, True),  # a bond; the pieces on either side decide
    ("C", True, True),
    ("CC", True, True),
    ("CCC", True, True),
    ("O", False, False),
    ("OC", False, True),
    ("CO", True, False),
    ("C(=O)N", True, False),
    ("NC(=O)", False, True),
    ("S(=O)(=O)N", False, False),
    ("NC(=O)N", False, False),
    ("N", False, False),
    ("N(C)", False, False),
    ("C(=O)", True, True),
    ("CCO", True, False),
    ("CCN", True, False),
    ("C#C", False, False),
    ("[C@H](C)", True, True),
    ("[C@@H](C)", True, True),
    ("[C@H](O)", False, False),
    ("[C@@H](O)", False, False),
]
HEADS: list[tuple[str, bool]] = [
    # What a chain starts with, and whether its bonding atom is a carbon.
    ("C", True),
    ("CC", True),
    ("FC(F)(F)", True),
    ("N#C", True),
    ("CO", False),
    ("O=C(O)", True),
    ("NC(=O)", True),
    ("CN(C)", False),
    ("F", False),
    ("Cl", False),
    ("O", False),
    ("N", False),
    ("c1ccccc1", True),
    ("C1CC1", True),
    ("C1COCCN1", False),
    ("CC(C)", True),
]
TAILS: list[tuple[str, bool]] = [
    ("C", True),
    ("F", False),
    ("Cl", False),
    ("OC", False),
    ("C(F)(F)F", True),
    ("C#N", True),
    ("O", False),
    ("N", False),
    ("N(C)C", False),
    ("C(C)C", True),
    ("C(=O)O", True),
    ("C(N)=O", True),
    ("c1ccccc1", True),
    ("C1CC1", True),
    ("N1CCOCC1", False),
    ("c1ccncc1", True),
    ("CO", True),
    ("C(C)(C)C", True),
]

# ---- Building blocks: ring systems of many rings -------------------------------------

SYSTEMS: list[Piece] = [
    # Fused, bridged and spiro systems of three to five rings, as pieces of a chain. Generic
    # carbon skeletons and their aza and oxa analogues, written for this benchmark.
    ("C12CC3CC(C1)CC(C3)(C2)", True, True),  # adamantane
    ("C1CCC2C(C1)CCC1C2CCC2(C)C(CCC12)", True, True),  # four fused rings, an angular methyl
    ("C1CC[C@H]2[C@@H](C1)CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](CC[C@H]12)", True, True),
    ("c1ccc2c(c1)Cc1cc(ccc12)", True, True),  # fluorene
    ("c1ccc2c(c1)[nH]c1cc(ccc12)", True, True),  # carbazole
    ("c1ccc2c(c1)Oc1cc(ccc1C2)", True, True),  # xanthene
    ("C1CC2CCC1CC2", True, True),  # bicyclo[2.2.2]octane
    ("C1CC2CC1CC2", True, True),  # norbornane
    ("C1CCC2CC3CC(CCC3CC2C1)", True, True),  # three fused six-membered rings in a row
    ("c1ccc2c(c1)CCc1cc(ccc1N2)", True, True),  # two benzene rings on a seven-membered one
    ("C1Cc2ccccc2C2CCC(CC12)", True, True),  # a benzene ring on two saturated ones
    ("C1C[C@@H]2CC[C@H](C1)N2", True, False),  # an 8-azabicyclo[3.2.1]octane
    ("C1CN2CCC1C(C2)", True, True),  # a 1-azabicyclo[2.2.2]octane
    ("C1CC2(CCC1)OCC1(CO2)CCC(CC1)", True, True),  # a dispiro ketal of three rings
    ("C1CC2C3CCC(C3)C2C1", True, True),  # a tricycle of five-membered rings
    ("c1cc2ccc3ccc(cc3c2cc1)", True, True),  # phenanthrene
    ("C1CC2CCC3C(CCC4CC(CCC34)O2)C1", True, True),  # fused and bridged rings, one an oxane
]

# ---- Building blocks: peptides ----------------------------------------------------------

SIDE_CHAINS = {
    "A": "C",
    "V": "C(C)C",
    "L": "CC(C)C",
    "I": "[C@@H](C)CC",
    "S": "CO",
    "T": "[C@H](C)O",
    "F": "Cc1ccccc1",
    "Y": "Cc1ccc(O)cc1",
    "W": "Cc1c[nH]c2ccccc12",
    "D": "CC(=O)O",
    "E": "CCC(=O)O",
    "N": "CC(N)=O",
    "Q": "CCC(N)=O",
    "K": "CCCCN",
    "R": "CCCNC(N)=N",
    "M": "CCSC",
    "H": "Cc1c[nH]cn1",
}
RESIDUES = "GP" + "".join(SIDE_CHAINS)
"""One-letter codes of the residues used; a lower-case letter is the D form."""


def residue(code: str) -> str:
    """A residue as text from its nitrogen to its carbonyl carbon, left open."""
    letter, mirror = code.upper(), code.islower()
    if letter == "G":
        return "NCC(=O)"
    if letter == "P":
        return f"N1CCC[C@{'@' if mirror else ''}H]1C(=O)"
    return f"N[C@{'' if mirror else '@'}H]({SIDE_CHAINS[letter]})C(=O)"


def linear_peptide(sequence: str, acetyl: bool = False, amide: bool = False) -> str:
    return (
        ("CC(=O)" if acetyl else "")
        + "".join(residue(code) for code in sequence)
        + ("N" if amide else "O")
    )


def cyclic_peptide(sequence: str) -> str:
    """Head to tail: the first nitrogen and the last carbonyl carbon close ring 99."""
    parts = [residue(code) for code in sequence]
    parts[0] = "N%99" + parts[0][1:]
    parts[-1] = parts[-1][: -len("C(=O)")] + "C%99(=O)"
    return "".join(parts)


# ---- Molecules ------------------------------------------------------------------------


def join(head: tuple[str, bool], pieces: list[Piece], tail: tuple[str, bool]) -> str | None:
    """The chain as text, or None if two atoms other than carbon would be bonded."""
    text, carbon = head
    for smiles, first, last in pieces:
        if smiles:
            if not (carbon or first):
                return None
            text, carbon = text + smiles, last
    if not (carbon or tail[1]):
        return None
    return text + tail[0]


def macrocycle(pieces: list[Piece], group: str = "") -> str | None:
    """The chain closed into a ring (ring 99) through a carbon at each end, the first of
    them with ``group`` on it."""
    return join((f"C%99({group})" if group else "C%99", True), pieces, ("C%99", True))


def atoms(smiles: str) -> int:
    """The heavy atoms of a molecule, or 0 if RDKit does not take it: it must parse, and
    its stereo centres must get their CIP labels, which the mapper asks for and which RDKit
    gives up on for some long chains of rings."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except RuntimeError:
        return 0
    return mol.GetNumHeavyAtoms()


def same(a: str, b: str) -> bool:
    return Chem.CanonSmiles(a) == Chem.CanonSmiles(b)


def flipped(text: str, draw: Draw) -> str | None:
    """``text`` with one stereo centre inverted, or None if it marks none."""
    marks = [i for i in range(len(text)) if text[i] == "@" and text[i - 1] != "@"]
    if not marks:
        return None
    i = draw.pick(marks)
    if text[i + 1] == "@":
        return text[:i] + text[i + 1 :]
    return text[:i] + "@" + text[i:]


# ---- Families of calls ----------------------------------------------------------------


def random_chain(
    draw: Draw, rings: int, pool: list[Piece], long_linkers: bool = False
) -> list[Piece]:
    """Rings from ``pool`` with a linker between two of them, as pieces. Linkers are drawn
    again until no two atoms other than carbon meet."""
    linkers = [*LINKERS, ("CCCC", True, True), ("CCOCC", True, True)] if long_linkers else LINKERS
    pieces = [draw.pick(pool)]
    while len(pieces) < 2 * rings - 1:
        ring = draw.pick(pool)
        while True:
            text, first, last = linker = draw.pick(linkers)
            before = pieces[-1][2]
            if (before or ring[1]) if not text else (before or first) and (last or ring[1]):
                break
        pieces += [linker, ring]
    return pieces


def edited(draw: Draw, pieces: list[Piece], pool: list[Piece]) -> list[Piece]:
    """The pieces with one changed: a ring or a linker replaced, or a ring decorated."""
    out = list(pieces)
    i = draw.below(len(out))
    kind = draw.below(3)
    if i % 2:
        out[i] = draw.pick(LINKERS)
    elif kind == 0 and out[i][0] in DECORATED:
        out[i] = draw.pick(DECORATED[out[i][0]])
    else:
        out[i] = draw.pick(pool)
    return out


def chains(draw: Draw, pool: list[Piece], most_rings: int) -> Iterator[Call]:
    """Pairs of chains of rings: one end group changed (a matched pair), one piece changed,
    a ring with its linker added, or one stereo centre inverted (two times in ten, when the
    chain has one)."""
    while True:
        pieces = random_chain(draw, draw.between(1, most_rings), pool)
        head, tail = draw.pick(HEADS), draw.pick(TAILS)
        source = join(head, pieces, tail)
        kind = draw.below(10)
        allow = False
        if kind < 4:
            target = join(head, pieces, draw.pick(TAILS))
        elif kind < 5:
            target = join(draw.pick(HEADS), pieces, tail)
        elif kind < 7:
            target = join(head, edited(draw, pieces, pool), tail)
        elif kind < 8:
            more = [*pieces, draw.pick(LINKERS), draw.pick(pool)]
            target = join(head, more, tail)
        else:
            # Inverted, and half the time with inversions forbidden: the mapper must then
            # leave the centre out and search for what to keep around it.
            target = flipped(source, draw) if source else None
            allow = draw.below(2) == 1
        if source and target:
            if kind < 8:
                allow = "@" in source + target and draw.below(2) == 1
            yield source, target, allow


def macrocycles(draw: Draw) -> Iterator[Call]:
    """Pairs of macrocycles: the same ring with another group on one of its atoms or, one
    time in five, with one piece of the ring changed."""
    while True:
        pieces = random_chain(draw, draw.between(2, 5), RINGS, long_linkers=True)
        pieces = [("CCC", True, True), *pieces, ("CCCC", True, True)]
        groups = [draw.pick(TAILS)[0] for _ in range(2)]
        if draw.below(5) == 0:
            source, target = macrocycle(pieces), macrocycle(edited(draw, pieces, RINGS))
        else:
            source, target = (macrocycle(pieces, group) for group in groups)
        if source and target:
            yield source, target, "@" in source + target


def peptides(draw: Draw, cyclic: bool, shortest: int, longest: int) -> Iterator[Call]:
    """Pairs of peptides: one to three residues replaced, one residue removed, the ends
    capped, or one residue in its D form."""
    while True:
        n = draw.between(shortest, longest)
        sequence = [draw.pick(RESIDUES) for _ in range(n)]
        other = list(sequence)
        kind = draw.below(10)
        allow = draw.below(2) == 1
        caps = (False, False)
        if kind < 6:
            for _ in range(draw.between(1, 3) if kind < 3 else 1):
                other[draw.below(n)] = draw.pick(RESIDUES)
        elif kind < 8:
            i = draw.below(n)
            other[i] = other[i].lower()
        elif kind < 9 and n > 2:
            del other[draw.below(n)]
        else:
            caps = (True, draw.below(2) == 1)
            other[draw.below(n)] = draw.pick(RESIDUES)
        if cyclic:
            yield cyclic_peptide("".join(sequence)), cyclic_peptide("".join(other)), allow
        else:
            yield (
                linear_peptide("".join(sequence)),
                linear_peptide("".join(other), *caps),
                allow,
            )


def cut_chains(draw: Draw, pool: list[Piece], most_rings: int) -> Iterator[Call]:
    """A long chain of rings against its first one to three rings: most of it removed."""
    while True:
        pieces = random_chain(draw, draw.between(4, most_rings), pool)
        head, tail = draw.pick(HEADS), draw.pick(TAILS)
        kept = pieces[: 2 * draw.between(1, 3) - 1]
        source, target = join(head, pieces, tail), join(head, kept, draw.pick(TAILS))
        if source and target:
            yield source, target, ("@" in source) and draw.below(2) == 1


def cut_peptides(draw: Draw) -> Iterator[Call]:
    """A peptide against its first two to four residues."""
    while True:
        n = draw.between(6, 30)
        sequence = "".join(draw.pick(RESIDUES) for _ in range(n))
        kept = sequence[: draw.between(2, 4)]
        yield linear_peptide(sequence), linear_peptide(kept), draw.below(2) == 1


def mirrored_residues(draw: Draw) -> Iterator[Call]:
    """A short peptide against itself with one residue in its D form, inversions forbidden:
    the mapper must leave that centre out and search for what to keep around it, which is
    the work of refinement."""
    while True:
        sequence = [draw.pick(RESIDUES[1:]) for _ in range(draw.between(2, 4))]
        other = list(sequence)
        i = draw.below(len(other))
        other[i] = other[i].lower()
        yield linear_peptide("".join(sequence)), linear_peptide("".join(other)), False


def unrelated(draw: Draw) -> Iterator[Call]:
    """Pairs of chains made apart from each other: little in common."""
    while True:
        rings = draw.between(1, 9)
        pair = [
            join(draw.pick(HEADS), random_chain(draw, rings, RINGS + SYSTEMS), draw.pick(TAILS))
            for _ in range(2)
        ]
        if pair[0] and pair[1]:
            yield pair[0], pair[1], False


FAMILIES: dict[str, Callable[[Draw], Iterator[Call]]] = {
    "chain": lambda draw: chains(draw, RINGS, 6),
    "long-chain": lambda draw: chains(draw, RINGS, 12),
    "systems": lambda draw: chains(draw, SYSTEMS, 6),
    "mixed": lambda draw: chains(draw, RINGS + SYSTEMS, 7),
    "macrocycle": macrocycles,
    "peptide": lambda draw: peptides(draw, False, 2, 12),
    "long-peptide": lambda draw: peptides(draw, False, 10, 24),
    "cyclic-peptide": lambda draw: peptides(draw, True, 5, 20),
    "unrelated": unrelated,
    "cut-chain": lambda draw: cut_chains(draw, RINGS, 22),
    "cut-mixed": lambda draw: cut_chains(draw, RINGS + SYSTEMS, 13),
    "cut-peptide": cut_peptides,
    "mirrored-residue": mirrored_residues,
}
"""The families, in the order that seeds them."""

QUOTAS: dict[str, tuple[int, int, int, int, int, int]] = {
    # Calls of each family in each band of pair size, smallest band first. Most pairs of
    # two large molecules take seconds, so the bands of large pairs are mostly a large
    # molecule against a small part of it, and the slow calls stay about one in five.
    "chain": (39, 26, 20, 16, 0, 0),
    "long-chain": (0, 0, 0, 0, 2, 2),
    "systems": (0, 8, 10, 8, 4, 0),
    "mixed": (0, 3, 6, 6, 4, 0),
    "macrocycle": (0, 0, 2, 2, 2, 0),
    "peptide": (6, 6, 7, 6, 4, 0),
    "long-peptide": (0, 0, 0, 0, 0, 3),
    "cyclic-peptide": (0, 0, 0, 2, 2, 2),
    "unrelated": (3, 3, 3, 2, 2, 0),
    "cut-chain": (0, 0, 0, 3, 14, 23),
    "cut-mixed": (0, 0, 0, 2, 4, 4),
    "cut-peptide": (0, 0, 0, 3, 12, 16),
    "mirrored-residue": (2, 4, 2, 0, 0, 0),
}

SKIP: frozenset[str] = frozenset(
    {
        "systems-9",
        "systems-10",
        "systems-18",
        "systems-20",
        "systems-26",
        "systems-31",
        "mixed-3",
        "peptide-8",
        "peptide-12",
        "peptide-83",
        "long-peptide-3",
        "long-peptide-5",
        "long-peptide-6",
        "long-peptide-7",
        "long-peptide-10",
        "cyclic-peptide-1",
        "cyclic-peptide-3",
        "cyclic-peptide-4",
        "cyclic-peptide-6",
        "cyclic-peptide-8",
        "cyclic-peptide-10",
        "cyclic-peptide-19",
        "unrelated-2",
        "unrelated-8",
        "cut-mixed-136",
        "cut-mixed-166",
        "mirrored-residue-2",
        "mirrored-residue-3",
        "mirrored-residue-5",
        "mirrored-residue-8",
        "mirrored-residue-9",
        "mirrored-residue-10",
        "mirrored-residue-17",
        "mirrored-residue-19",
        "mirrored-residue-20",
        "mirrored-residue-34",
        "mirrored-residue-57",
    }
)
"""Candidates left out, as family and place in the family's stream: their calls did not end
within four virtual seconds on the clock ``clock-60ae29dd4850`` when the benchmark was made
(2026-11-18), and the benchmark's calls must all run to their end."""


def band(heavy: int) -> int:
    return sum(heavy > top for top in BANDS)


def candidates() -> Iterator[tuple[str, int, Call]]:
    """Every call the quotas take, with its name and its pair's heavy atoms, family by
    family."""
    for index, (family, make) in enumerate(FAMILIES.items()):
        left = list(QUOTAS[family])
        stream = make(Draw(SEED + index))
        seen: set[Call] = set()
        place = 0
        while any(left):
            place += 1
            if place > 50_000:
                raise RuntimeError(f"{family}: the quotas {left} cannot be met")
            call = next(stream)
            source, target, _ = call
            sizes = atoms(source), atoms(target)
            if not all(sizes) or call in seen or same(source, target):
                continue
            where = band(sum(sizes))
            if not left[where]:
                continue
            seen.add(call)
            name = f"{family}-{place}"
            if name in SKIP:
                continue
            left[where] -= 1
            yield name, sum(sizes), call


def build() -> str:
    """The text of the benchmark file."""
    RDLogger.DisableLog("rdApp.*")
    try:
        calls = sorted({(heavy, *call) for _, heavy, call in candidates()})
    finally:
        RDLogger.EnableLog("rdApp.*")
    lines = [
        "# Calibration benchmark of atommapper.calibrate: calls of map_atoms on pairs of "
        "molecules.",
        "# Synthetic: every molecule was put together from building blocks by",
        f"# benchmarks/make_calibration_pairs.py (seed {SEED}); none comes from a database.",
        f"# {len(calls)} calls, about 50 in each of six bands of the pair's heavy atoms",
        "# (up to 50, 70, 90, 120, 200 and above), smallest pairs first.",
        "# Columns: source SMILES, target SMILES, allow_stereo_inversion (0 or 1).",
    ]
    lines += [f"{source}\t{target}\t{int(allow)}" for _, source, target, allow in calls]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = build()
    if "--check" in sys.argv[1:]:
        if OUTPUT.read_text() != text:
            print(f"{OUTPUT} is not what this script writes", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}: {text.count(chr(10)) - 6} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
