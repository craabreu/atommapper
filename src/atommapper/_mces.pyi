"""Type stub for the compiled fragment search engine (rust/src/lib.rs)."""

__version__: str

def fragments(
    labels1: list[int],
    bonds1: list[tuple[int, int]],
    labels2: list[int],
    bonds2: list[tuple[int, int]],
    *,
    max_steps: int,
    max_fragments: int,
    min_bonds: int,
    max_pairs: int = 20_000,
    best_only: bool = False,
    rings: tuple[list[bool], list[bool], list[bool], list[bool]] | None = None,
    time_limit_s: float | None = None,
    known_score: int = 0,
    work_limit: int | None = None,
    work_costs: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[
    list[tuple[list[tuple[int, int]], list[int]]],
    int,
    str | None,
    bool,
    int,
    int,
    int,
    tuple[int, int, int, int],
]: ...
