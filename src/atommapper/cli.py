"""The atommapper command."""

import argparse
import json
import sys
from collections.abc import Sequence

from .drawing import save_image
from .errors import InvalidInput
from .mapper import map_atoms
from .settings import Settings


def _panel_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--size must look like 400x300, got {text!r}") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"--size must be positive, got {text!r}")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atommapper",
        description="Map the atoms of two molecules under LOMAP's rules; print a numbered "
        "reaction SMILES or the full result.",
    )
    parser.add_argument("source", help="source molecule SMILES")
    parser.add_argument("target", help="target molecule SMILES")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument(
        "--draw",
        metavar="FILE",
        help="also draw the mapped pair to FILE, as SVG or PNG by its suffix (.svg, .png)",
    )
    parser.add_argument(
        "--size",
        metavar="WxH",
        type=_panel_size,
        default="400x300",
        help="size of each panel of the drawing, in pixels (default 400x300)",
    )
    parser.add_argument(
        "--numbers",
        action="store_true",
        help="label the kept atoms of the drawing with their identifiers",
    )
    parser.add_argument(
        "--objective",
        choices=("change", "bonds"),
        default="change",
        help="change (the smallest change, the default) or bonds (LOMAP's, the most bonds kept)",
    )
    parser.add_argument(
        "--fixed-elements",
        action="store_true",
        help="a kept atom keeps its element (LOMAP's element_change off)",
    )
    parser.add_argument(
        "--allow-stereo-inversion",
        action="store_true",
        help="a kept stereocentre or double bond may be inverted, each inversion counting "
        "one change (otherwise none may be kept inverted)",
    )
    parser.add_argument(
        "--nodes", type=int, default=None, help="branch-and-bound nodes for the whole call"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Stage 1 search steps for the whole call"
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=None,
        help="ceiling on compatible bond pairs Stage 1 searches (default 20000)",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="wall-clock limit for the whole call, seconds"
    )
    parser.add_argument(
        "--accept-partial", action="store_true", help="exit 0 even if a search budget ran out"
    )
    parser.epilog = (
        "exit status: 0 success, 1 invalid or unsupported input, 2 usage error, "
        "3 a search budget ran out (unless --accept-partial)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.draw is not None and not args.draw.lower().endswith((".svg", ".png")):
        parser.error(f"--draw takes a .svg or .png file, got {args.draw!r}")
    defaults = Settings()
    settings = Settings(
        node_budget=defaults.node_budget if args.nodes is None else args.nodes,
        stage1_steps=defaults.stage1_steps if args.steps is None else args.steps,
        stage1_bond_pairs=defaults.stage1_bond_pairs if args.pairs is None else args.pairs,
        time_limit_s=defaults.time_limit_s if args.timeout is None else args.timeout,
    )
    try:
        result = map_atoms(
            args.source,
            args.target,
            element_change=not args.fixed_elements,
            objective=args.objective,
            allow_stereo_inversion=args.allow_stereo_inversion,
            settings=settings,
        )
    except InvalidInput as error:
        print(f"atommapper: {error}", file=sys.stderr)
        return 1
    if result.status == "unsupported_input":
        print(f"atommapper: unsupported input: {result.reason}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result.reaction_smiles)
    if args.draw is not None:
        width, height = args.size
        save_image(result, args.draw, width=width, height=height, numbers=args.numbers)
    if result.status == "timed_out" and not args.accept_partial:
        print(f"atommapper: {result.reason}", file=sys.stderr)
        return 3
    return 0
