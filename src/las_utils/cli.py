from __future__ import annotations

import argparse
import os
from pathlib import Path

from .thin import ThinConfig, thin_las


def _positive_int(s: str) -> int:
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def _positive_float(s: str) -> float:
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="las-thin",
        description="Streamingly thin a large LAS by voxel grid (keep ≤1 point per voxel).",
    )

    p.add_argument("in_path", type=Path, help="Input .las path")
    p.add_argument("out_path", type=Path, help="Output .las path")

    g = p.add_argument_group("Thinning")
    g.add_argument(
        "--mode",
        choices=("xyz", "xy"),
        default="xyz",
        help="Voxel mode: xyz (3D) or xy (2D). Default: xyz",
    )
    g.add_argument(
        "--voxel",
        type=_positive_float,
        default=None,
        help="Voxel size in coordinate units (meters typically). If omitted and --target-* is set, auto-tune voxel.",
    )

    g2 = p.add_argument_group("Target size (auto-tune)")
    g2.add_argument(
        "--target-mb",
        type=_positive_float,
        default=None,
        help="Approx target output size in MiB (LAS). Enables auto-tune when --voxel not set.",
    )
    g2.add_argument(
        "--target-bytes",
        type=_positive_int,
        default=None,
        help="Approx target output size in bytes (LAS). Enables auto-tune when --voxel not set.",
    )
    g2.add_argument(
        "--sample-points",
        type=_positive_int,
        default=750_000,
        help="How many points to sample for auto-tune (distributed seeks). Default: 750000",
    )
    g2.add_argument(
        "--sample-windows",
        type=_positive_int,
        default=32,
        help="How many seek windows for auto-tune sampling (LAS only). Default: 32",
    )
    g2.add_argument(
        "--tune-iters",
        type=int,
        default=4,
        help="Auto-tune iterations. Default: 4",
    )

    g3 = p.add_argument_group("Performance / memory")
    g3.add_argument(
        "--chunk-points",
        type=_positive_int,
        default=1_000_000,
        help="Points per chunk. Default: 1000000",
    )
    g3.add_argument(
        "--bloom-mb",
        type=_positive_int,
        default=128,
        help="Bloom filter size in MiB (upper bound; actual uses nearest power-of-two bits <= this). Default: 128",
    )
    g3.add_argument(
        "--bloom-hashes",
        type=_positive_int,
        default=4,
        help="Number of Bloom hash functions. Default: 4",
    )

    g4 = p.add_argument_group("Coordinate origin")
    g4.add_argument(
        "--origin",
        choices=("header", "min"),
        default="header",
        help="Voxel origin: header (use header.mins) or min (compute mins on the fly; slower). Default: header",
    )

    p.add_argument(
        "--quiet",
        action="store_true",
        help="Less output.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    in_path = args.in_path
    out_path = args.out_path

    if in_path.resolve() == out_path.resolve():
        raise SystemExit("in_path and out_path must differ")
    if not in_path.exists():
        raise SystemExit(f"input does not exist: {in_path}")
    if out_path.exists():
        raise SystemExit(f"output already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_bytes = None
    if args.target_bytes is not None and args.target_mb is not None:
        raise SystemExit("use only one of --target-bytes or --target-mb")
    if args.target_bytes is not None:
        target_bytes = int(args.target_bytes)
    elif args.target_mb is not None:
        target_bytes = int(float(args.target_mb) * 1024 * 1024)

    cfg = ThinConfig(
        voxel=args.voxel,
        mode=args.mode,
        chunk_points=args.chunk_points,
        bloom_mb=args.bloom_mb,
        bloom_hashes=args.bloom_hashes,
        origin_mode=args.origin,
        target_bytes=target_bytes,
        sample_points=args.sample_points,
        sample_windows=args.sample_windows,
        tune_iters=args.tune_iters,
        quiet=args.quiet,
    )

    # Avoid Windows long-path surprises in logs
    os.environ.setdefault("PYTHONUTF8", "1")

    thin_las(in_path, out_path, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
