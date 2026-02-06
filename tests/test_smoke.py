from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from las_utils.thin import ThinConfig, thin_las, voxel_keys


def _write_synthetic_las(path: Path, n: int = 50_000) -> None:
    # Lots of duplicates by construction: x/y are on a coarse grid, z random.
    rng = np.random.default_rng(123)

    x = rng.integers(0, 200, size=n).astype(np.float64) * 0.2  # 0.2m grid
    y = rng.integers(0, 200, size=n).astype(np.float64) * 0.2
    z = rng.normal(10.0, 1.0, size=n).astype(np.float64)

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.write(path)


def test_thin_smoke_xyz(tmp_path: Path) -> None:
    in_path = tmp_path / "in.las"
    out_path = tmp_path / "out.las"

    _write_synthetic_las(in_path, n=80_000)

    cfg = ThinConfig(
        voxel=0.5,
        mode="xyz",
        chunk_points=10_000,
        bloom_mb=8,
        bloom_hashes=4,
        origin_mode="header",
        target_bytes=None,
        quiet=True,
    )
    thin_las(in_path, out_path, cfg)

    src = laspy.read(in_path)
    dst = laspy.read(out_path)

    assert len(dst.points) < len(src.points)

    origin = (float(src.header.mins[0]), float(src.header.mins[1]), float(src.header.mins[2]))
    keys = voxel_keys(np.asarray(dst.x), np.asarray(dst.y), np.asarray(dst.z), voxel=0.5, origin=origin, mode="xyz")
    assert np.unique(keys).size == keys.size


def test_thin_smoke_xy(tmp_path: Path) -> None:
    in_path = tmp_path / "in.las"
    out_path = tmp_path / "out.las"

    _write_synthetic_las(in_path, n=60_000)

    cfg = ThinConfig(
        voxel=0.5,
        mode="xy",
        chunk_points=12_000,
        bloom_mb=8,
        bloom_hashes=4,
        origin_mode="header",
        target_bytes=None,
        quiet=True,
    )
    thin_las(in_path, out_path, cfg)

    dst = laspy.read(out_path)
    assert len(dst.points) > 0
