from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import laspy
import numpy as np

Mode = Literal["xyz", "xy"]
OriginMode = Literal["header", "min"]


def _u64(x: np.ndarray) -> np.ndarray:
    return x.astype(np.uint64, copy=False)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """
    Vectorized splitmix64 for uint64 arrays.
    """
    x = _u64(x)
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return (z ^ (z >> np.uint64(31))) & np.uint64(0xFFFFFFFFFFFFFFFF)


def voxel_keys(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    voxel: float,
    origin: tuple[float, float, float],
    mode: Mode,
) -> np.ndarray:
    """
    Compute stable 64-bit keys for voxel coordinates.

    Notes:
    - For negative indices, we reinterpret int64 bits as uint64 (stable).
    - We hash indices (not floats) to be invariant to scale/offset encoding details.
    """
    if voxel <= 0:
        raise ValueError("voxel must be > 0")

    ox, oy, oz = origin
    inv = 1.0 / float(voxel)

    ix = np.floor((x - ox) * inv).astype(np.int64, copy=False)
    iy = np.floor((y - oy) * inv).astype(np.int64, copy=False)
    if mode == "xyz":
        iz = np.floor((z - oz) * inv).astype(np.int64, copy=False)
    else:
        iz = np.zeros_like(ix)

    # Reinterpret int64 -> uint64 (two's complement) for stable hashing
    ix_u = ix.view(np.uint64)
    iy_u = iy.view(np.uint64)
    iz_u = iz.view(np.uint64)

    # Mix coordinates into a single key
    s1 = np.uint64(0xD6E8FEB86659FD93)
    s2 = np.uint64(0xA5A3564E27F90B25)
    h = splitmix64(ix_u ^ splitmix64(iy_u + s1) ^ splitmix64(iz_u + s2))
    return h


@dataclass(frozen=True)
class ThinConfig:
    voxel: float | None
    mode: Mode = "xyz"
    chunk_points: int = 1_000_000

    bloom_mb: int = 128
    bloom_hashes: int = 4

    origin_mode: OriginMode = "header"

    # auto-tune
    target_bytes: int | None = None
    sample_points: int = 750_000
    sample_windows: int = 32
    tune_iters: int = 4

    quiet: bool = False


class BloomFilter:
    """
    Bloom filter using a compact uint8 bit-array and double hashing.

    We force number of bits to be a power of two so modulo becomes & mask.
    """

    def __init__(self, *, mb: int, k: int, seed: int = 0xC0FFEE):
        if mb <= 0:
            raise ValueError("mb must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")

        bytes_budget = int(mb) * 1024 * 1024
        bits_budget = bytes_budget * 8
        if bits_budget < 1024:
            bits_budget = 1024

        # Largest power-of-two <= budget
        m_bits = 1 << (int(bits_budget).bit_length() - 1)
        self._m_bits = np.uint64(m_bits)
        self._mask = np.uint64(m_bits - 1)
        self._k = int(k)

        self._bits = np.zeros(m_bits // 8, dtype=np.uint8)

        # seeds for double hashing
        self._seed1 = np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
        self._seed2 = np.uint64(seed) ^ np.uint64(0xBF58476D1CE4E5B9)

    @property
    def size_bytes(self) -> int:
        return int(self._bits.nbytes)

    @property
    def size_bits(self) -> int:
        return int(self._m_bits)

    def _hashes(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys = _u64(keys)
        h1 = splitmix64(keys + self._seed1)
        h2 = splitmix64(keys + self._seed2) | np.uint64(1)
        return h1, h2

    def add_if_new(self, keys: np.ndarray) -> np.ndarray:
        """
        For each key, returns True if it appears NEW (i.e. not all Bloom bits set yet),
        and sets the Bloom bits for those new keys.

        False-positives are possible (treated as already seen -> more thinning).
        """
        keys = _u64(keys)
        if keys.size == 0:
            return np.zeros((0,), dtype=bool)

        h1, h2 = self._hashes(keys)
        present = np.ones(keys.shape[0], dtype=bool)

        for i in range(self._k):
            idx = (h1 + np.uint64(i) * h2) & self._mask
            byte_idx = (idx >> np.uint64(3)).astype(np.intp, copy=False)
            bit = np.uint8(1) << (idx & np.uint64(7)).astype(np.uint8, copy=False)
            present &= (self._bits[byte_idx] & bit) != 0

        is_new = ~present
        if not np.any(is_new):
            return is_new

        new_keys = keys[is_new]
        h1n, h2n = self._hashes(new_keys)
        for i in range(self._k):
            idx = (h1n + np.uint64(i) * h2n) & self._mask
            byte_idx = (idx >> np.uint64(3)).astype(np.intp, copy=False)
            bit = np.uint8(1) << (idx & np.uint64(7)).astype(np.uint8, copy=False)
            np.bitwise_or.at(self._bits, byte_idx, bit)

        return is_new


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n/1024:.1f} KiB"
    if n < 1024**3:
        return f"{n/1024**2:.1f} MiB"
    return f"{n/1024**3:.2f} GiB"


def _log(cfg: ThinConfig, msg: str) -> None:
    if not cfg.quiet:
        print(msg)


def _compute_origin_from_header(header: laspy.LasHeader) -> tuple[float, float, float]:
    mins = header.mins
    return (float(mins[0]), float(mins[1]), float(mins[2]))


def _compute_origin_streaming(in_path: Path, cfg: ThinConfig) -> tuple[float, float, float]:
    # Slow path: scan mins (still streaming)
    minx = np.inf
    miny = np.inf
    minz = np.inf
    with laspy.open(in_path) as r:
        for chunk in r.chunk_iterator(cfg.chunk_points):
            x = chunk.x
            y = chunk.y
            z = chunk.z
            minx = min(minx, float(np.min(x)))
            miny = min(miny, float(np.min(y)))
            minz = min(minz, float(np.min(z)))
    return (float(minx), float(miny), float(minz))


def _sample_points_distributed(
    reader: laspy.LasReader,
    *,
    sample_points: int,
    windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a sample spread across the file by seeking to evenly spaced windows.

    Works well for uncompressed LAS where seeking is O(1).
    """
    n_total = int(reader.header.point_count)
    if n_total <= 0:
        return (np.empty((0,), dtype=np.float64),) * 3

    windows = max(1, int(windows))
    per_win = max(1, int(sample_points // windows))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []

    # Evenly spaced start positions
    for wi in range(windows):
        start = int((wi + 0.5) / windows * n_total)
        if start >= n_total:
            start = n_total - 1
        if start < 0:
            start = 0

        reader.seek(start)
        pts = reader.read_points(per_win)
        if len(pts) == 0:
            continue

        xs.append(np.asarray(pts.x))
        ys.append(np.asarray(pts.y))
        zs.append(np.asarray(pts.z))

        got = sum(a.size for a in xs)
        if got >= sample_points:
            break

    if not xs:
        return (np.empty((0,), dtype=np.float64),) * 3

    x = np.concatenate(xs, axis=0)[:sample_points]
    y = np.concatenate(ys, axis=0)[:sample_points]
    z = np.concatenate(zs, axis=0)[:sample_points]
    return x, y, z


def autotune_voxel(in_path: Path, cfg: ThinConfig, *, origin: tuple[float, float, float]) -> float:
    """
    Choose voxel size so that expected number of kept points roughly matches target_bytes.

    Uses a distributed sample and assumes keep_ratio scales approximately with 1/voxel^3.
    """
    assert cfg.target_bytes is not None

    with laspy.open(in_path) as r:
        header = r.header
        n_total = int(header.point_count)
        if n_total <= 0:
            return 1.0

        point_size = int(header.point_format.size)
        target_points = max(1.0, float(cfg.target_bytes) / float(point_size))
        target_ratio = min(1.0, max(1e-12, target_points / float(n_total)))

        x, y, z = _sample_points_distributed(
            r,
            sample_points=cfg.sample_points,
            windows=cfg.sample_windows,
        )

    if x.size == 0:
        return 1.0

    voxel = float(cfg.voxel) if cfg.voxel is not None else 0.5
    voxel = max(1e-6, voxel)

    _log(cfg, f"[tune] target_bytes={cfg.target_bytes} ({_fmt_bytes(cfg.target_bytes)}), target_ratio≈{target_ratio:.6g}")
    _log(cfg, f"[tune] sample_points={x.size}, windows={cfg.sample_windows}, start_voxel={voxel:g}")

    for it in range(max(1, int(cfg.tune_iters))):
        keys = voxel_keys(x, y, z, voxel=voxel, origin=origin, mode=cfg.mode)
        unique_vox = int(np.unique(keys).size)
        keep_ratio = max(1e-12, float(unique_vox) / float(keys.size))

        # Update step
        scale = (keep_ratio / target_ratio) ** (1.0 / 3.0)
        voxel_new = voxel * scale

        # clamp
        voxel_new = float(np.clip(voxel_new, 0.01, 200.0))
        _log(cfg, f"[tune] it={it} voxel={voxel:g} keep_ratio≈{keep_ratio:.6g} -> voxel={voxel_new:g}")
        voxel = voxel_new

    return float(voxel)


def thin_las(in_path: Path, out_path: Path, cfg: ThinConfig) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    with laspy.open(in_path) as r:
        header_in = r.header
        n_total = int(header_in.point_count)
        point_size = int(header_in.point_format.size)

    _log(cfg, f"[in] {in_path} points={n_total} point_size={point_size}B")

    # Origin
    if cfg.origin_mode == "header":
        with laspy.open(in_path) as r:
            origin = _compute_origin_from_header(r.header)
    else:
        _log(cfg, "[origin] scanning mins (slow path)")
        origin = _compute_origin_streaming(in_path, cfg)
    _log(cfg, f"[origin] {origin}")

    # voxel selection
    voxel = cfg.voxel
    if voxel is None:
        if cfg.target_bytes is None:
            voxel = 1.0
            _log(cfg, f"[voxel] not set; using default voxel={voxel:g}")
        else:
            voxel = autotune_voxel(in_path, cfg, origin=origin)
            _log(cfg, f"[voxel] tuned voxel={voxel:g}")
    else:
        _log(cfg, f"[voxel] fixed voxel={float(voxel):g}")

    bloom = BloomFilter(mb=cfg.bloom_mb, k=cfg.bloom_hashes)
    _log(cfg, f"[bloom] size≈{_fmt_bytes(bloom.size_bytes)} bits={bloom.size_bits} k={cfg.bloom_hashes}")

    kept = 0
    seen = 0

    with laspy.open(in_path) as r:
        out_header = r.header.copy()

        # Writer will update counts/bounds as we write.
        out_header.point_count = 0
        try:
            out_header.number_of_points_by_return = np.zeros_like(out_header.number_of_points_by_return)
        except Exception:
            pass

        with laspy.open(out_path, mode="w", header=out_header) as w:
            for chunk in r.chunk_iterator(cfg.chunk_points):
                n = len(chunk)
                if n == 0:
                    continue
                seen += n

                keys = voxel_keys(
                    np.asarray(chunk.x),
                    np.asarray(chunk.y),
                    np.asarray(chunk.z),
                    voxel=float(voxel),
                    origin=origin,
                    mode=cfg.mode,
                )

                # Unique within chunk first (cheaper Bloom checks)
                uniq_keys, uniq_idx = np.unique(keys, return_index=True)
                is_new = bloom.add_if_new(uniq_keys)
                if np.any(is_new):
                    keep_idx = uniq_idx[is_new]
                    pts = chunk[keep_idx]
                    w.write_points(pts)
                    kept += len(pts)

                if not cfg.quiet and (seen % (cfg.chunk_points * 10) < cfg.chunk_points):
                    _log(cfg, f"[progress] seen={seen}/{n_total} kept={kept}")

    try:
        out_size = out_path.stat().st_size
    except OSError:
        out_size = -1

    _log(
        cfg,
        f"[done] kept={kept} of {n_total} ({(kept/max(1,n_total))*100:.2f}%) out_size={_fmt_bytes(out_size) if out_size>=0 else 'n/a'}",
    )
