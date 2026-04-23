import glob
import os
import re
import zlib

import numpy as np
from astropy.io import fits
from astropy.table import Table

from desiproc.implement_astra import register_tracer_mapping
from desiproc.paths import safe_tag, zone_tag

from .base import ReleaseConfig


TRACERS = ['HOD']
TRACER_ALIAS = {'hod': 'HOD'}
TRACER_IDS = {'HOD': 0}
TRACER_FULL_LABELS = {(0, True): b'HOD',
                      (0, False): b'HOD'}

REQUIRED_COLUMNS = ('ID', 'MASS', 'X_RSD', 'Y_RSD', 'Z_RSD')
_SNAPSHOT_RE = re.compile(r'(?P<cosmo>c\d+_ph\d+)[/\\](?P<seed>seed\d+)[/\\](?P<hod>hod\d+)(?:\.fits)?$',
                          flags=re.IGNORECASE)
_ZONE_KEY_RE = re.compile(r'^c(?P<c>\d+)_ph(?P<ph>\d+)_seed(?P<seed>\d+)_hod(?P<hod>\d+)$',
                          flags=re.IGNORECASE)


def _progress(message):
    """
    Emit a progress message when ASTRA_PROGRESS is enabled.
    """
    if os.environ.get('ASTRA_PROGRESS'):
        print(f'[progress] {message}', flush=True)


def _zone_sort_key(zone):
    """
    Return a stable sorting key for labels like ``c000_ph000_seed0_hod000``.
    """
    text = str(zone).strip().lower()
    match = _ZONE_KEY_RE.match(text)
    if match is None:
        return (text,)
    return (int(match.group('c')),
            int(match.group('ph')),
            int(match.group('seed')),
            int(match.group('hod')))


def _build_zone_label(cosmo, seed, hod):
    """
    Build canonical snapshot label used as pipeline zone token.
    """
    return f'{str(cosmo).lower()}_{str(seed).lower()}_{str(hod).lower()}'


def _zone_from_token(token):
    """
    Normalize a user token to canonical zone label form.
    """
    text = str(token).strip()
    if not text:
        raise RuntimeError('Empty zone token is not valid')
    norm = text.replace('\\', '/')
    if norm.lower().endswith('.fits'):
        norm = norm[:-5]
    match = _SNAPSHOT_RE.search(norm)
    if match is not None:
        return _build_zone_label(match.group('cosmo'),
                                 match.group('seed'),
                                 match.group('hod'))
    candidate = norm.replace('/', '_').strip('_').lower()
    if _ZONE_KEY_RE.match(candidate):
        return candidate
    raise RuntimeError(f'Cannot parse snapshot token "{token}". '
                       'Use cXXX_phYYY_seedZ_hodWWW or cXXX_phYYY/seedZ/hodWWW.fits')


def discover_snapshot_map(base_dir):
    """
    Discover available snapshots under ``base_dir``.

    Returns:
        dict[str, str]: Mapping ``zone_label -> absolute snapshot path``.
    """
    root = os.path.abspath(os.path.expanduser(str(base_dir)))
    pattern = os.path.join(root, 'c*_ph*', 'seed*', 'hod*.fits')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(f'No snapshot files found under {root} matching {pattern}')

    mapping = {}
    for path in paths:
        rel = os.path.relpath(path, root).replace(os.sep, '/')
        match = _SNAPSHOT_RE.search(rel)
        if match is None:
            continue
        label = _build_zone_label(match.group('cosmo'),
                                  match.group('seed'),
                                  match.group('hod'))
        if label in mapping:
            raise RuntimeError(f'Duplicate snapshot label {label} for paths '
                               f'{mapping[label]} and {path}')
        mapping[label] = os.path.abspath(path)

    if not mapping:
        raise RuntimeError(f'No valid cXXX_phYYY/seedZ/hodWWW.fits snapshots found in {root}')
    return mapping


def _select_zones(snapshot_map, requested_tokens):
    """
    Resolve requested zone tokens against ``snapshot_map``.
    """
    available = set(snapshot_map.keys())
    if requested_tokens is None:
        return sorted(available, key=_zone_sort_key)

    selected = []
    seen = set()
    for token in requested_tokens:
        label = _zone_from_token(token)
        if label not in available:
            raise RuntimeError(f'Snapshot zone "{token}" resolved to "{label}" but it was not found. '
                               f'Available count={len(available)}')
        if label in seen:
            continue
        seen.add(label)
        selected.append(label)
    return sorted(selected, key=_zone_sort_key)


def _existing_raw_path(output_raw, zone_label, out_tag):
    """
    Return existing raw path for ``zone_label`` if present.
    """
    tsuf = safe_tag(out_tag)
    candidates = [os.path.join(output_raw, f'zone_{zone_label}{tsuf}.fits'),
                  os.path.join(output_raw, f'zone_{zone_label}{tsuf}.fits.gz')]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _raw_output_path(output_raw, zone_label, out_tag):
    """
    Raw output path for simulation snapshots.
    """
    tsuf = safe_tag(out_tag)
    return os.path.join(output_raw, f'zone_{zone_label}{tsuf}.fits')


def _chunk_rows():
    """
    Chunk size used when writing large snapshot-derived raw FITS tables.
    """
    try:
        return max(1, int(os.environ.get('ASTRA_SIM_CHUNK_ROWS', '500000')))
    except Exception:
        return 500000


def _random_pool_factor():
    """
    Multiplicative factor for synthetic random-pool size relative to Ndata.
    """
    try:
        return max(1, int(os.environ.get('ASTRA_SIM_NRAND_FACTOR', '20')))
    except Exception:
        return 20


def _resolve_lbox(x_values, y_values, z_values):
    """
    Resolve Lbox used to generate uniform random points in [0, Lbox].

    Priority:
    1) ``ASTRA_SIM_LBOX`` environment variable.
    2) Maximum span among X/Y/Z data coordinates.
    """
    env_value = os.environ.get('ASTRA_SIM_LBOX')
    if env_value is not None and str(env_value).strip() != '':
        try:
            val = float(env_value)
            if np.isfinite(val) and val > 0:
                return val
        except Exception:
            pass
        raise RuntimeError(f'Invalid ASTRA_SIM_LBOX value: {env_value}')

    spans = []
    for arr in (x_values, y_values, z_values):
        values = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            raise RuntimeError('Snapshot coordinate column contains no finite values')
        lo = float(np.min(values[finite]))
        hi = float(np.max(values[finite]))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            continue
        spans.append(span)
    if not spans:
        raise RuntimeError('Cannot infer a valid Lbox from snapshot coordinates')
    return float(max(spans))


def _random_pool_path(output_raw, zone_label, out_tag):
    """
    Random-pool path for one snapshot zone.
    """
    tsuf = safe_tag(out_tag)
    pool_dir = os.path.join(output_raw, 'random_pools')
    os.makedirs(pool_dir, exist_ok=True)
    return os.path.join(pool_dir, f'zone_{zone_label}{tsuf}_random_pool.fits')


def _random_seed_for_zone(zone_label):
    """
    Deterministic seed derived from the zone label.
    """
    text = str(zone_label).encode('utf-8', errors='ignore')
    return int(zlib.crc32(text) & 0xFFFFFFFF)


def _ensure_random_pool(pool_path, n_pool, lbox, zone_label):
    """
    Create or reuse a synthetic random pool with points in [0, Lbox]^3.
    """
    if os.path.exists(pool_path):
        try:
            with fits.open(pool_path, memmap=True) as hdul:
                nrows = len(hdul[1].data)
                hdr = hdul[1].header
                old_lbox = float(hdr.get('LBOX', np.nan))
                if int(nrows) == int(n_pool) and np.isfinite(old_lbox) and abs(old_lbox - float(lbox)) < 1e-6:
                    _progress(f'{zone_label}: reusing random pool {pool_path} (rows={n_pool}, Lbox={lbox:.6g})')
                    return
        except Exception:
            pass

    cols = [('RANDID', 'K'),
            ('XCART', 'E'),
            ('YCART', 'E'),
            ('ZCART', 'E')]
    coldefs = fits.ColDefs([fits.Column(name=name, format=fmt) for name, fmt in cols])
    hdu = fits.BinTableHDU.from_columns(coldefs, nrows=int(n_pool))
    hdu.header['LBOX'] = float(lbox)
    hdu.header['NPOOL'] = int(n_pool)
    hdu.header['ZONE'] = zone_tag(zone_label)

    tmp_path = f'{pool_path}.tmp'
    hdu.writeto(tmp_path, overwrite=True)

    seed = _random_seed_for_zone(zone_label)
    chunk_rows = _chunk_rows()
    rng = np.random.default_rng(seed)
    with fits.open(tmp_path, mode='update', memmap=True) as hdul:
        data = hdul[1].data
        for start in range(0, int(n_pool), chunk_rows):
            end = min(start + chunk_rows, int(n_pool))
            size = end - start
            rows = slice(start, end)
            data['RANDID'][rows] = np.arange(start, end, dtype=np.int64)
            data['XCART'][rows] = rng.uniform(0.0, float(lbox), size=size).astype(np.float32, copy=False)
            data['YCART'][rows] = rng.uniform(0.0, float(lbox), size=size).astype(np.float32, copy=False)
            data['ZCART'][rows] = rng.uniform(0.0, float(lbox), size=size).astype(np.float32, copy=False)
        hdul.flush()

    os.replace(tmp_path, pool_path)
    _progress(f'{zone_label}: random pool created at {pool_path} (rows={n_pool}, Lbox={lbox:.6g})')


def _sample_random_iterations_from_pool(out_data, pool_path, n_data, n_random, zone_label):
    """
    Fill random rows in output FITS by sampling from a persisted random pool.
    """
    chunk_rows = _chunk_rows()
    rand_offset = np.int64(1 << 62)
    with fits.open(pool_path, memmap=True) as pool_hdul:
        pool = pool_hdul[1].data
        n_pool = len(pool)
        if n_pool <= 0:
            raise RuntimeError(f'Random pool is empty for zone {zone_label}: {pool_path}')
        replace = bool(n_pool < n_data)
        for j in range(int(n_random)):
            rng = np.random.default_rng(j)
            sel = rng.choice(n_pool, int(n_data), replace=replace)
            iter_base = int(n_data) * (j + 1)
            for start in range(0, int(n_data), chunk_rows):
                end = min(start + chunk_rows, int(n_data))
                rows = slice(iter_base + start, iter_base + end)
                idx = sel[start:end]
                out_data['TARGETID'][rows] = rand_offset + np.asarray(pool['RANDID'][idx], dtype=np.int64)
                out_data['RANDITER'][rows] = np.int32(j)
                out_data['XCART'][rows] = np.asarray(pool['XCART'][idx], dtype=np.float32)
                out_data['YCART'][rows] = np.asarray(pool['YCART'][idx], dtype=np.float32)
                out_data['ZCART'][rows] = np.asarray(pool['ZCART'][idx], dtype=np.float32)
                out_data['SED_MASS'][rows] = np.nan
                out_data['MASS'][rows] = np.nan
                if 'IS_CENT' in out_data.columns.names:
                    out_data['IS_CENT'][rows] = False
            if (j + 1) % 5 == 0 or (j + 1) == int(n_random):
                _progress(f'{zone_label}: sampled random iteration {j+1}/{n_random} from pool')


def _write_snapshot_raw(snapshot_path, output_path, n_random, zone_label, release_tag, out_tag=None):
    """
    Build raw FITS for one snapshot using X/Y/Z in redshift space directly.
    """
    if int(n_random) < 1:
        raise RuntimeError('For YUAN23 snapshots, --n-random must be >= 1')

    snap = Table.read(snapshot_path, memmap=True)
    missing = [name for name in REQUIRED_COLUMNS if name not in snap.colnames]
    if missing:
        raise KeyError(f'Snapshot {snapshot_path} missing required columns: {missing}')

    n_data = len(snap)
    if n_data == 0:
        raise RuntimeError(f'Snapshot {snapshot_path} is empty')

    ids = np.asarray(snap['ID'], dtype=np.int64)
    mass = np.asarray(snap['MASS'], dtype=np.float32)
    x_rsd = np.asarray(snap['X_RSD'], dtype=np.float32)
    y_rsd = np.asarray(snap['Y_RSD'], dtype=np.float32)
    z_rsd = np.asarray(snap['Z_RSD'], dtype=np.float32)
    has_is_cent = 'IS_CENT' in snap.colnames
    if has_is_cent:
        is_cent = np.asarray(snap['IS_CENT']).astype(bool)
    else:
        is_cent = None

    lbox = _resolve_lbox(x_rsd, y_rsd, z_rsd)
    n_pool = int(_random_pool_factor()) * int(n_data)
    if n_pool < int(n_data):
        n_pool = int(n_data)
    pool_path = _random_pool_path(os.path.dirname(output_path), zone_label, out_tag=out_tag)
    _ensure_random_pool(pool_path, n_pool=n_pool, lbox=lbox, zone_label=zone_label)

    n_random = int(n_random)
    total_rows = int(n_data) * int(n_random + 1)
    if total_rows <= 0:
        raise RuntimeError(f'Invalid total row count for {snapshot_path}: {total_rows}')

    cols = [('TARGETID', 'K'),
            ('TRACER_ID', 'B'),
            ('TRACERTYPE', '24A'),
            ('RANDITER', 'J'),
            ('XCART', 'E'),
            ('YCART', 'E'),
            ('ZCART', 'E'),
            ('SED_MASS', 'E'),
            ('MASS', 'E')]
    if has_is_cent:
        cols.append(('IS_CENT', 'L'))

    coldefs = fits.ColDefs([fits.Column(name=name, format=fmt) for name, fmt in cols])
    hdu = fits.BinTableHDU.from_columns(coldefs, nrows=total_rows)
    hdu.header['ZONE'] = zone_tag(zone_label)
    hdu.header['RELEASE'] = str(release_tag) if release_tag is not None else 'YUAN23'
    hdu.header['SNAPSHOT'] = os.path.basename(snapshot_path)
    hdu.header['LBOX'] = float(lbox)
    hdu.header['NPOOL'] = int(n_pool)

    tmp_path = f'{output_path}.tmp'
    hdu.writeto(tmp_path, overwrite=True)

    tracer_id = np.uint8(TRACER_IDS['HOD'])
    tracer_bytes = b'HOD'
    chunk_rows = _chunk_rows()

    with fits.open(tmp_path, mode='update', memmap=True) as hdul:
        data = hdul[1].data

        for start in range(0, n_data, chunk_rows):
            end = min(start + chunk_rows, n_data)
            rows = slice(start, end)
            data['TARGETID'][rows] = ids[start:end]
            data['TRACER_ID'][rows] = tracer_id
            data['TRACERTYPE'][rows] = tracer_bytes
            data['RANDITER'][rows] = -1
            data['XCART'][rows] = x_rsd[start:end]
            data['YCART'][rows] = y_rsd[start:end]
            data['ZCART'][rows] = z_rsd[start:end]
            data['SED_MASS'][rows] = mass[start:end]
            data['MASS'][rows] = mass[start:end]
            if has_is_cent:
                data['IS_CENT'][rows] = is_cent[start:end]

        _progress(f'{zone_label}: wrote data rows={n_data}')

        for j in range(n_random):
            iter_base = n_data * (j + 1)
            for start in range(0, n_data, chunk_rows):
                end = min(start + chunk_rows, n_data)
                rows = slice(iter_base + start, iter_base + end)
                data['TRACER_ID'][rows] = tracer_id
                data['TRACERTYPE'][rows] = tracer_bytes

        _sample_random_iterations_from_pool(data, pool_path, n_data=n_data,
                                            n_random=n_random, zone_label=zone_label)

        hdul.flush()

    os.replace(tmp_path, output_path)


def build_raw_zone(zone_label, snapshot_path, output_raw, n_random, out_tag, release_tag):
    """
    Build and persist one simulation-zone raw table.
    """
    os.makedirs(output_raw, exist_ok=True)

    existing = _existing_raw_path(output_raw, zone_label, out_tag)
    if existing is not None:
        try:
            tbl = Table.read(existing, memmap=True)
            print(f'[yuan23] reuse existing raw {existing}', flush=True)
            return tbl
        except Exception as exc:
            print(f'[yuan23] warning: cannot read existing raw {existing} ({exc}); rebuilding', flush=True)

    output_path = _raw_output_path(output_raw, zone_label, out_tag)
    _progress(f'{zone_label}: building raw table from {snapshot_path}')
    _write_snapshot_raw(snapshot_path, output_path, n_random=n_random,
                        zone_label=zone_label, release_tag=release_tag,
                        out_tag=out_tag)
    _progress(f'{zone_label}: raw table written to {output_path}')
    return Table.read(output_path, memmap=True)


def create_config(args):
    """
    Create release configuration for YUAN23 HOD snapshots.
    """
    base_dir = args.base_dir
    if not base_dir:
        raise RuntimeError('--base-dir is required for YUAN23/SIM release')

    snapshot_map = discover_snapshot_map(base_dir)
    zones = _select_zones(snapshot_map, args.zones)
    if not zones:
        raise RuntimeError(f'No snapshots selected under {base_dir}')

    print(f'[yuan23] discovered snapshots={len(snapshot_map)} selected={len(zones)}', flush=True)
    register_tracer_mapping(TRACER_IDS, TRACER_FULL_LABELS)

    def _build(zone, real_tables, random_tables, sel_tracers, parsed_args, release_tag):
        label = str(zone).strip().lower()
        snapshot_path = snapshot_map.get(label)
        if snapshot_path is None:
            raise RuntimeError(f'Snapshot zone "{label}" is not available in discovery map')
        return build_raw_zone(label, snapshot_path, parsed_args.raw_out,
                              parsed_args.n_random, out_tag=parsed_args.out_tag,
                              release_tag=release_tag)

    return ReleaseConfig(name='YUAN23', release_tag='YUAN23',
                         tracers=TRACERS, tracer_alias=TRACER_ALIAS,
                         real_suffix=None, random_suffix=None,
                         n_random_files=0,
                         real_columns=['ID', 'MASS', 'X_RSD', 'Y_RSD', 'Z_RSD'],
                         random_columns=[],
                         use_dr2_preload=False,
                         preload_kwargs={},
                         zones=zones,
                         build_raw=_build,
                         combine_outputs=False,
                         skip_preload=True)
