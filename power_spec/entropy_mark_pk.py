import argparse, json, os, time
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack
from pypower import CatalogFFTPower

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams.update({'grid.linewidth': 0.15,
                     'text.usetex': True})

PROB_COLS_DEFAULT = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
MARKED_COLOR = 'gold'
UNMARKED_COLOR = 'white'
NGC_RA_MIN_DEG = 90.0
NGC_RA_MAX_DEG = 300.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/PIP/')
    parser.add_argument('--tracer', type=str, default='BGS_ANY')
    parser.add_argument('--astra-prob-file', type=str,
                        default='/pscratch/sd/v/vtorresg/cosmic-web/dr2/probabilities/bgs/ngc/zone_NGC_BGS_probability_iterdata.fits.gz')
    parser.add_argument('--prob-cols', nargs=4, default=list(PROB_COLS_DEFAULT),
                        metavar=('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'))
    parser.add_argument('--unmatched-policy', type=str, default='drop', choices=['drop', 'unity', 'error'])
    parser.add_argument('--mark-normalize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--random-index', type=int, default=0)
    parser.add_argument('--n-random-files', type=int, default=5)
    parser.add_argument('--zmin', type=float, default=0.1)
    parser.add_argument('--zmax', type=float, default=0.4)
    parser.add_argument('--h0', type=float, default=100.0)
    parser.add_argument('--om0', type=float, default=0.315)
    parser.add_argument('--grid', type=int, default=256)
    parser.add_argument('--mas', type=str, default='CIC', choices=['NGP', 'CIC', 'TSC', 'PCS'])
    parser.add_argument('--interlacing', type=int, default=2)
    parser.add_argument('--box-padding', type=float, default=50.0)
    parser.add_argument('--boxsize', type=float, default=0.0)
    parser.add_argument('--nthreads', type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument('--random-subsample', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--subtract-shotnoise', action='store_true')
    parser.add_argument('--sky-region', type=str, default='NGC', choices=['NGC', 'ALL'])
    parser.add_argument('--ratio-source', type=str, default='used', choices=['raw', 'used'])
    parser.add_argument('--ratio-denom-min', type=float, default=0.0)
    parser.add_argument('--kmin-ratio', type=float, default=0.05)
    parser.add_argument('--kmax-ratio', type=float, default=0.2)
    parser.add_argument('--outdir', type=str, default='')
    return parser.parse_args()


def resolve_outdir(user_outdir):
    if user_outdir:
        outdir = Path(user_outdir).expanduser().resolve()
    else:
        pscratch = os.environ.get('PSCRATCH')
        if pscratch:
            outdir = Path(pscratch) / 'fisher_info' / 'pypower_entropy_pk'
        else:
            outdir = Path.cwd() / 'outputs' / 'pypower_entropy_pk'
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def read_fits_columns_case_insensitive(path, columns):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if data is None:
            raise RuntimeError(f'No table data found in FITS file: {path}')
        names = list(data.names or [])
        lookup = {name.upper(): name for name in names}
        missing = [col for col in columns if col.upper() not in lookup]
        if missing:
            raise KeyError(f'Missing columns in {path}: {missing}')
        out = {}
        for col in columns:
            out[col] = np.asarray(data[lookup[col.upper()]]).copy()
    return Table(out)


def load_and_stack_randoms(base_dir, tracer, start_index, n_random_files, columns, zmin, zmax):
    indices = list(range(start_index, start_index + n_random_files))
    paths = [Path(base_dir) / f'{tracer}_{idx}_clustering.ran.fits' for idx in indices]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Random catalog not found: {path}')

    random_tables = []
    for path in paths:
        rand_i = read_fits_columns_case_insensitive(str(path), columns)
        m_i = (rand_i['Z'] >= zmin) & (rand_i['Z'] < zmax)
        random_tables.append(rand_i[m_i])

    rand = random_tables[0] if len(random_tables) == 1 else vstack(random_tables)
    return rand, indices, [str(p) for p in paths]


def get_weight_column(table):
    if 'WEIGHT' in table.colnames:
        return np.asarray(table['WEIGHT'], dtype=np.float64)
    return np.ones(len(table), dtype=np.float64)


def radec_to_cartesian(ra_deg, dec_deg, chi):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    chi = np.asarray(chi, dtype=np.float64)
    cos_dec = np.cos(dec)
    x = chi * cos_dec * np.cos(ra)
    y = chi * cos_dec * np.sin(ra)
    z = chi * np.sin(dec)
    return np.column_stack([x, y, z]).astype(np.float64)


def ngc_ra_mask(ra_deg):
    ra = np.asarray(ra_deg, dtype=np.float64)
    return np.isfinite(ra) & (ra > NGC_RA_MIN_DEG) & (ra < NGC_RA_MAX_DEG)


def mas_to_resampler(mas):
    mapping = {'NGP': 'ngp', 'CIC': 'cic', 'TSC': 'tsc', 'PCS': 'pcs'}
    return mapping[mas.upper()]


def compute_box_geometry(pos_d, pos_r, boxsize_arg, box_padding):
    mins = np.minimum(np.min(pos_d, axis=0), np.min(pos_r, axis=0)).astype(np.float64)
    maxs = np.maximum(np.max(pos_d, axis=0), np.max(pos_r, axis=0)).astype(np.float64)
    lengths = maxs - mins

    if boxsize_arg > 0.0:
        boxsize = float(boxsize_arg)
        center = 0.5 * (mins + maxs)
        origin = center - 0.5 * boxsize
    else:
        boxsize = float(np.max(lengths) + 2.0 * box_padding)
        center = 0.5 * (mins + maxs)
        origin = center - 0.5 * boxsize

    if boxsize <= 0.0:
        raise RuntimeError('Invalid boxsize computed from input positions.')

    return boxsize, center, origin


def shift_and_clip_to_box(positions, origin, boxsize):
    shifted = (positions.astype(np.float64) - origin).astype(np.float64)
    upper = np.nextafter(np.float64(boxsize), np.float64(0.0))
    np.clip(shifted, 0.0, upper, out=shifted)
    return shifted


def make_k_edges(boxsize, nmesh):
    dk = 2.0 * np.pi / float(boxsize)
    k_nyquist = np.pi * float(nmesh) / float(boxsize)
    edges = np.arange(0.0, k_nyquist + dk, dk, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([0.0, max(k_nyquist, dk)], dtype=np.float64)
    return edges


def to_scalar_float(value, default=np.nan):
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(np.real(arr.ravel()[0]))


def extract_monopole(poles):
    remove_shotnoise_supported = True
    try:
        pk0 = poles(ell=0, complex=False, remove_shotnoise=False)
    except TypeError:
        remove_shotnoise_supported = False
        pk0 = poles(ell=0, complex=False)

    k = np.asarray(poles.k, dtype=np.float64)
    pk0 = np.asarray(pk0, dtype=np.float64)
    nmodes = getattr(poles, 'nmodes', None)
    if nmodes is None:
        nmodes = np.full_like(k, np.nan, dtype=np.float64)
    else:
        nmodes = np.asarray(nmodes, dtype=np.float64)
    shotnoise = to_scalar_float(getattr(poles, 'shotnoise', np.nan), default=np.nan)
    return {'k': k, 'pk0': pk0, 'nmodes': nmodes, 'shotnoise': shotnoise,
            'remove_shotnoise_supported': remove_shotnoise_supported}


def compute_pk_pypower(data_positions, data_weights, edges, boxsize, boxcenter,
                       nmesh, resampler, interlacing,
                       random_positions=None, random_weights=None):
    kwargs = {'data_positions1': data_positions,
              'data_weights1': data_weights,
              'edges': edges,
              'ells': (0,),
              'position_type': 'pos',
              'boxsize': boxsize,
              'boxcenter': boxcenter,
              'nmesh': nmesh,
              'resampler': resampler,
              'interlacing': interlacing}
    if random_positions is not None:
        kwargs['randoms_positions1'] = random_positions
        kwargs['randoms_weights1'] = random_weights

    result = CatalogFFTPower(**kwargs)
    return extract_monopole(result.poles)


def align_to_k(k_target, k_src, values_src):
    k_target = np.asarray(k_target, dtype=np.float64)
    k_src = np.asarray(k_src, dtype=np.float64)
    values_src = np.asarray(values_src, dtype=np.float64)
    if k_target.shape == k_src.shape and np.allclose(k_target, k_src, rtol=1e-8, atol=1e-12):
        return values_src.copy()
    order = np.argsort(k_src)
    return np.interp(k_target, k_src[order], values_src[order], left=np.nan, right=np.nan)


def compute_normalized_shannon_entropy(probabilities):
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError('Expected a 2D array of probabilities.')
    n_classes = probs.shape[1]
    if n_classes <= 1:
        raise ValueError('Need at least 2 classes for entropy normalization.')

    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, None)
    row_sum = np.sum(probs, axis=1, keepdims=True)
    valid_sum = row_sum[:, 0] > 0.0

    probs_norm = np.zeros_like(probs)
    probs_norm[valid_sum] = probs[valid_sum] / row_sum[valid_sum]

    log_probs = np.zeros_like(probs_norm)
    positive = probs_norm > 0.0
    log_probs[positive] = np.log(probs_norm[positive])
    entropy = -np.sum(probs_norm * log_probs, axis=1)
    entropy_norm = entropy / np.log(float(n_classes))
    entropy_norm = np.clip(entropy_norm, 0.0, 1.0)

    return entropy_norm, probs_norm, valid_sum


def build_entropy_mark(targetid_data, w_data, astra_prob_file, prob_cols, unmatched_policy, normalize_mark):
    astra = read_fits_columns_case_insensitive(astra_prob_file, ['TARGETID'] + list(prob_cols))
    tid_a = np.asarray(astra['TARGETID'], dtype=np.int64)
    probs_a = np.column_stack([np.asarray(astra[col], dtype=np.float64) for col in prob_cols]).astype(np.float64)

    order_a = np.argsort(tid_a)
    tid_sorted = tid_a[order_a]
    probs_sorted = probs_a[order_a]

    tid_unique, unique_first = np.unique(tid_sorted, return_index=True)
    probs_unique = probs_sorted[unique_first]
    n_duplicate_ids = int(tid_sorted.size - tid_unique.size)

    tid_d = np.asarray(targetid_data, dtype=np.int64)
    idx = np.searchsorted(tid_unique, tid_d, side='left')
    valid = idx < tid_unique.size
    matched = np.zeros_like(valid)
    if np.any(valid):
        idx_valid = idx[valid]
        matched[valid] = tid_unique[idx_valid] == tid_d[valid]

    n_unmatched = int(np.sum(~matched))
    if n_unmatched > 0 and unmatched_policy == 'error':
        raise RuntimeError(f'{n_unmatched} data TARGETID values were not matched in astra-prob-file.')

    mark = np.zeros(len(tid_d), dtype=np.float64)
    entropy_matched = np.array([], dtype=np.float64)
    n_invalid_probsum = 0
    if np.any(matched):
        probs_matched = probs_unique[idx[matched]]
        entropy_matched, _, valid_sum = compute_normalized_shannon_entropy(probs_matched)
        n_invalid_probsum = int(np.sum(~valid_sum))
        mark[matched] = entropy_matched

    if unmatched_policy == 'unity':
        mark[~matched] = 1.0

    sw = float(np.sum(w_data, dtype=np.float64))
    if sw <= 0.0:
        raise RuntimeError('Non-positive sum of baseline data weights; cannot normalize marks.')

    mark_weighted_mean_before = float(np.sum(w_data.astype(np.float64) * mark.astype(np.float64)) / sw)
    if normalize_mark and mark_weighted_mean_before > 0.0:
        mark /= mark_weighted_mean_before
    mark_weighted_mean_after = float(np.sum(w_data.astype(np.float64) * mark.astype(np.float64)) / sw)

    entropy_stats = {'count': int(entropy_matched.size),
                     'mean': float(np.mean(entropy_matched)) if entropy_matched.size > 0 else np.nan,
                     'std': float(np.std(entropy_matched)) if entropy_matched.size > 0 else np.nan,
                     'min': float(np.min(entropy_matched)) if entropy_matched.size > 0 else np.nan,
                     'max': float(np.max(entropy_matched)) if entropy_matched.size > 0 else np.nan,
                     'p16': float(np.quantile(entropy_matched, 0.16)) if entropy_matched.size > 0 else np.nan,
                     'p50': float(np.quantile(entropy_matched, 0.50)) if entropy_matched.size > 0 else np.nan,
                     'p84': float(np.quantile(entropy_matched, 0.84)) if entropy_matched.size > 0 else np.nan}

    info = {'n_data': int(len(tid_d)),
            'n_astra_rows': int(len(tid_a)),
            'n_astra_unique_targetid': int(len(tid_unique)),
            'n_astra_duplicate_targetid': n_duplicate_ids,
            'n_matched': int(np.sum(matched)),
            'n_unmatched': n_unmatched,
            'matched_fraction': float(np.mean(matched)),
            'n_invalid_probability_sum': n_invalid_probsum,
            'prob_cols': list(prob_cols),
            'entropy_formula': 'H_norm = -sum(p_i ln p_i)/ln(Nclass)',
            'n_classes': int(len(prob_cols)),
            'unmatched_policy': unmatched_policy,
            'mark_normalize': bool(normalize_mark),
            'mark_weighted_mean_before_norm': mark_weighted_mean_before,
            'mark_weighted_mean_after_norm': mark_weighted_mean_after,
            'entropy_stats_matched': entropy_stats}
    return mark, matched, info


def main():
    args = parse_args()
    outdir = resolve_outdir(args.outdir)
    tag = f'{args.tracer}_z{args.zmin:.3f}_{args.zmax:.3f}_N{args.grid}'
    prob_cols = list(args.prob_cols)

    data_path = Path(args.base_dir) / f'{args.tracer}_clustering.dat.fits'
    random_indices = list(range(args.random_index, args.random_index + args.n_random_files))
    rand_paths = [Path(args.base_dir) / f'{args.tracer}_{idx}_clustering.ran.fits' for idx in random_indices]

    t0 = time.time()
    print(f'---> data catalog:   {data_path}')
    print('---> random catalogs: ' + ', '.join(str(p) for p in rand_paths[:5])
          + (' ...' if len(rand_paths) > 5 else ''))
    print(f'---> random stacking: start={args.random_index}, '
          f'n_files={args.n_random_files}, indices={random_indices}')
    print(f'---> astra probs: {args.astra_prob_file}')
    print(f'---> prob columns: {prob_cols}')
    print(f'---> outdir: {outdir}')
    print(f'---> nmesh: {args.grid}')
    print(f'---> mas/resampler: {args.mas}/{mas_to_resampler(args.mas)}')
    print(f'---> interlacing: {args.interlacing}')

    real_cols = ['RA', 'DEC', 'Z', 'WEIGHT', 'TARGETID']
    rand_cols = ['RA', 'DEC', 'Z', 'WEIGHT']

    real = read_fits_columns_case_insensitive(str(data_path), real_cols)
    rand, random_indices_used, rand_paths_used = load_and_stack_randoms(base_dir=args.base_dir,
                                                                        tracer=args.tracer,
                                                                        start_index=args.random_index,
                                                                        n_random_files=args.n_random_files,
                                                                        columns=rand_cols,
                                                                        zmin=args.zmin,
                                                                        zmax=args.zmax)

    n_data_before_selection = len(real)
    n_random_before_selection = len(rand)

    md = (real['Z'] >= args.zmin) & (real['Z'] < args.zmax)
    mr = np.ones(len(rand), dtype=bool)
    sky_region_desc = 'ALL sky'
    if args.sky_region == 'NGC':
        md &= ngc_ra_mask(real['RA'])
        mr &= ngc_ra_mask(rand['RA'])
        sky_region_desc = f'NGC sky ({NGC_RA_MIN_DEG:.0f} < RA < {NGC_RA_MAX_DEG:.0f} deg)'
    real = real[md]
    rand = rand[mr]

    print(f'---> sky region selection: {sky_region_desc}')
    print(f'---> selected data objects:   {len(real)} (from {n_data_before_selection})')
    print(f'---> selected random objects: {len(rand)} (from {n_random_before_selection})')

    if len(real) == 0:
        raise RuntimeError('No data objects selected after cuts.')
    if len(rand) == 0:
        raise RuntimeError('No random objects selected after cuts.')

    if args.random_subsample < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = rng.random(len(rand)) < args.random_subsample
        rand = rand[keep]
        print(f'---> random subsample fraction={args.random_subsample:.3f}, kept={len(rand)}')
    if len(rand) == 0:
        raise RuntimeError('Random catalog is empty after random subsampling.')

    cosmo = FlatLambdaCDM(H0=args.h0, Om0=args.om0)
    chi_d = np.asarray(cosmo.comoving_distance(real['Z']).value, dtype=np.float64)
    chi_r = np.asarray(cosmo.comoving_distance(rand['Z']).value, dtype=np.float64)

    pos_d = radec_to_cartesian(real['RA'], real['DEC'], chi_d)
    pos_r = radec_to_cartesian(rand['RA'], rand['DEC'], chi_r)
    w_d = get_weight_column(real)
    w_r = get_weight_column(rand)
    targetid_d = np.asarray(real['TARGETID'], dtype=np.int64)

    boxsize, center, origin = compute_box_geometry(pos_d, pos_r, boxsize_arg=args.boxsize,
                                                   box_padding=args.box_padding)
    pos_d = shift_and_clip_to_box(pos_d, origin=origin, boxsize=boxsize)
    pos_r = shift_and_clip_to_box(pos_r, origin=origin, boxsize=boxsize)
    boxcenter = np.array([0.5 * boxsize] * 3, dtype=np.float64)

    sw_d = float(np.sum(w_d, dtype=np.float64))
    sw_r = float(np.sum(w_r, dtype=np.float64))
    sw2_d = float(np.sum(w_d ** 2, dtype=np.float64))
    if sw_d <= 0.0:
        raise RuntimeError('Sum of data weights is <= 0.')
    if sw_r <= 0.0:
        raise RuntimeError('Sum of random weights is <= 0.')
    alpha = sw_d / sw_r
    volume = boxsize ** 3
    print(f'---> boxsize [Mpc/h]: {boxsize:.3f}')

    resampler = mas_to_resampler(args.mas)
    edges = make_k_edges(boxsize=boxsize, nmesh=args.grid)

    print('---> computing unmarked P(k) with PyPower ...')
    pk_unmarked = compute_pk_pypower(data_positions=pos_d,
                                     data_weights=w_d,
                                     random_positions=pos_r,
                                     random_weights=w_r,
                                     edges=edges,
                                     boxsize=boxsize,
                                     boxcenter=boxcenter,
                                     nmesh=args.grid,
                                     resampler=resampler,
                                     interlacing=args.interlacing)
    k = pk_unmarked['k']
    pk_raw = pk_unmarked['pk0']
    nmodes = pk_unmarked['nmodes']

    shotnoise_unmarked_analytic = volume * sw2_d / (sw_d * sw_d)
    shotnoise_unmarked = (pk_unmarked['shotnoise']
                          if np.isfinite(pk_unmarked['shotnoise'])
                          else shotnoise_unmarked_analytic)
    pk_used = pk_raw - shotnoise_unmarked if args.subtract_shotnoise else pk_raw.copy()

    print('---> building entropy mark from ASTRA probabilities ...')
    mark_entropy, matched, mark_info = build_entropy_mark(targetid_data=targetid_d,
                                                          w_data=w_d,
                                                          astra_prob_file=args.astra_prob_file,
                                                          prob_cols=prob_cols,
                                                          unmatched_policy=args.unmatched_policy,
                                                          normalize_mark=args.mark_normalize)
    if np.any(~matched):
        print(f"---> unmatched TARGETID: {int(np.sum(~matched))} (policy={args.unmatched_policy})")

    w_dm = w_d * mark_entropy
    sw_dm = float(np.sum(w_dm, dtype=np.float64))
    sw2_dm = float(np.sum(w_dm ** 2, dtype=np.float64))
    if sw_dm <= 0.0:
        raise RuntimeError('Entropy-marked total weight is <= 0. Check matching and mark settings.')

    print('---> computing entropy-marked P(k) with PyPower ...')
    pk_marked_result = compute_pk_pypower(data_positions=pos_d,
                                          data_weights=w_dm,
                                          random_positions=pos_r,
                                          random_weights=w_r,
                                          edges=edges,
                                          boxsize=boxsize,
                                          boxcenter=boxcenter,
                                          nmesh=args.grid,
                                          resampler=resampler,
                                          interlacing=args.interlacing)
    pk_marked_raw = align_to_k(k, pk_marked_result['k'], pk_marked_result['pk0'])

    shotnoise_marked_analytic = volume * sw2_dm / (sw_dm * sw_dm)
    shotnoise_marked = (pk_marked_result['shotnoise']
                        if np.isfinite(pk_marked_result['shotnoise'])
                        else shotnoise_marked_analytic)
    pk_marked_used = pk_marked_raw - shotnoise_marked if args.subtract_shotnoise else pk_marked_raw.copy()

    if args.ratio_source == 'raw':
        ratio_num = pk_marked_raw
        ratio_den = pk_raw
    else:
        ratio_num = pk_marked_used
        ratio_den = pk_used

    ratio_arr = np.full_like(pk_used, np.nan)
    ratio_masked_arr = np.full_like(pk_used, np.nan)
    delta_arr = np.full_like(pk_used, np.nan)

    ratio_kmask = k >= args.kmin_ratio
    if args.kmax_ratio > 0.0:
        ratio_kmask &= k <= args.kmax_ratio

    good_ratio = (np.isfinite(ratio_num)
                  & np.isfinite(ratio_den)
                  & (np.abs(ratio_den) > args.ratio_denom_min))
    ratio_arr[good_ratio] = ratio_num[good_ratio] / ratio_den[good_ratio]
    delta_arr[good_ratio] = ratio_num[good_ratio] - ratio_den[good_ratio]
    good_ratio_masked = good_ratio & ratio_kmask
    ratio_masked_arr[good_ratio_masked] = ratio_arr[good_ratio_masked]

    csv_path = outdir / f'pk_entropy_marked_{tag}.csv'
    fig_pk_path = outdir / f'pk_entropy_marked_{tag}.png'
    fig_ratio_path = outdir / f'pk_entropy_over_unmarked_{tag}.png'
    fig_delta_path = outdir / f'pk_entropy_minus_unmarked_{tag}.png'
    meta_path = outdir / f'run_metadata_pk_entropy_marked_{tag}.json'

    csv_cols = [k, pk_raw, pk_used, nmodes,
                pk_marked_raw, pk_marked_used,
                ratio_arr, ratio_masked_arr,
                delta_arr]
    csv_header_cols = ['k_h_mpc',
                       'pk_unmarked_raw',
                       'pk_unmarked_used',
                       'nmodes',
                       'pk_entropy_raw',
                       'pk_entropy_used',
                       'pk_entropy_over_unmarked',
                       'pk_entropy_over_unmarked_masked',
                       'delta_pk_entropy_minus_unmarked']
    np.savetxt(csv_path, np.column_stack(csv_cols), delimiter=',',
               header=','.join(csv_header_cols), comments='')

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    m_un = (k > 0.0) & np.isfinite(pk_used) & (pk_used > 0.0)
    m_mk = (k > 0.0) & np.isfinite(pk_marked_used) & (pk_marked_used > 0.0)
    ax.loglog(k[m_un], pk_used[m_un], lw=1.5, color=UNMARKED_COLOR, label=r'$P_{\rm unmarked}(k)$')
    ax.loglog(k[m_mk], pk_marked_used[m_mk], lw=1.5, color=MARKED_COLOR,
              label=r'$P_{\rm entropy}(k)$')
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    title = f'{args.tracer} entropy mark  ({args.zmin:.2f} < z < {args.zmax:.2f})'
    if args.subtract_shotnoise:
        title += '  [shot-noise subtracted]'
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_pk_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    m_ratio = (k > 0.0) & np.isfinite(ratio_masked_arr)
    ax.semilogx(k[m_ratio], ratio_masked_arr[m_ratio], lw=1.6, color=MARKED_COLOR)
    ax.axhline(1.0, color='black', ls='--', lw=1.0, alpha=0.8)
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P_{\rm entropy}(k) / P_{\rm unmarked}(k)$')
    k_range_txt = f'{args.kmin_ratio:.3f} < k'
    if args.kmax_ratio > 0.0:
        k_range_txt += f' < {args.kmax_ratio:.3f}'
    ax.set_title(f'{args.tracer} entropy ratio  [{k_range_txt}]')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(fig_ratio_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    m_delta = (k > 0.0) & np.isfinite(delta_arr)
    m_delta &= k >= args.kmin_ratio
    if args.kmax_ratio > 0.0:
        m_delta &= k <= args.kmax_ratio
    ax.semilogx(k[m_delta], delta_arr[m_delta], lw=1.6, color=MARKED_COLOR)
    ax.axhline(0.0, color='black', ls='--', lw=1.0, alpha=0.8)
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$\Delta P(k) = P_{\rm entropy}(k)-P_{\rm unmarked}(k)$')
    ax.set_title(f'{args.tracer} entropy delta  [{k_range_txt}]')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(fig_delta_path, dpi=300)
    plt.close(fig)

    elapsed = time.time() - t0
    metadata = {'tracer': args.tracer,
                'zmin': args.zmin,
                'zmax': args.zmax,
                'base_dir': args.base_dir,
                'astra_prob_file': args.astra_prob_file,
                'prob_cols': prob_cols,
                'unmatched_policy': args.unmatched_policy,
                'mark_normalize': bool(args.mark_normalize),
                'marking': mark_info,
                'random_index': args.random_index,
                'n_random_files': args.n_random_files,
                'random_indices_used': random_indices_used,
                'random_catalog_paths_used': rand_paths_used,
                'n_data': len(real),
                'n_random': len(rand),
                'n_data_before_selection': n_data_before_selection,
                'n_random_before_selection': n_random_before_selection,
                'sky_region': args.sky_region,
                'sky_region_mask': sky_region_desc,
                'sum_w_data': sw_d,
                'sum_w_random': sw_r,
                'sum_w_entropy_marked': sw_dm,
                'alpha_unmarked': alpha,
                'grid': args.grid,
                'mas': args.mas,
                'resampler': resampler,
                'interlacing': args.interlacing,
                'boxsize_mpc_h': boxsize,
                'box_center_xyz_mpc_h': center.tolist(),
                'box_padding_mpc_h': args.box_padding,
                'volume_mpc3_h3': volume,
                'subtract_shotnoise': args.subtract_shotnoise,
                'ratio_source': args.ratio_source,
                'ratio_denom_min': args.ratio_denom_min,
                'kmin_ratio_h_mpc': args.kmin_ratio,
                'kmax_ratio_h_mpc': args.kmax_ratio,
                'shotnoise_unmarked': shotnoise_unmarked,
                'shotnoise_unmarked_analytic': shotnoise_unmarked_analytic,
                'shotnoise_unmarked_pypower': pk_unmarked['shotnoise'],
                'shotnoise_entropy_marked': shotnoise_marked,
                'shotnoise_entropy_marked_analytic': shotnoise_marked_analytic,
                'shotnoise_entropy_marked_pypower': pk_marked_result['shotnoise'],
                'nthreads_hint': args.nthreads,
                'random_subsample': args.random_subsample,
                'elapsed_sec': elapsed,
                'outputs': {'pk_csv': str(csv_path),
                            'pk_plot': str(fig_pk_path),
                            'pk_ratio_plot': str(fig_ratio_path),
                            'pk_delta_plot': str(fig_delta_path)},
                'engine': 'pypower.CatalogFFTPower',
                'remove_shotnoise_flag_supported': bool(pk_unmarked['remove_shotnoise_supported']),
                'notes': ['Entropy mark is normalized Shannon entropy from ASTRA probabilities.',
                          'Entropy formula: H_norm = -sum_i p_i ln(p_i) / ln(4).']}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'---> wrote: {csv_path}')
    print(f'---> wrote: {fig_pk_path}')
    print(f'---> wrote: {fig_ratio_path}')
    print(f'---> wrote: {fig_delta_path}')
    print(f'---> wrote: {meta_path}')
    print(f'---> elapsed: {elapsed:.2f} s')


if __name__ == '__main__':
    main()