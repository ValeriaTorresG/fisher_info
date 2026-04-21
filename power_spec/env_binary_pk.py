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
ENV_LABELS_DEFAULT = ['void', 'sheet', 'filament', 'knot']
ENV_COLOR_MAP = {'void': 'cyan', 'sheet': 'orange', 'filament': 'limegreen', 'knot': 'magenta'}
NGC_RA_MIN_DEG = 90.0
NGC_RA_MAX_DEG = 300.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/PIP/')
    parser.add_argument('--tracer', type=str, default='BGS_ANY')
    parser.add_argument('--astra-prob-file', type=str, default='/pscratch/sd/v/vtorresg/cosmic-web/dr2/probabilities/bgs/ngc/zone_NGC_BGS_probability_iterdata.fits.gz')
    parser.add_argument('--prob-cols', nargs=4, default=list(PROB_COLS_DEFAULT), metavar=('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'))
    parser.add_argument('--env-labels', nargs=4, default=list(ENV_LABELS_DEFAULT), metavar=('void', 'sheet', 'filament', 'knot'))
    parser.add_argument('--unmatched-policy', type=str, default='drop', choices=['drop', 'error'])
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
    parser.add_argument('--sky-region', type=str, default='NGC', choices=['NGC', 'ALL'])
    parser.add_argument('--subtract-shotnoise', action='store_true')
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
            outdir = Path(pscratch) / 'fisher_info' / 'pypower_env_pk'
        else:
            outdir = Path.cwd() / 'outputs' / 'pypower_env_pk'
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


def build_hard_environment_classes(targetid_data, astra_prob_file, prob_cols):
    astra = read_fits_columns_case_insensitive(astra_prob_file, ['TARGETID'] + list(prob_cols))
    tid_a = np.asarray(astra['TARGETID'], dtype=np.int64)
    probs_a = np.column_stack([np.asarray(astra[col], dtype=np.float64) for col in prob_cols]).astype(np.float64)
    probs_a = np.nan_to_num(probs_a, nan=0.0, posinf=0.0, neginf=0.0)
    probs_a = np.clip(probs_a, 0.0, 1.0)

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

    class_index = np.full(len(tid_d), -1, dtype=np.int16)
    if np.any(matched):
        class_index[matched] = np.argmax(probs_unique[idx[matched]], axis=1).astype(np.int16)

    info = {'n_data': int(len(tid_d)),
            'n_astra_rows': int(len(tid_a)),
            'n_astra_unique_targetid': int(len(tid_unique)),
            'n_astra_duplicate_targetid': n_duplicate_ids,
            'n_matched': int(np.sum(matched)),
            'matched_fraction': float(np.mean(matched))}
    return class_index, matched, info


def main():
    args = parse_args()
    if len(set(args.env_labels)) != 4:
        raise ValueError('--env-labels must contain 4 unique names.')

    outdir = resolve_outdir(args.outdir)
    tag = f'{args.tracer}_z{args.zmin:.3f}_{args.zmax:.3f}_N{args.grid}'
    prob_cols = list(args.prob_cols)
    env_labels = [str(lbl).strip() for lbl in args.env_labels]
    class_index_map = {label: i for i, label in enumerate(env_labels)}

    data_path = Path(args.base_dir) / f'{args.tracer}_clustering.dat.fits'
    random_indices = list(range(args.random_index, args.random_index + args.n_random_files))
    rand_paths = [Path(args.base_dir) / f'{args.tracer}_{idx}_clustering.ran.fits' for idx in random_indices]

    t0 = time.time()
    print(f'---> data catalog:   {data_path}')
    print('---> random catalogs: ' + ', '.join(str(p) for p in rand_paths[:5])
          + (' ...' if len(rand_paths) > 5 else ''))
    print(f'---> random stacking: start={args.random_index}, '
          f'n_files={args.n_random_files}, indices={random_indices}')
    print(f'---> astra probs:     {args.astra_prob_file}')
    print(f'---> prob columns:    {prob_cols}')
    print(f'---> environment map: {class_index_map}')
    print(f'---> outdir:          {outdir}')
    print(f'---> nmesh:           {args.grid}')
    print(f'---> mas/resampler:   {args.mas}/{mas_to_resampler(args.mas)}')
    print(f'---> interlacing:     {args.interlacing}')

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
    print(f'---> selected data objects: {len(real)} (from {n_data_before_selection})')
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
    sw2_r = float(np.sum(w_r ** 2, dtype=np.float64))
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

    print('---> classifying environments from max probability ...')
    class_index, matched, class_info = build_hard_environment_classes(
        targetid_data=targetid_d, astra_prob_file=args.astra_prob_file, prob_cols=prob_cols)
    n_unmatched = int(np.sum(~matched))
    if n_unmatched > 0 and args.unmatched_policy == 'error':
        raise RuntimeError(f'{n_unmatched} data TARGETID values were not matched in astra-prob-file.')
    if n_unmatched > 0:
        print(f'---> unmatched TARGETID in astra-prob-file: {n_unmatched} (treated as 0 weight in all environments)')

    sw_d_matched = float(np.sum(w_d[matched], dtype=np.float64))

    pk_env_raw = {}
    pk_env_used = {}
    ratio_env_over_unmarked = {}
    ratio_env_over_unmarked_masked = {}
    shotnoise_env = {}
    shotnoise_env_pypower = {}
    class_stats = {}
    ratio_kmask = k >= args.kmin_ratio
    if args.kmax_ratio > 0.0:
        ratio_kmask &= k <= args.kmax_ratio

    for i_env, env_label in enumerate(env_labels):
        print(f'---> computing environment P(k): {env_label}')
        mask_env = class_index == i_env
        w_env = w_d * mask_env.astype(np.float64)
        n_env = int(np.sum(mask_env))
        sw_env = float(np.sum(w_env, dtype=np.float64))
        sw2_env = float(np.sum(w_env ** 2, dtype=np.float64))

        frac_count_matched = float(n_env / np.sum(matched)) if np.sum(matched) > 0 else np.nan
        frac_weight_all = float(sw_env / sw_d) if sw_d > 0 else np.nan
        frac_weight_matched = float(sw_env / sw_d_matched) if sw_d_matched > 0 else np.nan

        class_stats[env_label] = {'class_index': int(i_env),
                                  'n_objects': n_env,
                                  'fraction_of_matched_by_count': frac_count_matched,
                                  'sum_weights': sw_env,
                                  'fraction_of_all_weight': frac_weight_all,
                                  'fraction_of_matched_weight': frac_weight_matched}

        if sw_env <= 0.0:
            pk_env_raw[env_label] = np.full_like(pk_raw, np.nan)
            pk_env_used[env_label] = np.full_like(pk_raw, np.nan)
            ratio_env_over_unmarked[env_label] = np.full_like(pk_raw, np.nan)
            ratio_env_over_unmarked_masked[env_label] = np.full_like(pk_raw, np.nan)
            shotnoise_env[env_label] = np.nan
            shotnoise_env_pypower[env_label] = np.nan
            continue

        pk_env_result = compute_pk_pypower(data_positions=pos_d,
                                           data_weights=w_env,
                                           random_positions=pos_r,
                                           random_weights=w_r,
                                           edges=edges,
                                           boxsize=boxsize,
                                           boxcenter=boxcenter,
                                           nmesh=args.grid,
                                           resampler=resampler,
                                           interlacing=args.interlacing)
        pk_env_raw[env_label] = align_to_k(k, pk_env_result['k'], pk_env_result['pk0'])

        shotnoise_env_analytic = volume * sw2_env / (sw_env * sw_env)
        shotnoise_env_pypower[env_label] = pk_env_result['shotnoise']
        shotnoise_env[env_label] = (shotnoise_env_pypower[env_label]
                                    if np.isfinite(shotnoise_env_pypower[env_label])
                                    else shotnoise_env_analytic)

        pk_env_used[env_label] = (pk_env_raw[env_label] - shotnoise_env[env_label]
                                  if args.subtract_shotnoise else pk_env_raw[env_label].copy())

        ratio_arr = np.full_like(pk_used, np.nan)
        ratio_masked_arr = np.full_like(pk_used, np.nan)
        good_ratio = (np.isfinite(pk_env_used[env_label])
                      & np.isfinite(pk_used)
                      & (np.abs(pk_used) > args.ratio_denom_min))
        ratio_arr[good_ratio] = pk_env_used[env_label][good_ratio] / pk_used[good_ratio]
        good_ratio_masked = good_ratio & ratio_kmask
        ratio_masked_arr[good_ratio_masked] = ratio_arr[good_ratio_masked]
        ratio_env_over_unmarked[env_label] = ratio_arr
        ratio_env_over_unmarked_masked[env_label] = ratio_masked_arr

    csv_path = outdir / f'pk_env_binary_{tag}.csv'
    fig_path = outdir / f'pk_env_binary_{tag}.png'
    fig_ratio_path = outdir / f'pk_env_over_unmarked_{tag}.png'
    meta_path = outdir / f'run_metadata_pk_env_binary_{tag}.json'

    csv_cols = [k, pk_raw, pk_used, nmodes]
    csv_header_cols = ['k_h_mpc', 'pk_unmarked_raw', 'pk_unmarked_used', 'nmodes']
    for env_label in env_labels:
        safe = env_label.lower().replace(' ', '_')
        csv_cols.extend([pk_env_raw[env_label],
                         pk_env_used[env_label],
                         ratio_env_over_unmarked[env_label],
                         ratio_env_over_unmarked_masked[env_label]])
        csv_header_cols.extend([f'pk_{safe}_raw',
                                f'pk_{safe}_used',
                                f'pk_{safe}_over_unmarked',
                                f'pk_{safe}_over_unmarked_masked'])
    np.savetxt(csv_path, np.column_stack(csv_cols), delimiter=',',
               header=','.join(csv_header_cols), comments='')

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    m_un = (k > 0.0) & np.isfinite(pk_used) & (pk_used > 0.0)
    ax.loglog(k[m_un], pk_used[m_un], lw=1.6, color='white', label='unmarked')
    for env_label in env_labels:
        m_env = (k > 0.0) & np.isfinite(pk_env_used[env_label]) & (pk_env_used[env_label] > 0.0)
        ax.loglog(k[m_env], pk_env_used[env_label][m_env], lw=1.4,
                  color=ENV_COLOR_MAP.get(env_label.lower(), None),
                  label=env_label)
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    title = f'{args.tracer} env binary weights ({args.zmin:.2f} < z < {args.zmax:.2f})'
    if args.subtract_shotnoise:
        title += '  [shot-noise subtracted]'
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for env_label in env_labels:
        m_ratio = (k > 0.0) & np.isfinite(ratio_env_over_unmarked_masked[env_label])
        ax.semilogx(k[m_ratio], ratio_env_over_unmarked_masked[env_label][m_ratio], lw=1.4,
                    color=ENV_COLOR_MAP.get(env_label.lower(), None),
                    label=env_label)
    ax.axhline(1.0, color='black', ls='--', lw=1.0, alpha=0.8)
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P_{\rm env}(k) / P_{\rm unmarked}(k)$')
    k_range_txt = f'{args.kmin_ratio:.3f} < k'
    if args.kmax_ratio > 0.0:
        k_range_txt += f' < {args.kmax_ratio:.3f}'
    ax.set_title(f'{args.tracer} env / unmarked  ({args.zmin:.2f} < z < {args.zmax:.2f})  [{k_range_txt}]')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_ratio_path, dpi=300)
    plt.close(fig)

    elapsed = time.time() - t0
    metadata = {'tracer': args.tracer,
                'zmin': args.zmin,
                'zmax': args.zmax,
                'base_dir': args.base_dir,
                'astra_prob_file': args.astra_prob_file,
                'prob_cols': prob_cols,
                'env_labels': env_labels,
                'class_index_map': class_index_map,
                'unmatched_policy': args.unmatched_policy,
                'classification_info': class_info,
                'class_stats': class_stats,
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
                'ratio_denom_min': args.ratio_denom_min,
                'kmin_ratio_h_mpc': args.kmin_ratio,
                'kmax_ratio_h_mpc': args.kmax_ratio,
                'shotnoise_unmarked': shotnoise_unmarked,
                'shotnoise_unmarked_analytic': shotnoise_unmarked_analytic,
                'shotnoise_unmarked_pypower': pk_unmarked['shotnoise'],
                'shotnoise_env': shotnoise_env,
                'shotnoise_env_pypower': shotnoise_env_pypower,
                'nthreads_hint': args.nthreads,
                'random_subsample': args.random_subsample,
                'elapsed_sec': elapsed,
                'outputs': {'pk_csv': str(csv_path),
                            'pk_plot': str(fig_path),
                            'pk_env_over_unmarked_plot': str(fig_ratio_path)},
                'engine': 'pypower.CatalogFFTPower',
                'remove_shotnoise_flag_supported': bool(pk_unmarked['remove_shotnoise_supported']),
                'notes': ['Hard environment classification uses argmax over probability columns.',
                          'For each environment spectrum, data weights are binary (1 for class, 0 otherwise).']}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'---> wrote: {csv_path}')
    print(f'---> wrote: {fig_path}')
    print(f'---> wrote: {fig_ratio_path}')
    print(f'---> wrote: {meta_path}')
    print(f'---> elapsed: {elapsed:.2f} s')


if __name__ == '__main__':
    main()