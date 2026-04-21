import argparse, json, os, time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('dark_background')
plt.rcParams.update({'grid.linewidth': 0.15,
                     'text.usetex': True})

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack
from pypower import CatalogFFTPower

colors = ['cyan', 'orange', 'limegreen', 'magenta']
UNMARKED_COLOR = 'white'
WINDOW_COLOR = 'grey'
MARK_COLS_ALL = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
MARK_COLOR_MAP = {'PVOID': colors[0],
                  'PSHEET': colors[1],
                  'PFILAMENT': colors[2],
                  'PKNOT': colors[3]}
NGC_RA_MIN_DEG = 90.0
NGC_RA_MAX_DEG = 300.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str,
                        default='/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/PIP/')
    parser.add_argument('--tracer', type=str, required=True)
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
    parser.add_argument('--window-pk', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ratio-source', type=str, default='raw', choices=['raw', 'used'])
    parser.add_argument('--kmin-ratio', type=float, default=0.05)
    parser.add_argument('--kmax-ratio', type=float, default=0.2)
    parser.add_argument('--ratio-denom-min', type=float, default=0.0)
    parser.add_argument('--sky-region', type=str, default='NGC', choices=['NGC', 'ALL'])
    parser.add_argument('--astra-prob-file', type=str, default='')
    parser.add_argument('--mark-col', type=str,
                        default='', choices=['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'])
    parser.add_argument('--mark-cols', nargs='+', default=list(MARK_COLS_ALL),
                        choices=['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'])
    parser.add_argument('--mark-power', type=float, default=1.0)
    parser.add_argument('--mark-eps', type=float, default=0.0)
    parser.add_argument('--mark-normalize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--unmatched-policy', type=str, default='unity', choices=['unity', 'drop'])
    parser.add_argument('--outdir', type=str, default='')
    return parser.parse_args()


def resolve_outdir(user_outdir):
    if user_outdir:
        outdir = Path(user_outdir).expanduser().resolve()
    else:
        pscratch = os.environ.get('PSCRATCH')
        if pscratch:
            outdir = Path(pscratch) / 'fisher_info' / 'pypower_pk'
        else:
            outdir = Path.cwd() / 'outputs' / 'pypower_pk'
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def get_weight_column(table):
    if 'WEIGHT' in table.colnames:
        return np.asarray(table['WEIGHT'], dtype=np.float64)
    return np.ones(len(table), dtype=np.float64)


def read_fits_columns(path, columns):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if data is None:
            raise RuntimeError(f'No table data found in FITS file: {path}')
        names = set(data.names or [])
        missing = [col for col in columns if col not in names]
        if missing:
            raise KeyError(f'Missing columns in {path}: {missing}')
        out = Table({col: np.asarray(data[col]).copy() for col in columns})
    return out


def load_and_stack_randoms(base_dir, tracer, start_index, n_random_files, columns, zmin, zmax):
    indices = list(range(start_index, start_index + n_random_files))
    paths = [Path(base_dir) / f'{tracer}_{idx}_clustering.ran.fits' for idx in indices]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Random catalog not found: {path}')

    random_tables = []
    for path in paths:
        rand_i = read_fits_columns(str(path), columns)
        m_i = (rand_i['Z'] >= zmin) & (rand_i['Z'] < zmax)
        random_tables.append(rand_i[m_i])

    rand = random_tables[0] if len(random_tables) == 1 else vstack(random_tables)
    return rand, indices, [str(p) for p in paths]


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
    k_sorted = k_src[order]
    values_sorted = values_src[order]
    return np.interp(k_target, k_sorted, values_sorted, left=np.nan, right=np.nan)


def build_mark_arrays(targetid_data, w_data, astra_path, mark_cols, mark_power, mark_eps,
                      unmatched_policy, normalize):
    cols = ['TARGETID'] + mark_cols
    astra = read_fits_columns(astra_path, cols)
    tid_a = np.asarray(astra['TARGETID'], dtype=np.int64)
    order_a = np.argsort(tid_a)
    tid_sorted = tid_a[order_a]

    tid_d = np.asarray(targetid_data, dtype=np.int64)
    idx = np.searchsorted(tid_sorted, tid_d, side='left')
    valid = idx < tid_sorted.size
    matched = np.zeros_like(valid)
    if np.any(valid):
        valid_idx = idx[valid]
        matched[valid] = tid_sorted[valid_idx] == tid_d[valid]
    valid = matched

    w_sum = float(np.sum(w_data, dtype=np.float64))
    if w_sum <= 0.0:
        raise RuntimeError('Non-positive sum of baseline data weights; cannot normalize marks.')

    mark_arrays = {}
    mark_infos = {}
    for mark_col in mark_cols:
        prob = np.asarray(astra[mark_col], dtype=np.float64)
        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        prob = np.clip(prob, 0.0, 1.0)
        mark_a = np.power(prob + mark_eps, mark_power).astype(np.float64)
        mark_sorted = mark_a[order_a]

        mark = np.ones(len(tid_d), dtype=np.float64)
        mark[valid] = mark_sorted[idx[valid]]
        if unmatched_policy == 'drop':
            mark[~valid] = 0.0

        mean_mark_w = float(np.sum(w_data.astype(np.float64) * mark.astype(np.float64))) / w_sum
        if normalize and mean_mark_w > 0.0:
            mark /= mean_mark_w

        info = {'n_data': int(len(tid_d)),
                'n_astra': int(len(tid_a)),
                'n_matched': int(np.sum(valid)),
                'matched_fraction': float(np.mean(valid)),
                'mark_col': mark_col,
                'mark_power': mark_power,
                'mark_eps': mark_eps,
                'mark_weighted_mean_before_norm': mean_mark_w,
                'mark_weighted_mean_after_norm': float(np.sum(w_data.astype(np.float64) * mark.astype(np.float64)) / w_sum),
                'unmatched_policy': unmatched_policy,
                'normalize': bool(normalize)}
        mark_arrays[mark_col] = mark
        mark_infos[mark_col] = info

    return mark_arrays, mark_infos


def main():
    args = parse_args()
    outdir = resolve_outdir(args.outdir)
    tag = f'{args.tracer}_z{args.zmin:.3f}_{args.zmax:.3f}_N{args.grid}'
    mark_cols = [args.mark_col] if args.mark_col else list(dict.fromkeys(args.mark_cols))

    data_path = Path(args.base_dir) / f'{args.tracer}_clustering.dat.fits'
    random_indices = list(range(args.random_index, args.random_index + args.n_random_files))
    rand_paths = [Path(args.base_dir) / f'{args.tracer}_{idx}_clustering.ran.fits' for idx in random_indices]

    t0 = time.time()
    print(f'---> data catalog:   {data_path}')
    print('---> random catalogs: ' + ', '.join(str(p) for p in rand_paths[:5])
          + (' ...' if len(rand_paths) > 5 else ''))
    print(f'---> random stacking: start={args.random_index}, '
          f'n_files={args.n_random_files}, indices={random_indices}')
    print(f'---> outdir:         {outdir}')
    print(f'---> nmesh:          {args.grid}')
    print(f'---> mas/resampler:  {args.mas}/{mas_to_resampler(args.mas)}')
    print(f'---> interlacing:    {args.interlacing}')
    print(f'---> nthreads hint:  {args.nthreads}')

    base_cols = ['RA', 'DEC', 'Z', 'WEIGHT']
    use_mark = bool(args.astra_prob_file)
    real_cols = base_cols + (['TARGETID'] if use_mark else [])
    rand_cols = base_cols

    real = read_fits_columns(str(data_path), real_cols)
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

    if args.random_subsample < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = rng.random(len(rand)) < args.random_subsample
        rand = rand[keep]
        print(f'---> random subsample fraction={args.random_subsample:.3f}, kept={len(rand)}')

    cosmo = FlatLambdaCDM(H0=args.h0, Om0=args.om0)
    chi_d = np.asarray(cosmo.comoving_distance(real['Z']).value, dtype=np.float64)
    chi_r = np.asarray(cosmo.comoving_distance(rand['Z']).value, dtype=np.float64)

    pos_d = radec_to_cartesian(real['RA'], real['DEC'], chi_d)
    pos_r = radec_to_cartesian(rand['RA'], rand['DEC'], chi_r)
    w_d = get_weight_column(real)
    w_r = get_weight_column(rand)
    targetid_d = np.asarray(real['TARGETID'], dtype=np.int64) if use_mark else None

    boxsize, center, origin = compute_box_geometry(pos_d, pos_r, boxsize_arg=args.boxsize,
                                                   box_padding=args.box_padding)
    pos_d = shift_and_clip_to_box(pos_d, origin=origin, boxsize=boxsize)
    pos_r = shift_and_clip_to_box(pos_r, origin=origin, boxsize=boxsize)
    boxcenter = np.array([0.5 * boxsize] * 3, dtype=np.float64)

    print(f'---> boxsize [Mpc/h]: {boxsize:.3f}')

    sw_d = float(np.sum(w_d, dtype=np.float64))
    sw_r = float(np.sum(w_r, dtype=np.float64))
    sw2_d = float(np.sum(w_d.astype(np.float64) ** 2))
    sw2_r = float(np.sum(w_r.astype(np.float64) ** 2))
    alpha = sw_d / sw_r

    resampler = mas_to_resampler(args.mas)
    edges = make_k_edges(boxsize=boxsize, nmesh=args.grid)

    print('---> computing P(k) with PyPower ...')
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
    volume = boxsize ** 3

    shot_noise_analytic = volume * sw2_d / (sw_d * sw_d)
    shot_noise_pypower = pk_unmarked['shotnoise']
    shot_noise = shot_noise_pypower if np.isfinite(shot_noise_pypower) else shot_noise_analytic

    if args.subtract_shotnoise:
        pk_used = pk_raw - shot_noise
    else:
        pk_used = pk_raw.copy()

    pk_window = np.full_like(pk_used, np.nan)
    shot_noise_window = np.nan
    shot_noise_window_pypower = np.nan
    if args.window_pk:
        print('---> computing P_window(k) from random field with PyPower ...')
        pk_window_result = compute_pk_pypower(data_positions=pos_r,
                                              data_weights=alpha * w_r,
                                              edges=edges,
                                              boxsize=boxsize,
                                              boxcenter=boxcenter,
                                              nmesh=args.grid,
                                              resampler=resampler,
                                              interlacing=args.interlacing)
        pk_window = align_to_k(k, pk_window_result['k'], pk_window_result['pk0'])
        shot_noise_window_analytic = volume * sw2_r / (sw_r * sw_r)
        shot_noise_window_pypower = pk_window_result['shotnoise']
        shot_noise_window = (shot_noise_window_pypower
                             if np.isfinite(shot_noise_window_pypower)
                             else shot_noise_window_analytic)

    ratio_pk_window = np.full_like(pk_used, np.nan)
    good = np.isfinite(pk_window) & (pk_window > 0.0)
    ratio_pk_window[good] = pk_used[good] / pk_window[good]

    pk_marked = {}
    pk_marked_used = {}
    ratio_marked_unmarked = {}
    ratio_marked_unmarked_masked = {}
    delta_marked_unmarked = {}
    shot_noise_marked = {}
    shot_noise_marked_pypower = {}
    mark_info = {}

    if use_mark:
        print(f'---> building ASTRA mark arrays for: {mark_cols}')
        mark_arrays, mark_info = build_mark_arrays(targetid_data=targetid_d,
                                                   w_data=w_d,
                                                   astra_path=args.astra_prob_file,
                                                   mark_cols=mark_cols,
                                                   mark_power=args.mark_power,
                                                   mark_eps=args.mark_eps,
                                                   unmatched_policy=args.unmatched_policy,
                                                   normalize=args.mark_normalize)

        ratio_kmask = k >= args.kmin_ratio
        if args.kmax_ratio > 0.0:
            ratio_kmask &= k <= args.kmax_ratio

        for mark_col in mark_cols:
            print(f'---> processing marked field for {mark_col} ...')
            mark = mark_arrays[mark_col]
            w_dm = w_d * mark
            sw_dm = float(np.sum(w_dm, dtype=np.float64))
            sw2_dm = float(np.sum(w_dm.astype(np.float64) ** 2))
            if sw_dm <= 0.0:
                raise RuntimeError(f'Marked total weight is <= 0 for {mark_col}. '
                                   'Check mark settings and unmatched policy.')

            mean_mark = sw_dm / sw_d
            mark_info[mark_col]['mean_mark_for_field'] = float(mean_mark)
            mark_info[mark_col]['field_normalization'] = ('PyPower FKP field with marked data weights: ')

            print(f'---> computing P_marked(k) with PyPower for {mark_col} ...')
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
            pk_marked[mark_col] = pk_marked_raw

            shot_noise_marked_analytic = volume * sw2_dm / (sw_dm * sw_dm)
            shot_noise_marked_pypower[mark_col] = pk_marked_result['shotnoise']
            shot_noise_marked[mark_col] = (shot_noise_marked_pypower[mark_col]
                                           if np.isfinite(shot_noise_marked_pypower[mark_col])
                                           else shot_noise_marked_analytic)

            if args.subtract_shotnoise:
                pk_marked_used[mark_col] = pk_marked[mark_col] - shot_noise_marked[mark_col]
            else:
                pk_marked_used[mark_col] = pk_marked[mark_col].copy()

            if args.ratio_source == 'raw':
                ratio_num = pk_marked[mark_col]
                ratio_den = pk_raw
            else:
                ratio_num = pk_marked_used[mark_col]
                ratio_den = pk_used

            ratio_arr = np.full_like(pk_used, np.nan)
            delta_arr = np.full_like(pk_used, np.nan)
            ratio_masked_arr = np.full_like(pk_used, np.nan)

            good_ratio = (np.isfinite(ratio_num)
                          & np.isfinite(ratio_den)
                          & (np.abs(ratio_den) > args.ratio_denom_min))
            ratio_arr[good_ratio] = ratio_num[good_ratio] / ratio_den[good_ratio]
            delta_arr[good_ratio] = ratio_num[good_ratio] - ratio_den[good_ratio]
            good_ratio_masked = good_ratio & ratio_kmask
            ratio_masked_arr[good_ratio_masked] = ratio_arr[good_ratio_masked]

            ratio_marked_unmarked[mark_col] = ratio_arr
            delta_marked_unmarked[mark_col] = delta_arr
            ratio_marked_unmarked_masked[mark_col] = ratio_masked_arr

    csv_path = outdir / f'pk_pypower_{tag}.csv'
    fig_path = outdir / f'pk_pypower_{tag}.png'
    fig_marked_pk_path = outdir / f'pk_marked_spectrum_{tag}.png'
    fig_ratio_path = outdir / f'pk_over_window_{tag}.png'
    fig_mark_ratio_path = outdir / f'pk_marked_over_unmarked_{tag}.png'
    fig_delta_mark_path = outdir / f'pk_marked_minus_unmarked_{tag}.png'
    meta_path = outdir / f'run_metadata_pk_pypower_{tag}.json'

    csv_cols = [k, pk_raw, pk_used, nmodes, pk_window, ratio_pk_window]
    csv_header_cols = ['k_h_mpc', 'pk_raw', 'pk_used', 'nmodes', 'pk_window', 'pk_over_window']
    if use_mark:
        for mark_col in mark_cols:
            suffix = mark_col.lower()
            csv_cols.extend([pk_marked[mark_col],
                             pk_marked_used[mark_col],
                             ratio_marked_unmarked[mark_col],
                             ratio_marked_unmarked_masked[mark_col],
                             delta_marked_unmarked[mark_col]])
            csv_header_cols.extend([f'pk_marked_raw_{suffix}',
                                    f'pk_marked_used_{suffix}',
                                    f'pk_marked_over_unmarked_{suffix}',
                                    f'pk_marked_over_unmarked_masked_{suffix}',
                                    f'delta_pk_marked_minus_unmarked_{suffix}'])

    np.savetxt(csv_path, np.column_stack(csv_cols), delimiter=',',
               header=','.join(csv_header_cols), comments='')

    mask = (k > 0.0) & np.isfinite(pk_used) & (pk_used > 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(k[mask], pk_used[mask], lw=1.5, label=r'$P_{\rm unmarked}(k)$',
              color=UNMARKED_COLOR)
    if use_mark:
        for mark_col in mark_cols:
            mmk = (k > 0.0) & np.isfinite(pk_marked_used[mark_col]) & (pk_marked_used[mark_col] > 0.0)
            ax.loglog(k[mmk], pk_marked_used[mark_col][mmk], lw=1.5,
                      label=fr'$P_{{\rm marked}}(k)$ {mark_col}',
                      color=MARK_COLOR_MAP.get(mark_col))
    if args.window_pk:
        mw = (k > 0.0) & np.isfinite(pk_window) & (pk_window > 0.0)
        ax.loglog(k[mw], pk_window[mw], lw=1.2, ls='--',
                  label=r'$P_{\rm window}(k)$', color=WINDOW_COLOR)
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    ax.grid(alpha=0.3, which='both')
    title = f'{args.tracer}  ${args.zmin:.2f} < z < {args.zmax:.2f}$'
    if args.subtract_shotnoise:
        title += '  (shot-noise subtracted)'
    ax.set_title(title)
    if args.window_pk or use_mark:
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=360)
    plt.close(fig)

    if args.window_pk:
        mr = (k > 0.0) & np.isfinite(ratio_pk_window) & (ratio_pk_window > 0.0)
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.semilogx(k[mr], ratio_pk_window[mr], lw=1.5, color=WINDOW_COLOR)
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$P(k)/P_{\rm window}(k)$')
        ax.grid(alpha=0.3, which='both')
        ax.set_title(f'BGS ${args.zmin:.2f} < z < {args.zmax:.2f}$')
        fig.tight_layout()
        fig.savefig(fig_ratio_path, dpi=360)
        plt.close(fig)

    if use_mark:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.grid(lw=0.2)
        ax.loglog(k[mask], pk_used[mask], lw=1.5,
                  label=r'$P_{\rm unmarked}(k)$', color=UNMARKED_COLOR)
        for mark_col in mark_cols:
            mmk = (k > 0.0) & np.isfinite(pk_marked_used[mark_col]) & (pk_marked_used[mark_col] > 0.0)
            ax.loglog(k[mmk], pk_marked_used[mark_col][mmk], lw=1.5,
                      label=fr'$P_{{\rm marked}}(k)$ {mark_col}',
                      color=MARK_COLOR_MAP.get(mark_col))
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
        ax.grid(alpha=0.3, which='both')
        ax.set_title(f'{args.tracer} marked spectra  ${args.zmin:.2f} < z < {args.zmax:.2f}$')
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_marked_pk_path, dpi=360)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for mark_col in mark_cols:
            mm = (k > 0.0) & np.isfinite(ratio_marked_unmarked_masked[mark_col]) & np.isfinite(k)
            ax.semilogx(k[mm], ratio_marked_unmarked_masked[mark_col][mm], lw=1.5,
                        color=MARK_COLOR_MAP.get(mark_col), label=mark_col)
        ax.axhline(1.0, ls='--', lw=1.0, color='black', alpha=0.7)
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$P_{\rm marked}(k) / P_{\rm unmarked}(k)$')
        ax.grid(alpha=0.3, which='both')
        k_range_txt = f'{args.kmin_ratio:.3f} < k'
        if args.kmax_ratio > 0.0:
            k_range_txt += f' < {args.kmax_ratio:.3f}'
        ax.set_title(f'{args.tracer} environments '
                     f'${args.zmin:.2f} < z < {args.zmax:.2f}$  $[{k_range_txt}]$')
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_mark_ratio_path, dpi=360)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for mark_col in mark_cols:
            mdm = (k > 0.0) & np.isfinite(delta_marked_unmarked[mark_col]) & np.isfinite(k)
            mdm &= k >= args.kmin_ratio
            if args.kmax_ratio > 0.0:
                mdm &= k <= args.kmax_ratio
            ax.semilogx(k[mdm], delta_marked_unmarked[mark_col][mdm], lw=1.5,
                        color=MARK_COLOR_MAP.get(mark_col), label=mark_col)
        ax.axhline(0.0, ls='--', lw=1.0, color='black', alpha=0.7)
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$\Delta P(k)=P_{\rm marked}-P_{\rm unmarked}$')
        ax.grid(alpha=0.3, which='both')
        ax.set_title(f'{args.tracer} environments  ${args.zmin:.2f} < z < {args.zmax:.2f}$  $[{k_range_txt}]$')
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_delta_mark_path, dpi=360)
        plt.close(fig)

    elapsed = time.time() - t0
    metadata = {'tracer': args.tracer,
                'zmin': args.zmin,
                'zmax': args.zmax,
                'base_dir': args.base_dir,
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
                'alpha': alpha,
                'grid': args.grid,
                'mas': args.mas,
                'resampler': resampler,
                'interlacing': args.interlacing,
                'boxsize_mpc_h': boxsize,
                'box_center_xyz_mpc_h': center.tolist(),
                'box_padding_mpc_h': args.box_padding,
                'volume_mpc3_h3': volume,
                'shot_noise_estimate': shot_noise,
                'shot_noise_estimate_analytic': shot_noise_analytic,
                'shot_noise_estimate_pypower': shot_noise_pypower,
                'shot_noise_window_estimate': shot_noise_window,
                'shot_noise_window_estimate_pypower': shot_noise_window_pypower,
                'shot_noise_marked_estimate': shot_noise_marked,
                'shot_noise_marked_estimate_pypower': shot_noise_marked_pypower,
                'subtract_shotnoise': args.subtract_shotnoise,
                'window_pk': args.window_pk,
                'ratio_source': args.ratio_source,
                'kmin_ratio_h_mpc': args.kmin_ratio,
                'kmax_ratio_h_mpc': args.kmax_ratio,
                'ratio_denom_min': args.ratio_denom_min,
                'mark_cols_used': mark_cols,
                'marking': mark_info,
                'nthreads_hint': args.nthreads,
                'random_subsample': args.random_subsample,
                'elapsed_sec': elapsed,
                'outputs': {'pk_csv': str(csv_path),
                            'pk_plot': str(fig_path),
                            'pk_marked_spectrum_plot': str(fig_marked_pk_path) if use_mark else None,
                            'pk_over_window_plot': str(fig_ratio_path) if args.window_pk else None,
                            'pk_marked_over_unmarked_plot': str(fig_mark_ratio_path) if use_mark else None,
                            'delta_pk_marked_minus_unmarked_plot': str(fig_delta_mark_path) if use_mark else None},
                'engine': 'pypower.CatalogFFTPower',
                'remove_shotnoise_flag_supported': bool(pk_unmarked['remove_shotnoise_supported']),
                'notes': ['Power spectra are computed with pypower using FFT-based catalog estimator.',
                          'P_window(k) and P/P_window are provided for relative analyses.']}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'---> wrote: {csv_path}')
    print(f'---> wrote: {fig_path}')
    if use_mark:
        print(f'---> wrote: {fig_marked_pk_path}')
    if args.window_pk:
        print(f'---> wrote: {fig_ratio_path}')
    if use_mark:
        print(f'---> wrote: {fig_mark_ratio_path}')
        print(f'---> wrote: {fig_delta_mark_path}')
    print(f'---> wrote: {meta_path}')
    print(f'---> elapsed: {elapsed:.2f} s')


if __name__ == '__main__':
    main()