#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == '':
    here = Path(__file__).resolve()
    src_root = here.parents[1]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))

from releases.yuan23 import discover_snapshot_map, _zone_sort_key  # noqa: E402


DEFAULT_BASE = '/pscratch/sd/n/ntbfin/emulator/hods/z0.8/yuan23_prior'


def parse_args():
    parser = argparse.ArgumentParser(description='List YUAN23 snapshot zones discovered by the pipeline release.')
    parser.add_argument('--base-dir', default=DEFAULT_BASE, help='Snapshot root directory.')
    parser.add_argument('--limit', type=int, default=None, help='Optional maximum number of zones to print.')
    parser.add_argument('--zones-only', action='store_true', help='Print only zone labels (one per line).')
    return parser.parse_args()


def main():
    args = parse_args()
    mapping = discover_snapshot_map(args.base_dir)
    zones = sorted(mapping.keys(), key=_zone_sort_key)
    if args.limit is not None and args.limit > 0:
        zones = zones[:args.limit]

    if args.zones_only:
        for zone in zones:
            print(zone)
        return

    print(f'base_dir={os.path.abspath(args.base_dir)}')
    print(f'total_snapshots={len(mapping)}')
    for zone in zones:
        print(f'{zone} -> {mapping[zone]}')


if __name__ == '__main__':
    main()

