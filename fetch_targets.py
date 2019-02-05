#!/bin/env python3

import os
import sys
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Generate list of scenes matching certain criterion.')
parser.add_argument('--targets', type=str, help='path to target list')
parser.add_argument('--output', type=str, default='scenes', help='directory to output to')
parser.add_argument('--wait', type=int, default=1, help='delay between file requests')
parser.add_argument('--dryrun', action='store_true', help='just print out commands to run')
parser.add_argument('--overwrite', action='store_true', help='always overwrite files')
args = parser.parse_args()

fname_fmt = '{prod}_{chan}.{ext}'
url_fmt = 'gs://gcp-public-data-landsat/LE07/01/{path:03d}/{row:03d}/{prod}/{prod}_{chan}.{ext}'

fetch_list = []
for i, targ in pd.read_csv(args.targets).dropna().iterrows():
    scene, prod = targ['SCENE_ID'], targ['PRODUCT_ID']
    path, row = targ['WRS_PATH'], targ['WRS_ROW']

    fname = fname_fmt.format(prod=prod, chan='B8', ext='TIF')
    fpath = os.path.join(args.output, fname)
    if args.overwrite or not os.path.isfile(fpath):
        url = url_fmt.format(prod=prod, path=path, row=row, chan='B8', ext='TIF')
        fetch_list.append((prod, fpath, url))

    fname = fname_fmt.format(prod=prod, chan='MTL', ext='txt')
    fpath = os.path.join(args.output, fname)
    if args.overwrite or not os.path.isfile(fpath):
        url = url_fmt.format(prod=prod, path=path, row=row, chan='MTL', ext='txt')
        fetch_list.append((prod, fpath, url))

for prod, fpath, url in sorted(fetch_list, key=lambda x: x[0]):
    print(f'Fetching {prod}: {url} -> {fpath}')
    cmd = f'gsutil cp {url} {fpath}'
    if args.dryrun:
        print(cmd)
    else:
        os.system(cmd)
        print()
        time.sleep(args.wait)

if args.dryrun:
    print(len(fetch_list))
