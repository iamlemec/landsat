#!/bin/env python3

import os
import sys
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Generate list of scenes matching certain criterion.')
parser.add_argument('--targets', type=str, help='path to target list')
parser.add_argument('--output', type=str, default='scenes', help='directory to output to')
args = parser.parse_args()

fname_fmt = '{prod}_B8.TIF'
url_fmt = 'gs://gcp-public-data-landsat/LE07/01/{path:03d}/{row:03d}/{prod}/{prod}_B8.TIF'
overwrite = False
wait = 60

fetch_list = []
for i, targ in pd.read_csv(args.targets).iterrows():
    scene, prod = targ['SCENE_ID'], targ['PRODUCT_ID']
    path, row = targ['WRS_PATH'], targ['WRS_ROW']

    fname = fname_fmt.format(prod=prod)
    fpath = os.path.join(args.output, fname)
    if not overwrite and os.path.isfile(fpath):
        continue

    url = url_fmt.format(prod=prod, path=path, row=row)
    fetch_list.append((prod, fpath, url))

for prod, fpath, url in sorted(fetch_list, key=lambda x: x[0]):
    print(f'Fetching {prod}: {url} -> {fpath}')
    os.system(f'gsutil cp {url} {fpath}')
    print()
    time.sleep(wait)

