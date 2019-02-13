#!/bin/env python3

import os
import glob
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='Generate list of scenes matching certain criterion.')
parser.add_argument('--input', type=str, default='scenes', help='directory to intput from')
parser.add_argument('--output', type=str, default='thumbs', help='directory to output to')
parser.add_argument('--size', type=int, default=512, help='thumbnail size')
parser.add_argument('--format', type=str, default='png', help='image format for output')
parser.add_argument('--overwrite', action='store_true', help='whether to overwrite existing')
args = parser.parse_args()

Image.MAX_IMAGE_PIXELS = 1000000000
size = (args.size, args.size)

thumb_list = []
for fn in glob.glob(f'{args .input}/*.TIF'):
    dpath, fname = os.path.split(fn)
    fbase, _ = os.path.splitext(fname)
    tpath = os.path.join(args.output, f'{fbase}.{args.format}')
    if args.overwrite or not os.path.isfile(tpath):
        thumb_list.append((fn, tpath))

for spath, tpath in sorted(thumb_list, key=lambda x: x[0]):
    print(f'Converting: {spath} -> {tpath}')
    im = Image.open(spath)
    thumb = im.resize(size)
    thumb.save(tpath)
