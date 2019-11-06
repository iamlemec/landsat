import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image
from pyproj import Proj
from scipy.ndimage.filters import gaussian_filter
from coord_transform import wgs2utm
from multiprocessing import Pool

# lest pillow complain
Image.MAX_IMAGE_PIXELS = 1000000000

# to limit directory sizes
def store_chunk(loc, tag, ext='jpg', overwrite=False):
    tag = f'{tag:07d}'
    sub = tag[:4]
    psub = f'{loc}/{sub}'
    os.makedirs(psub, exist_ok=True)
    ptag = f'{psub}/{tag}.{ext}'
    if overwrite or not os.path.exists(ptag):
        return ptag
    else:
        return None

# sigma: blur in meters
# norm: density units
def extract_density_mat(mat, px, py, rad=256, size=256, sigma=2, norm=300, image=True):
    # extract area
    den = mat[py-rad:py+rad, px-rad:px+rad].toarray()

    # gaussian blur
    if sigma is not None:
        den = gaussian_filter(den, sigma=sigma)

    # normalize and rectify
    if norm is not None:
        # norm = np.quantile(den[den>0], rect)
        den = den/(den+norm)

    if image:
        # quantize, pitch correct, and overly inspect
        im = Image.fromarray((255*den).astype(np.uint8), 'L')
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im = im.resize((size, size), resample=Image.LANCZOS)
        return im
    else:
        return den

def extract_density_coords(lon, lat, density, **kwargs):
    utm = wgs2utm(lon, lat)
    cells = pd.read_csv(f'{density}/utm_cells.csv', index_col='utm')
    pixel, N = cells.loc[utm, 'pixel'], cells.loc[utm, 'N']
    west, south, span = cells.loc[utm, ['utm_west', 'utm_south', 'size']]
    proj = Proj(f'+proj=utm +zone={utm}, +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    hist = pd.read_csv(f'{density}/density_utm{utm}_{pixel}px.csv')
    mat = sp.csr_matrix((hist['density'], (hist['pix_north'], hist['pix_east'])), shape=(N, N))

    x, y = proj(lon, lat)
    fx, fy = (x-west)/span, (y-south)/span
    px, py = int(fx*N), int(fy*N)

    return extract_density_mat(mat, px, py, **kwargs)

def extract_density_utm(utm, firms, density, output, overwrite=False, ext='jpg', **kwargs):
    cells = pd.read_csv(f'{density}/utm_cells.csv', index_col='utm')
    pixel, N = cells.loc[utm, 'pixel'], cells.loc[utm, 'N']
    west, south, span = cells.loc[utm, ['utm_west', 'utm_south', 'size']]
    proj = Proj(f'+proj=utm +zone={utm}, +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    hist = pd.read_csv(f'{density}/density_utm{utm}_{pixel}px.csv')
    mat = sp.csr_matrix((hist['density'], (hist['pix_north'], hist['pix_east'])), shape=(N, N))

    for _, tag, lon, lat in firms[['id', 'lon_wgs84', 'lat_wgs84']].itertuples():
        if (fname := store_chunk(output, tag, ext=ext, overwrite=overwrite)) is None:
            continue

        x, y = proj(lon, lat)
        fx, fy = (x-west)/span, (y-south)/span
        px, py = int(fx*N), int(fy*N)

        im = extract_density_mat(mat, px, py, **kwargs)
        im.save(fname)

if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool

    # parse input arguments
    parser = argparse.ArgumentParser(description='patent application parser')
    parser.add_argument('firms', type=str, help='firm data file')
    parser.add_argument('density', type=str, help='path to density directory')
    parser.add_argument('output', type=str, help='directory to output to')
    parser.add_argument('--sample', type=int, default=None, help='sample only N firms')
    parser.add_argument('--overwrite', action='store_true', help='clobber existing files')
    parser.add_argument('--threads', type=int, default=10, help='number of threads to use')
    parser.add_argument('--chunksize', type=int, default=1_000, help='chunksize to overlay')
    args = parser.parse_args()

    firms = pd.read_csv(args.firms, usecols=['id', 'utm_zone', 'lon_wgs84', 'lat_wgs84'])
    if args.sample is not None:
        firms = firms.sample(n=args.sample)

    firms = firms.sort_values(by=['utm_zone']).reset_index(drop=True)
    firms = firms.rename_axis('row', axis=0).reset_index()
    firms['row_group'] = firms['row'] // args.chunksize

    utm_grp = firms.groupby(['utm_zone', 'row_group'])
    utm_map = [(z, firms.loc[i]) for (z, g), i in utm_grp.groups.items()]
    print(len(utm_map))

    opts = {'overwrite': args.overwrite}
    def extract_func(z, f):
        extract_density_utm(z, f, args.density, args.output, **opts)

    with Pool(args.threads) as pool:
        pool.starmap(extract_func, utm_map, chunksize=1)
