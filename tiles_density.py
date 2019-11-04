import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image
from pyproj import Proj
from scipy.ndimage.filters import gaussian_filter
from coord_transform import bd2wgs, gcj2wgs, wgs2utm

# lest pillow complain
Image.MAX_IMAGE_PIXELS = 1000000000

# sigma: blur in meters
# norm: density units
def extract_density_core(mat, px, py, rad, size=256, sigma=2, norm=300, image=True):
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

def extract_density_tile(lon, lat, rad, density='density', size=256, sigma=25, norm=300, proj='bd-09', image=True):
    lon, lat = ensure_wgs(lon, lat, proj)
    utm_zone = wgs2utm(lon, lat)
    cells = pd.read_csv(f'{density}/utm_cells.csv', index_col='utm')
    pixel, N = cells.loc[utm_zone, 'pixel'], cells.loc[utm_zone, 'N']
    west, south, span = cells.loc[utm_zone, ['utm_west', 'utm_south', 'size']]

    proj = Proj(f'+proj=utm +zone={utm_zone}, +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    hist = pd.read_csv(f'{density}/density_utm{utm_zone}_{pixel}px.csv')
    mat = sp.csr_matrix((hist['density'], (hist['pix_north'], hist['pix_east'])), shape=(N, N))

    x, y = proj(lon, lat)
    fx = (x-west)/span
    fy = (y-south)/span
    px, py = int(fx*N), int(fy*N)

    psigma = sigma/pixel
    return extract_density_core(mat, px, py, rad, size=size, sigma=psigma, norm=norm, image=image)

# firms is a (id, lon, lat) filename or dataframe. assumes WGS84 datum
def extract_density_firm(firms, density, output, rad=[256, 1024], size=256, sigma=25, norm=300, overwrite=False, ext='jpg', log=True):
    if type(firms) is str:
        firms = pd.read_csv(firms)
        firms['id'] = firms['id'].astype(np.int)
    if type(rad) is int:
        rad = [rad]

    cells = pd.read_csv(f'{density}/utm_cells.csv', index_col='utm')

    firms = firms.sort_values(by=['utm_zone', 'id'])
    utm_grp = firms.groupby('utm_zone')
    utm_map = utm_grp.groups
    utm_len = utm_grp.size()
    if log: print(len(utm_map))

    for utm_zone in utm_map:
        if log: print(f'{utm_zone}: {utm_len[utm_zone]}')

        pixel, N = cells.loc[utm_zone, 'pixel'], cells.loc[utm_zone, 'N']
        west, south, span = cells.loc[utm_zone, ['utm_west', 'utm_south', 'size']]
        proj = Proj(f'+proj=utm +zone={utm_zone}, +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
        psigma = sigma/pixel

        hist = pd.read_csv(f'{density}/density_utm{utm_zone}_{pixel}px.csv')
        mat = sp.csr_matrix((hist['density'], (hist['pix_north'], hist['pix_east'])), shape=(N, N))

        for idx in utm_map[utm_zone]:
            tag, lon, lat = firms.loc[idx, ['id', 'lon_wgs84', 'lat_wgs84']]
            x, y = proj(lon, lat)
            fx, fy = (x-west)/span, (y-south)/span
            px, py = int(fx*N), int(fy*N)

            for r in rad:
                path = f'{output}/{r}px'
                fname = store_chunk(tag, path, ext=ext)
                if overwrite or not os.path.exists(fname):
                    im = extract_density_core(mat, px, py, r, size=size, sigma=psigma, norm=norm)
                    im.save(fname)

# parallel version
def extract_density_par(firms, rad, nproc=4, **kwargs):
    from multiprocessing import Pool
    def func(df):
        extract_density_firm(df, rad, **kwargs)
    chunks = np.array_split(firms, nproc)
    with Pool(nproc) as p:
        p.map(func, chunks)
