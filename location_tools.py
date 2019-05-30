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

# all purpose transform
def ensure_wgs(lon, lat, proj):
    if proj == 'bd-09':
        return bd2wgs(lon, lat)
    elif proj == 'gcj-02':
        return gcj2wgs(lon, lat)
    elif proj == 'wgs-84':
        return lon, lat
    else:
        raise('Unknown projection')

# tuple to args
def argify(f):
    def f1(x):
        y = f(*x)
        return list(y) if type(y) is tuple else y
    return f1

# to limit directory sizes
def store_chunk(tag, loc, ext='jpg'):
    tag = f'{tag:07d}'
    sub = tag[:4]
    psub = f'{loc}/{sub}'
    os.makedirs(psub, exist_ok=True)
    ptag = f'{psub}/{tag}.{ext}'
    return ptag

##
## scenes
##

# load scene index
def load_index(index):
    return pd.read_csv(index).dropna(subset=['PRODUCT_ID'])

# find scene from wgs84 coordinates
def find_scene(lon, lat, index, best=True):
    if type(index) is str:
        index = load_index(index)
    prods = index[
        (index['NORTH_LAT'] >= lat) &
        (index['SOUTH_LAT'] <= lat) &
        (index['EAST_LON' ] >= lon) &
        (index['WEST_LON' ] <= lon)
    ]
    if len(prods) == 0:
        return None
    if best:
        c_lon = 0.5*(prods['WEST_LON']+prods['EAST_LON'])
        c_lat = 0.5*(prods['NORTH_LAT']+prods['SOUTH_LAT'])
        dist = np.sqrt((c_lon-lon)**2+(c_lat-lat)**2)
        best = prods.loc[dist.idxmin()]
        return best['PRODUCT_ID']
    else:
        return prods

# parse scene metadata
def parse_mtl(fname):
    # hacky parse
    lines = [s.strip() for s in open(fname)]
    corners = [s for s in lines if s.startswith('CORNER')]
    utm = [s for s in lines if s.startswith('UTM')]
    fields = corners + utm
    vals = pd.Series(dict([s.split(' = ') for s in fields]))
    meta = vals.astype(np.float)

    # additional utm stats
    meta['UTM_WEST'] = 0.5*(meta['CORNER_LL_PROJECTION_X_PRODUCT']+meta['CORNER_UL_PROJECTION_X_PRODUCT'])
    meta['UTM_EAST'] = 0.5*(meta['CORNER_LR_PROJECTION_X_PRODUCT']+meta['CORNER_UR_PROJECTION_X_PRODUCT'])
    meta['UTM_NORTH'] = 0.5*(meta['CORNER_UL_PROJECTION_Y_PRODUCT']+meta['CORNER_UR_PROJECTION_Y_PRODUCT'])
    meta['UTM_SOUTH'] = 0.5*(meta['CORNER_LL_PROJECTION_Y_PRODUCT']+meta['CORNER_LR_PROJECTION_Y_PRODUCT'])
    meta['UTM_WIDTH'] = meta['UTM_EAST'] - meta['UTM_WEST']
    meta['UTM_HEIGHT'] = meta['UTM_NORTH']- meta['UTM_SOUTH']

    return meta

# load scene imagery and metadata
def load_scene(pid, chan='B8'):
    meta = parse_mtl(f'scenes/{pid}_MTL.txt')
    image = Image.open(f'scenes/{pid}_{chan}.TIF')
    return meta, image

# find scenes corresponding. data is (id, lon, lat) list
def index_firm_scenes(firms, fout, index):
    if type(firms) is str:
        firms = pd.read_csv(firms)[['id', 'lon_bd09', 'lat_bd09']].dropna()
    if type(index) is str:
        index = load_index(index)
    scene = lambda lon, lat: find_scene(lon, lat, index)
    firms[['lon_wgs84', 'lat_wgs84']] = firms[['lon_bd09', 'lat_bd09']].apply(argify(bd2wgs), raw=True, result_type='expand', axis=1)
    firms['utm_zone'] = firms[['lon_wgs84', 'lat_wgs84']].apply(argify(wgs2utm), raw=True, axis=1)
    firms['prod_id'] = firms[['lon_wgs84', 'lat_wgs84']].apply(argify(scene), raw=True, axis=1)
    firms.to_csv(fout, index=False)

##
## satellite
##

# assumes WGS84 datum
def extract_satelite_core(lon, lat, meta, image, rad=512):
    utm_zone = meta['UTM_ZONE']
    utm_hemi = 'north' if lat >= 0 else 'south'
    utm_proj = Proj(f'+proj=utm +zone={utm_zone}, +{utm_hemi} +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    x, y = utm_proj(lon, lat)
    fx = (x-meta['UTM_WEST'])/meta['UTM_WIDTH']
    fy = (y-meta['UTM_SOUTH'])/meta['UTM_HEIGHT']

    sx, sy = image.size
    px, py = int(fx*sx), int((1-fy)*sy) # image origin is top-left
    box = (px-rad, py-rad, px+rad, py+rad)
    im = image.crop(box)

    return im

# extract just one tile, for testing
def extract_satelite_tile(lon, lat, rad, pid=None, index=None, size=None, proj='bd-09', chan='B8', image=True, resample=Image.LANCZOS):
    lon, lat = ensure_wgs(lon, lat, proj)
    if pid is None:
        pid = find_scene(lon, lat, index)
    meta, image = load_scene(pid, chan=chan)
    im = extract_satelite_core(lon, lat, meta, image, rad=rad)
    im = im.resize((size, size), resample=resample)
    if image:
        return im
    else:
        return np.asarray(im)

# firms is a (tag, lon, lat, prod) file. assumes WGS84 datum
def extract_satelite_firm(firms, rad, size=256, resample=Image.LANCZOS, chan='B8', output='tiles/landsat', ext='jpg', overwrite=False):
    if type(firms) is str:
        firms = pd.read_csv(firms)
    if type(rad) is int:
        rad = [rad]

    firms = firms.sort_values(by=['prod_id', 'id'])
    pmap = firms.groupby('prod_id').groups
    print(len(pmap))

    for pid in pmap:
        print(pid)
        meta, image = load_scene(pid, chan=chan)
        for idx in pmap[pid]:
            tag, lon, lat = firms.loc[idx][['id', 'lon_wgs84', 'lat_wgs84']]
            for r in rad:
                path = f'{output}/{r}px'
                fname = store_chunk(tag, path, ext=ext)
                if overwrite or not os.path.exists(fname):
                    tile = extract_satelite_core(lon, lat, meta, image, rad=r)
                    tile = tile.resize((size, size), resample=resample)
                    tile.save(fname)

##
## density
##

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

def extract_density_tile(lon, lat, density='density', rad=128, size=256, sigma=25, norm=300, proj='bd-09', image=True):
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
def extract_density_firm(firms, rad, size=256, sigma=25, norm=300, overwrite=False, density='density', output='tiles/density', ext='jpg', log=True):
    if type(firms) is str:
        firms = pd.read_csv(firms)
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
