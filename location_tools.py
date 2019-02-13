import os
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Proj
from coord_transform import bd2wgs, gcj2wgs
import mectools.hyper as hy

def ensure_wgs(lon, lat, proj):
    if proj == 'bd-09':
        return bd2wgs(lon, lat)
    elif proj == 'gcj':
        return gcj2wgs(lon, lat)
    elif proj == 'wgs':
        return lon, lat
    else:
        raise('Unknown projection')

def find_scene(lon, lat, index, best=True):
    if type(index) is str:
        index = pd.read_csv(index)
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
        return index

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

def load_scene(pid, chan='B8'):
    meta = parse_mtl(f'scenes/{pid}_MTL.txt')
    image = Image.open(f'scenes/{pid}_{chan}.TIF')
    return meta, image

# assumes WGS84 datum
def extract_tile(lon, lat, meta, image, rad=512):
    utm_zone = meta['UTM_ZONE']
    utm_hemi = 'north' if lat >= 0 else 'south'
    utm_proj = Proj(f'+proj=utm +zone={utm_zone}, +{utm_hemi} +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    x, y = utm_proj(lon, lat)
    fx = (x-meta['UTM_WEST'])/meta['UTM_WIDTH']
    fy = 1 - (y-meta['UTM_SOUTH'])/meta['UTM_HEIGHT']

    sx, sy = image.size
    px, py = int(fx*sx), int(fy*sy)
    box = (px-rad, py-rad, px+rad, py+rad)
    tile = image.crop(box)

    return tile

def extract_tile_once(lon, lat, index, rad, size=None, proj='bd-09', chan='B8', resample=Image.LANCZOS):
    lon, lat = ensure_wgs(lon, lat, proj)
    pid = find_scene(lon, lat, index)
    meta, image = load_scene(pid, chan=chan)
    tile = extract_tile(lon, lat, meta, image, rad=rad)
    if size is not None:
        tile = tile.resize((size, size), resample=resample)
    return tile

# find scenes corresponding. data is (tag, lon, lat) list
def index_firm_scenes(firms, fout, index, chan='B8', proj='bd-09'):
    if type(firms) is str:
        firms = pd.read_csv(firms)
    if type(index) is str:
        index = pd.read_csv(index)
    firms = [(tag, *ensure_wgs(lon, lat, proj=proj)) for tag, lon, lat in firms[['tag', 'lon', 'lat']].values]
    prods = [(tag, lon, lat, find_scene(lon, lat, index)) for tag, lon, lat in hy.progress(firms, per=100_000)]
    prods = pd.DataFrame(prods, columns=['tag', 'lon', 'lat', 'prod'])
    prods.to_csv(fout, index=False)

# scenes is a (tag, lon, lat, prod) file. assumes WGS84 datum
def extract_firm_tiles(prods, rad, size=256, resample=Image.LANCZOS, chan='B8', loc='tiles', ext='jpg'):
    if type(prods) is str:
        prods = pd.read_csv(fin)
    prods = prods.sort_values(by=['prod', 'tag'])
    pmap = prods.groupby('prod').groups
    print(len(pmap))

    for pid in pmap:
        print(pid)
        meta, image = load_scene(pid, chan=chan)
        for idx in pmap[pid]:
            tag, lon, lat = prods.loc[idx][['tag', 'lon', 'lat']]
            for r in rad:
                fname = f'{loc}/{tag}_r{r}_s{size}.{ext}'
                if not os.path.exists(fname):
                    tile = extract_tile(lon, lat, meta, image, rad=r)
                    tile = tile.resize((size, size), resample=resample)
                    tile.save(fname)

# indexing
# fname_scenes = 'targets/google_scenes_2002_summer.csv'
# fname_census = '../cluster/census/census_2004_geocode.csv'
# fname_target = 'targets/census_firms_2004.csv'
# cols = {
#     'No': 'tag',
#     'longitude': 'lon',
#     'latitude': 'lat'
# }
# census = pd.read_csv(fname_census, usecols=coldefs).rename(columns=coldefs).dropna()
# location_tools.index_firm_scenes(census, fout=fname_target, index=fname_scenes)
# location_tools.extract_firm_tiles(fname_target, [512, 1024])
