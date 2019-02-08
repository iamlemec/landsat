import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Proj
from coord_transform import bd2wgs, gcj2wgs

def find_scene(lon, lat, index, best=True):
    if type(index) is str:
        index = pd.read_csv(index)
    prods = index.query(f'NORTH_LAT >= {lat} and SOUTH_LAT <= {lat} and EAST_LON >= {lon} and WEST_LON <= {lon}')
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

def extract_tile(lon, lat, meta, image, rad=512, proj='bd-09'):
    if proj == 'bd-09':
        lon, lat = bd2wgs(lon, lat)
    elif proj == 'gcj':
        lon, lat = gcj2wgs(lon, lat)
    elif prof == 'wgs':
        pass
    else:
        raise('Unknown projection')

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

def extract_tile_once(lon, lat, index, chan='B8', rad=512, proj='bd-09'):
    pid = find_scene(lon, lat, index)
    meta, image = load_scene(pid, chan=chan)
    tile = extract_tile(lon, lat, meta, image, rad=rad, proj=proj)
    return tile

# data is a (fname, lon, lat) list
def extract_tile_batch(data, index, chan='B8', rad=512, proj='bd-09'):
    if type(index) is str:
        index = pd.read_csv(index)

    prods = [(fn, lon, lat, find_scene(lon, lat, index)) for fn, lon, lat in data]
    prods = pd.DataFrame(prods, columns=['fname', 'lon', 'lat', 'prod'])
    pmap = prods.groupby('prod').groups
    print(len(pmap))

    for pid in pmap:
        meta, image = load_scene(pid, chan=chan)
        for idx in pmap[pid]:
            fn, lon, lat = prods.loc[idx][['fname', 'lon', 'lat']]
            tile = extract_tile(lon, lat, meta, image, rad=rad, proj=proj)
            tile.save(fn)

