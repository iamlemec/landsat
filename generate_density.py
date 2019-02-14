# generate an atlas: a collection of locally flat maps, one for each relevant UTM zone

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image
from pyproj import Proj
from coord_transform import bd2wgs, wgs2utm
from mectools.hyper import progress

parser = argparse.ArgumentParser(description='Generate clusting and information')
parser.add_argument('--data', type=str, help='Path to firm data file')
parser.add_argument('--dense', type=str, default='density', help='Density directory')
parser.add_argument('--tile', type=str, default='tiles/cluster', help='Tile directory')
# parser.add_argument('--width', type=int, default=None, help='Gaussian kernel width')
# parser.add_argument('--weight', type=str, default=None, help='Weighting variable')
# parser.add_argument('--ind', type=int, default=None, help='Industry to select')
args = parser.parse_args()

# tools
def argify(f):
    def f1(x):
        y = f(*x)
        return list(y) if type(y) is tuple else y
    return f1

# utm gridding
size = 1e6 # kilometers
pixel = 150 # meters
N = int(np.round(size/pixel))
rad = 128 # pixel radius
ulon = np.linspace(-size/2, size/2, N+1)
ulat = np.linspace(-size/2, size/2, N+1)

# utm info
utm_cent = pd.read_csv('meta/utm_centers.csv', index_col='utm')
utm_cent['lon_west'] = utm_cent['lon'] - 3
utm_cent['lon_east'] = utm_cent['lon'] + 3
utm_cent['lat_north'] = utm_cent['lat'] + 4
utm_cent['lat_south'] = utm_cent['lat'] - 4
utm_cent['utm_west'] = utm_cent['east'] - size/2
utm_cent['utm_east'] = utm_cent['east'] + size/2
utm_cent['utm_north'] = utm_cent['north'] + size/2
utm_cent['utm_south'] = utm_cent['north'] - size/2
utm_cent['size'] = size
utm_cent['pixel'] = pixel
utm_cent['N'] = N

# constants
coldefs = {
    'No': 'id',
    'longitude': 'lon_bd09',
    'latitude': 'lat_bd09',
}

# load firm data
firms = pd.read_csv(args.data, usecols=coldefs).rename(columns=coldefs)
firms = firms.dropna().drop_duplicates('id')
firms['id'] = firms['id'].astype(np.int)

# find all relevant UTM zones and make converters
firms[['lon_wgs84', 'lat_wgs84']] = firms[['lon_bd09', 'lat_bd09']].apply(argify(bd2wgs), raw=True, result_type='expand', axis=1)
firms['utm_zone'] = firms[['lon_wgs84', 'lat_wgs84']].apply(argify(wgs2utm), axis=1)
utm_zones = sorted(firms['utm_zone'].unique())
utm_proj = {z: Proj(f'+proj=utm +zone={z}, +ellps=WGS84 +datum=WGS84 +units=m +no_defs') for z in utm_zones}
print(utm_zones)

# save utm cell info
utm_cent.loc[utm_zones].to_csv(f'{args.dense}/utm_cells.csv')

# group by utm zone and compute histograms
dense = {}
groups = firms.groupby('utm_zone').groups
for zone, idx in groups.items():
    this_cent = utm_cent.loc[zone]
    df = firms[
        (firms['lon_wgs84'] >= this_cent['lon_west'] - 1) &
        (firms['lon_wgs84'] <  this_cent['lon_east'] + 1) &
        (firms['lat_wgs84'] >= this_cent['lat_south'] - 1) &
        (firms['lat_wgs84'] <  this_cent['lat_north'] + 1)
    ].copy()

    this_proj = utm_proj[zone]
    df[['utm_east', 'utm_north']] = df[['lon_wgs84', 'lat_wgs84']].apply(argify(this_proj), raw=True, result_type='expand', axis=1)
    df['utm0_east'] = df['utm_east'] - this_cent.loc['east'] # recenter coordinates
    df['utm0_north'] = df['utm_north'] - this_cent.loc['north'] # recenter coordinates

    df['pix_east'] = np.digitize(df['utm0_east'], ulon)
    df['pix_north'] = np.digitize(df['utm0_north'], ulat)
    df1 = df[(df['pix_east']>0)&(df['pix_east']<N+1)&(df['pix_north']>0)&(df['pix_north']<N+1)]
    df1[['pix_east', 'pix_north']] -= 1
    print(zone, len(df), len(df1))

    hist = df1.groupby(['pix_east', 'pix_north']).size().rename('count').reset_index()
    hist['density'] = hist['count']/(pixel/1e3)**2 # firms per square kilometer
    hist.to_csv(f'{args.dense}/density_utm{zone}_{pixel}px.csv', index=False)
    dense[zone] = sp.csr_matrix((hist['density'], (hist['pix_north'], hist['pix_east'])), shape=(N, N))

# extract firm tiles
# for i, info in progress(firms.iterrows(), per=100_000):
#     fid = info['id']
#     zone = info['utm_zone']
#     hist = dense[zone]
#     proj = utm_proj[zone]
#     cent = utm_cent.loc[zone]
#
#     x, y = proj(info['lon_wgs84'], info['lat_wgs84'])
#     fx = (x-cent['utm_west'])/(cent['utm_east']-cent['utm_west'])
#     fy = (y-cent['utm_south'])/(cent['utm_north']-cent['utm_south'])
#     px, py = int(fx*N), int(fy*N)
#
#     tile = hist[px-rad:px+rad, py-rad:py+rad].toarray()
#     tile = tile/4 # to get most of the action in [0, 255]
#
#     # flat = tile.flatten()
#     # posi = flat[flat>0]
#     # if len(posi) > 0:
#     #     print(fid, np.mean(posi), np.max(posi), np.quantile(posi, [0.01, 0.99]))
#
#     tile = np.clip(tile, 0, 255).astype(np.uint8)
#     im = Image.fromarray(tile)
#     im.save(f'{args.tile}/{fid}.png')
