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
parser.add_argument('data', type=str, help='Path to firm data file')
parser.add_argument('dense', type=str, help='Density directory')
args = parser.parse_args()

# tools
def argify(f):
    def f1(x):
        y = f(*x)
        return list(y) if type(y) is tuple else y
    return f1

# utm gridding
size = 1e6 # kilometers
pixel = 15 # meters
N = int(np.round(size/pixel))
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

# load firm data
firms = pd.read_csv(args.data)
firms['id'] = firms['id'].astype(np.int)
firms['sic4'] = firms['sic4'].astype(np.int)
firms['sic2'] = firms['sic4'] // 100

# find all relevant UTM zones and make converters
utm_zones = sorted(firms['utm_zone'].unique())
utm_proj = {z: Proj(f'+proj=utm +zone={z}, +ellps=WGS84 +datum=WGS84 +units=m +no_defs') for z in utm_zones}
print(utm_zones)

# save utm cell info
utm_cent.loc[utm_zones].to_csv(f'{args.dense}/utm_cells.csv')

# group by utm zone and compute histograms
dense = {}
for zone, idx in firms.groupby('utm_zone').groups.items():
    this_cent = utm_cent.loc[zone]
    df = firms[
        (firms['lon_wgs84'] >= this_cent['lon_west'] - 2) &
        (firms['lon_wgs84'] <  this_cent['lon_east'] + 2) &
        (firms['lat_wgs84'] >= this_cent['lat_south'] - 2) &
        (firms['lat_wgs84'] <  this_cent['lat_north'] + 2)
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

    hist1 = df1.groupby(['pix_east', 'pix_north']).size().rename('count').reset_index()
    hist1['density'] = hist1['count']/(pixel/1e3)**2 # firms per square kilometer
    hist1.to_csv(f'{args.dense}/total_utm{zone}_{pixel}px.csv', index=False)

    hist2 = df1.groupby(['sic2', 'pix_east', 'pix_north']).size().rename('count').reset_index()
    hist2['density'] = hist2['count']/(pixel/1e3)**2 # firms per square kilometer
    hist2.to_csv(f'{args.dense}/industry_utm{zone}_{pixel}px.csv', index=False)
