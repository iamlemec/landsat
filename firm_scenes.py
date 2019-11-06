import argparse
import numpy as np
import pandas as pd
from coord_transform import bd2wgs, wgs2utm

# python3 firm_scenes.py data/firms/census2004_geocode.csv index/firms/census2004_mincloud2002.csv --index index/scenes/google_scenes_2002_mincloud.csv

parser = argparse.ArgumentParser(description='Index which scene each firm belongs to.')
parser.add_argument('firms', type=str, help='path to firm location data')
parser.add_argument('output', type=str, help='path to output file')
parser.add_argument('--index', type=str, help='scene index to use')
args = parser.parse_args()

# load firm data
firms = pd.read_csv(args.firms, usecols=['id', 'lon_bd09', 'lat_bd09'])
firms = firms[(firms['id'] % 1) == 0]
firms['id'] = firms['id'].astype('Int64')
firms = firms.dropna()

# load index data
index = pd.read_csv(args.index, usecols=['PRODUCT_ID', 'NORTH_LAT', 'SOUTH_LAT', 'EAST_LON', 'WEST_LON']).dropna()
index = index.rename(columns={'PRODUCT_ID': 'prod_id'})

# find scene centers
index['cent_lon'] = 0.5*(index['WEST_LON']+index['EAST_LON'])
index['cent_lat'] = 0.5*(index['NORTH_LAT']+index['SOUTH_LAT'])

# analytic transformations
firms[['lon_wgs84', 'lat_wgs84']] = firms[['lon_bd09', 'lat_bd09']].apply(lambda lonlat: list(bd2wgs(*lonlat)), raw=True, result_type='expand', axis=1)
firms['utm_zone'] = firms[['lon_wgs84', 'lat_wgs84']].apply(lambda lonlat: wgs2utm(*lonlat), raw=True, axis=1)

# find scenes containing firms
match_firm, match_scene = np.nonzero(
      (index['NORTH_LAT'][None,:] >= firms['lat_wgs84'][:,None])
    & (index['SOUTH_LAT'][None,:] <= firms['lat_wgs84'][:,None])
    & (index['EAST_LON' ][None,:] >= firms['lon_wgs84'][:,None])
    & (index['WEST_LON' ][None,:] <= firms['lon_wgs84'][:,None])
)
match = pd.DataFrame({
    'id': firms['id'].iloc[match_firm].values,
    'prod_id': index['prod_id'].iloc[match_scene].values
})

# find best match scenes
match = match.merge(firms[['id', 'lon_wgs84', 'lat_wgs84']], on='id')
match = match.merge(index[['prod_id', 'cent_lon', 'cent_lat']], on='prod_id')
match['dist'] = np.sqrt((match['cent_lon']-match['lon_wgs84'])**2+(match['cent_lat']-match['lat_wgs84'])**2)
best = match.groupby('id')['dist'].idxmax()
prods = match[['id', 'prod_id']].loc[best]
firms = firms.merge(prods, on='id', how='left')

# save results
firms.to_csv(args.output, index=False)
