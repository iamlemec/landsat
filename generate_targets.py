import argparse
import pandas as pd

# python3 generate_targets.py --index=meta/google_landsat_index.csv --output=targets/google_scenes_2002.csv --lat_min=15 --lat_max=55 --lon_min=70 --lon_max=135 --date_min="2002-01-01" --date_max="2002-12-31"
# python3 generate_targets.py --index=meta/google_landsat_index.csv --output=targets/google_scenes_2002_cloud.csv --lat_min=15 --lat_max=55 --lon_min=70 --lon_max=135 --date_min="2002-01-01" --date_max="2002-12-31" --cloud_max=20
# python3 generate_targets.py --index=meta/google_landsat_index.csv --output=targets/google_scenes_2007_summer.csv --lat_min=15 --lat_max=55 --lon_min=70 --lon_max=135 --date_min="2002-03-01" --date_max="2002-08-31" --cloud_max=20
# python3 generate_targets.py --index=meta/google_landsat_index.csv --output=targets/google_scenes_2002_mincloud.csv --lat_min=15 --lat_max=55 --lon_min=70 --lon_max=135 --date_min="2002-03-01" --date_max="2002-08-31"

parser = argparse.ArgumentParser(description='Generate list of scenes matching certain criterion.')
parser.add_argument('--index', type=str, help='path to full index')
parser.add_argument('--output', type=str, help='path to output to')
parser.add_argument('--lat_min', type=int, help='minimal WRS path to use')
parser.add_argument('--lat_max', type=int, help='maximal WRS path to use')
parser.add_argument('--lon_min', type=int, help='minimal WRS row to use')
parser.add_argument('--lon_max', type=int, help='maximal WRS row to use')
parser.add_argument('--date_min', type=str, help='minimal date to use')
parser.add_argument('--date_max', type=str, help='maximal date to use')
parser.add_argument('--cloud_max', type=int, default=100, help='maximal allowed cloud cover (out of 100)')
parser.add_argument('--spacecraft', type=str, default='LANDSAT_7', help='spacecraft to use')
args = parser.parse_args()

print('loading full index')
index = pd.read_csv(args.index).dropna(subset=['PRODUCT_ID'])

print('selecting on spacecraft')
index = index.query(f'SPACECRAFT_ID == "{args.spacecraft}"')

print('selecting on location')
index = index[
    (index['NORTH_LAT'] >= args.lat_min) &
    (index['SOUTH_LAT'] <= args.lat_max) &
    (index['EAST_LON' ] >= args.lon_min) &
    (index['WEST_LON' ] <= args.lon_max)
]

print('selecting on date')
index['DATE_ACQUIRED'] = pd.to_datetime(index['DATE_ACQUIRED'])
index = index.query(f'DATE_ACQUIRED >= "{args.date_min}" and DATE_ACQUIRED <= "{args.date_max}"')

print('selecting on cloud cover')
index = index.query(f'CLOUD_COVER >= 0 and CLOUD_COVER <= {args.cloud_max}')

print('finding most recent match')
index = index.loc[index.groupby(['WRS_PATH', 'WRS_ROW'])['CLOUD_COVER'].idxmin()]

print('saving to file')
index.to_csv(args.output, index=False)
