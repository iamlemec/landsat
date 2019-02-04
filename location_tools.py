import pandas as pd

targets0 = 'targets/google_scenes_2002.csv'

def find_scene(lon, lat, targets=targets0):
    index = pd.read_csv(targets)
    index = index.query(f'NORTH_LAT >= {lat} and SOUTH_LAT <= {lat} and EAST_LON >= {lon} and WEST_LON <= {lon}')
    return index

