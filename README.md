## Geographic Clustering

Need to set up Google Cloud SDK and gsutil. More info on Google hosted Landsat data at: [https://cloud.google.com/storage/docs/public-datasets/landsat](https://cloud.google.com/storage/docs/public-datasets/landsat).

#### Scene selection

Download full index from Google: [gs://gcp-public-data-landsat/index.csv.gz](gs://gcp-public-data-landsat/index.csv.gz), save to `index/landsat/google_landsat_index.csv`.

Given the master index of scenes, need to filter by spatiotemporal range and select for optimal viewing conditions (due to cloud coverage). This is done using `filter_scenes.py`, which outputs to something like `data/scenes/google_scenes_2002_mincloud.csv`.

Fetch selected scenes with `fetch_scenes.py`. Save to `data/scenes` directory.

#### Firm resolution

Index which scene each firm is located in. Use `firm_scenes.py` and output to something like `index/firms/census2004_mincloud2002.csv`. This also performs the `BD-09` to `WGS-84` conversion and generates `UTM` values.

#### Density generation

Generate sparse firm density information partitioned by expanded, partially overlapping `UTM` squares. Run `firm_density.py` to store in `data/density`.

#### Extract firm tiles

Use `generate_tiles.py` to generate satellite and density tiles from firm data and store in `data/tiles/*`.

#### Train algorithm

Primary training code is in `notebooks/dataset.ipynb`.

### Notes

Some notes on technical matters.

#### Coordinates

- `WGS-84`: standard coordinate system used by GPS, developed by US DOD
- `GCJ-02`: coordinate system used by Chinese, location is fuzzed
- `BD-09`: updated coordinate system used by Baidu, more fuzzing
