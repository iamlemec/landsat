# Landsat

Need to set up Google Cloud SDK and gsutil. Download full index from Google: gs://gcp-public-data-landsat/index.csv.gz (`meta/google_landsat_index.csv`)

`generate_targets.py`: filter master index to spatiotemporal range (`targets/google_scenes_2002_mincloud.csv`)
`location_tools.py:index_firm_scenes`: resolve firms to satelite tiles (`index/census2004_mincloud2002.csv`)
`generate_density.py`: construct sparse density files by UTM square (`density`)
`lcoation_tools.py:extract_*`: generate tiles from firm data (`tiles/*`)

More info on Google hosted Landsat data at: https://cloud.google.com/storage/docs/public-datasets/landsat

