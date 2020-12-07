# train predictor

import os
import numpy as np
import pandas as pd
import PIL
import statsmodels.formula.api as smf
import sklearn.model_selection as sk
import mectools.data as dt
import tensorflow as tf
from tensorflow import keras

# args
seed = 2384923
samp = 0.01
BATCH_SIZE = 32
size = 1024

# random init
state = np.random.RandomState(seed)

# functions
def load_path(tag, base='tiles/density', ext='jpg'):
    tag = f'{tag:07d}'
    sub = tag[:4]
    return f'{base}/{size}px/{sub}/{tag}.{ext}'

# load in firm and location data
firms = pd.read_csv('firms/census_2004_geocode.csv', usecols=['id', 'industry', 'income', 'total_assets', 'employees'])
targ = pd.read_csv('targets/census_firms_2004.csv', usecols=['id', 'lat_wgs84', 'lon_wgs84'])
firms = pd.merge(firms, targ, on='id', how='left').dropna()

# downsample for now
firms = firms.sample(frac=samp)

# resolve image paths
firms['file'] = firms['id'].apply(load_path)
firms['fexist'] = firms['file'].apply(os.path.exists)
firms = firms[firms['fexist']]

# calculate outcome stats
firms['prod'] = firms['income']/firms['employees']
firms['lprod'] = dt.log(firms['prod'])
firms = firms.dropna(subset=['lprod'])

# calculate residual performance
reg_ind = smf.ols('lprod ~ 0 + C(industry)', data=firms).fit()
firms['lprod_resid'] = reg_ind.resid

# map into feature/label vocab
features = np.stack([np.array(PIL.Image.open(fn)) for fn in firms['file']])
labels = firms['lprod_resid'].values

# conform to shapes
features = features[:,:,:,None].astype(np.float32)/255 # single channel image
labels = labels.astype(np.float32)[:,None]

# do train/test split
X_train, X_valid, y_train, y_valid = sk.train_test_split(features, labels, test_size=0.2, random_state=seed)

# define keras model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=4, activation='relu'),
    keras.layers.Conv2D(filters=32, kernel_size=4, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=4),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(filters=64, kernel_size=4, activation='relu'),
    keras.layers.Conv2D(filters=64, kernel_size=4, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=4),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1)
])

# train keras model
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=100, validation_data=[X_valid, y_valid])

# evaluate model
yhat_valid = model.predict(X_valid)
res = pd.DataFrame({'y': y_valid[:,0], 'yhat': yhat_valid[:,0]})
res['err'] = res['yhat'] - res['y']
