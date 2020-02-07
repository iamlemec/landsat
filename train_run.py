#!/usr/bin/env python

import os
from glob import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import mectools.data as dt
import sklearn.model_selection as sk
import statsmodels.formula.api as smf

import train_data as data
import train_tools as tools
import train_models as models

from mectools.plotter import plotter
plt = plotter(backend='agg')

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# path = 'data/tiles_fast/asie2003'
# source = 'asie'
# year = 2003
def load_dataset(path, source, year, channels=['density'], val_frac=0.2, batch_size=128, buf_size=10_000, landsat='mincloud2002', ivar='id', yvar='log_tfp', split='rand'):
    if source == 'asie':
        firms = data.load_asie_firms(year, landsat)
    elif source == 'census':
        firms = data.load_census_firms(year, landsat)

    # random geographic split
    if split == 'geo':
        state = np.random.RandomState(21921351)
        df_train, df_valid = tools.categ_split(firms, 'city', val_frac, state=state)
        print(len(df_valid)/(len(firms)))
    else:
        df_train, df_valid = sk.train_test_split(firms, test_size=val_frac)

    def parse_function(fid, out):
        image = tf.concat([data.load_tile(fid, f'{path}/{ch}') for ch in channels], -1)
        return (fid, image), out

    def make_dataset(df):
        fids = tf.constant(df[ivar])
        labels = tf.reshape(tf.cast(tf.constant(df[yvar]), tf.float32), (-1, 1))
        data = tf.data.Dataset.from_tensor_slices((fids, labels))
        data = data.map(parse_function)
        data = data.shuffle(buffer_size=buf_size)
        data = data.batch(batch_size)
        data = data.repeat()
        return data

    return make_dataset(df_train), make_dataset(df_valid)

def train_model(train, valid, pix=256, nchan=1, epochs=5, steps_per_epoch=2000, validation_steps=100):
    model = models.gen_dual_medium(pix, nchan)
    history = model.fit(train, validation_data=valid, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    return model, history

def plot_variation(model, dataset, save=None):
    x_test, y_test, yh_test = tools.predict_data(model, dataset, 100)
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(11, 5))
    tools.eval_model(y_test, yh_test, N=10, axs=(ax0, ax1), qmin=0.02, qmax=0.98)
    if save is not None:
        fig.savefig(save)
    return fig
