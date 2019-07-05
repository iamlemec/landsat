import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl

##
## model fit evaluation
##

color = (0.32628988850442137, 0.6186236063052672, 0.802798923490965)
def plot_errors(y, s, data, ax, alpha=0.4):
    indx, vals, stds = data.index, data[y], data[s]
    lo1, hi1 = vals - stds, vals + stds
    lo2, hi2 = lo1 - stds, hi1 + stds
    ax.plot(indx, vals, color=color)
    ax.fill_between(indx, lo1, hi1, color=color, alpha=alpha)
    ax.fill_between(indx, lo2, hi2, color=color, alpha=0.5*alpha)
    
def eval_model(y, yhat, ymin=-2, ymax=2, nbins=10, axs=None):
    ax0, ax1 = axs

    res = pd.DataFrame({'y': y, 'yhat': yhat}).astype(np.float)
    res['err'] = res['yhat'] - res['y']
    res1 = res.query(f'y > {ymin} and y < {ymax} and yhat > {ymin} and yhat < {ymax}')
    ax0.hexbin(res1['y'], res1['yhat'], cmap=mpl.cm.Blues, gridsize=20)

    bins = np.linspace(ymin, ymax, nbins)
    res['ybin'] = np.digitize(res['y'], bins)
    res['ybin'] = np.minimum(nbins-1, res['ybin'])
    # bmean = res.groupby('ybin')['yhat'].mean()
    bmean = res.groupby('ybin')['yhat'].agg({'mean': np.mean, 'var': np.std, 'size': len})
    bmean['std'] = bmean['var']/np.sqrt(bmean['size'])
    bmean = bmean.reindex(np.arange(nbins))
    bmean.index = bins
    # bmean.plot(ax=ax1)
    plot_errors('mean', 'std', bmean, ax1)

    ax0.set_xlabel('True Productivity')
    ax0.set_ylabel('Predicted Productivity')
    ax0.set_title('Joint Distribution')
    ax1.set_xlabel('True Productivity')
    # ax1.set_ylabel('Predicted Productivity')
    ax1.set_title(f'Binned Results ({nbins})')

def predict_data(model, data, steps):
    it = iter(data)
    x_test, y_test = zip(*[next(it) for _ in range(steps)])
    yh_test = [model.predict(x) for x in x_test]
    x_test = [np.concat(x) for x in zip(*x_test)]
    y_test = np.concat(y_test).squeeze()
    yh_test = np.concat(yh_test).squeeze()
    return x_test, y_test, yh_test

##
## image loading
##

def load_tile(fid, source, size, base='../tiles', ext='jpg'):
    tag = tf.strings.as_string(fid, width=7, fill='0')
    sub = tf.strings.substr(tag, 0, 4)
    res = tf.strings.join([tf.strings.as_string(size), 'px'])
    fn = tf.strings.join([tag, ext], '.')
    fp = tf.strings.join([base, source, res, sub, fn], '/')
    dat = tf.io.read_file(fp)
    img = tf.image.decode_jpeg(dat, channels=1)
    return img
