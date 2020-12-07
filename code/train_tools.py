import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn.model_selection as sk
from scipy.interpolate import InterpolatedUnivariateSpline

##
## model fit evaluation
##

def interp_pandas(data, index=None, xmin=None, xmax=None, N=None):
    index0 = data.index
    if index is None:
        if xmin is None: xmin = index0.min()
        if xmax is None: xmax = index0.max()
        if N is None: N = 100
        index = np.linspace(index0.min(), index0.max(), N)
    def terp(x):
        spline = InterpolatedUnivariateSpline(index0, x)
        return pd.Series(spline(index), index=index)
    if type(data) is pd.Series:
        return terp(data)
    elif type(data) is pd.DataFrame:
        return pd.DataFrame({x: terp(data[x]) for x in data})

color = (0.32628988850442137, 0.6186236063052672, 0.802798923490965)
def plot_errors(y, s, data, ax, alpha=0.4, interp=True):
    if interp:
        data1 = interp_pandas(data[[y, s]])
    else:
        data1 = data.copy()
    indx, vals, stds = data1.index, data1[y], data1[s]
    lo1, hi1 = vals - stds, vals + stds
    lo2, hi2 = lo1 - stds, hi1 + stds
    ax.plot(indx, vals, color=color)
    ax.fill_between(indx, lo1, hi1, color=color, alpha=alpha)
    ax.fill_between(indx, lo2, hi2, color=color, alpha=0.5*alpha)

lnorm = mpl.colors.LogNorm()
def eval_model(y, yhat, N=10, axs=None, qmin=None, qmax=None, ymin=None, ymax=None):
    ax0, ax1 = axs
    if qmin is not None: ymin = np.quantile(y, qmin)
    if qmax is not None: ymax = np.quantile(y, qmax)

    res = pd.DataFrame({'y': y, 'yhat': yhat}).astype(np.float)
    ax0.hexbin(res['y'], res['yhat'], cmap=mpl.cm.Blues, norm=lnorm,
               gridsize=20, extent=(ymin, ymax, ymin, ymax))

    bins = np.linspace(ymin, ymax, N)
    res['ybin'] = np.digitize(res['y'], bins)
    res['ybin'] = np.minimum(N-1, res['ybin'])
    bmean = res.groupby('ybin')['yhat'].agg(mean=np.mean, var=np.std, size=len)
    bmean['std'] = bmean['var']/np.sqrt(bmean['size'])
    bmean = bmean.reindex(np.arange(N))
    bmean.index = bins
    plot_errors('mean', 'std', bmean, ax1)

    ax0.set_xlabel('True Productivity')
    ax0.set_ylabel('Predicted Productivity')
    ax0.set_title('Joint Distribution')
    ax1.set_xlabel('True Productivity')
    # ax1.set_ylabel('Predicted Productivity')
    ax1.set_title(f'Binned Results ({N})')

def predict_data(model, data, steps):
    it = iter(data)
    x_test, y_test = zip(*[next(it) for _ in range(steps)])
    yh_test = [model.predict(x) for x in x_test]
    x_test = [np.concat(x) for x in zip(*x_test)]
    y_test = np.concat(y_test).squeeze()
    yh_test = np.concat(yh_test).squeeze()
    return x_test, y_test, yh_test

# semi-balanced category split
def categ_split(data, cat, val_frac, state=None):
    cat_train, cat_valid = sk.train_test_split(data[cat].unique(), test_size=val_frac, random_state=state)
    cat_group = pd.concat([
        pd.DataFrame({cat: cat_train, 'group': 'train'}),
        pd.DataFrame({cat: cat_valid, 'group': 'valid'})
    ], axis=0).set_index(cat)
    data_group = data[[cat]].join(cat_group, on=cat)
    df_train = data[data_group['group']=='train']
    df_valid = data[data_group['group']=='valid']
    return df_train, df_valid
