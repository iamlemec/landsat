import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import tensorflow as tf
import mectools.data as dt
import schema

##
## firm loading
##

def load_asie_firms(year, landsat):
    # load in firm and location data
    cols = schema.asie[year]
    firms = pd.read_csv(f'../firms/asie{year}_geocode.csv', usecols=cols).rename(cols, axis=1)
    targ = pd.read_csv(f'../index/asie{year}_{landsat}.csv', usecols=['id', 'lat_wgs84', 'lon_wgs84', 'prod_id'])
    firms = pd.merge(firms, targ, on='id', how='left').dropna(subset=['id', 'sic4', 'prod_id', 'prefecture'])

    # regulate id
    firms['id'] = firms['id'].astype(np.int)
    firms['sic4'] = firms['sic4'].astype(np.int)
    firms = firms.drop_duplicates(subset='id').set_index('id', drop=False)

    # geographic location
    firms['prefecture'] = pd.to_numeric(firms['prefecture'], errors='coerce').astype(np.int)
    firms['city'] = firms['prefecture'].apply(lambda x: int(str(x)[:3]))

    # industry classify
    firms['sic2'] = firms['sic4'] // 100

    # calculate outcome stats
    firms['prod'] = firms['value_added']/firms['employees']

    # logify
    for c in ['value_added', 'assets_fixed', 'wages', 'prod']:
        firms[f'log_{c}'] = dt.log(firms[c])

    # filter out bad ones
    firms = firms.dropna(subset=['log_value_added', 'log_assets_fixed', 'log_wages'])

    # compute tfp as residual
    firms['log_tfp'] = smf.ols('log_value_added ~ log_assets_fixed + log_wages', data=firms).fit().resid
    firms['log_tfp_resid'] = smf.ols('log_tfp ~ C(sic2) + C(city)', data=firms).fit().resid
    firms['log_prod_resid'] = smf.ols('log_prod ~ C(sic2) + C(city)', data=firms).fit().resid

    return firms

def load_census_firms(year, landsat):
    # load in firm and location data
    cols = schema.census[year]
    firms = pd.read_csv(f'../firms/census{year}_geocode.csv', usecols=cols).rename(cols, axis=1)
    targ = pd.read_csv(f'../index/census{year}_{landsat}.csv', usecols=['id', 'lat_wgs84', 'lon_wgs84', 'prod_id'])
    firms = pd.merge(firms, targ, on='id', how='left').dropna(subset=['id', 'sic4', 'prod_id', 'location_code'])

    # regulate id
    firms['id'] = firms['id'].astype(np.int)
    firms['sic4'] = firms['sic4'].astype(np.int)
    firms = firms.drop_duplicates(subset='id').set_index('id', drop=False)

    # geographic location
    firms['prefecture'] = firms['location_code'].astype(np.int).apply(lambda x: int(str(x)[:4]))
    firms['city'] = firms['prefecture'].apply(lambda x: int(str(x)[:3]))

    # industry classify
    firms['sic2'] = firms['sic4'] // 100

    # calculate outcome stats
    firms['prod'] = firms['income']/firms['employees']

    # logify
    for c in ['income', 'prod']:
        firms[f'log_{c}'] = dt.log(firms[c])

    # filter out bad ones
    firms = firms.dropna(subset=['log_income', 'log_prod'])

    # compute tfp as residual
    firms['log_prod_resid'] = smf.ols('log_prod ~ C(sic2)', data=firms).fit().resid

    return firms
    
##
## image loading
##

def load_tile(fid, base, ext='jpg'):
    tag = tf.strings.as_string(fid, width=7, fill='0')
    sub = tf.strings.substr(tag, 0, 4)
    fn = tf.strings.join([tag, ext], '.')
    fp = tf.strings.join([base, sub, fn], '/')
    dat = tf.io.read_file(fp)
    img = tf.image.decode_jpeg(dat, channels=1)
    return img
