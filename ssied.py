import numpy as np
import pandas as pd
from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing
from datetime import datetime
import glob, re
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

data = {
    'tra': pd.read_csv('air_visit_data.csv'),
    'as': pd.read_csv('air_store_info.csv'),
    'hs': pd.read_csv('hpg_store_info.csv'),
    'ar': pd.read_csv('air_reserve.csv'),
    'hr': pd.read_csv('hpg_reserve.csv'),
    'id': pd.read_csv('store_id_relation.csv'),
    'tes': pd.read_csv('sample_submission.csv'),
    'hol': pd.read_csv('date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[[
        'reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime': 'visit_date'})

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow':
    [i] * len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].min().rename(columns={'visitors': 'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].mean().rename(columns={'visitors': 'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].median().rename(columns={'visitors': 'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].max().rename(columns={'visitors': 'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].count().rename(columns={'visitors': 'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])

pd.DataFrame(train).to_csv("stary_train.csv", index = True)
pd.DataFrame(test).to_csv("stary_test.csv", index = True)

train.drop('air_store_id', inplace = True, axis = 1)
train.drop('visit_date', inplace = True, axis = 1)
target = train.visitors.values
train.drop('visitors', inplace = True, axis = 1)
train.drop('year', inplace = True, axis = 1)
train.drop('month', inplace = True, axis = 1)
train.drop('air_genre_name', inplace = True, axis = 1)
train.drop('air_area_name', inplace = True, axis = 1)
train.drop('latitude', inplace = True, axis = 1)
train.drop('longitude', inplace = True, axis = 1)
train.fillna(-1, inplace = True)
pd.DataFrame(train).to_csv("nowy_train.csv", index = True)

testids = test.id.values
test.drop(['id', 'visitors'], inplace = True, axis = 1)
test.drop('visit_date', inplace = True, axis = 1)
test.drop('air_store_id', inplace = True, axis = 1)
test.drop('year', inplace = True, axis = 1)
test.drop('month', inplace = True, axis = 1)
test.drop('air_genre_name', inplace = True, axis = 1)
test.drop('air_area_name', inplace = True, axis = 1)
test.drop('latitude', inplace = True, axis = 1)
test.drop('longitude', inplace = True, axis = 1)
test.fillna(1, inplace = True)


standardScaler = StandardScaler()
train[train.columns] = np.round(standardScaler.fit_transform(train), 4)
test[test.columns] = np.round(standardScaler.transform(test), 4)

extraTreesClasiifier = ExtraTreesClassifier(n_estimators = 10, max_features = 5, criterion = 'entropy', min_samples_split = 2,
                                            max_depth = 5, min_samples_leaf = 1, n_jobs = 1)
extraTreesClasiifier.fit(train, target)
x_pred = extraTreesClasiifier.predict_proba(train)
new_x_pred = x_pred[:,1]*1.721
new_x_pred = np.clip(new_x_pred, 1e-6, 1-1e-6)
print(log_loss(target, np.clip(x_pred[:,1]*1.721, 1e-6, 1-1e-6)))

y_pred = extraTreesClasiifier.predict_proba(test)

submission = pd.DataFrame({'Id': testids,
                           'Pred': np.clip(y_pred[:,1], 1e-7, 1-1e-7)})

submission.to_csv('submission.csv', index = True)