import numpy as np # biblioteka do operacji naukowych
import pandas as pd # bibliteka do zarządzania zbiorami danych

from sklearn import preprocessing # biblioteka do analizy danych i data-miningu
from sklearn.ensemble import ExtraTreesClassifier # las losowy
from sklearn.preprocessing import StandardScaler # standaryzacja

# odczyt danych z plików wraz z przypisaniem do zmiennych
data = {
    'tra': pd.read_csv('air_visit_data.csv'), # główne dane treningowe (plik zawiera dane historyczne)
    'as': pd.read_csv('air_store_info.csv'),
    'hs': pd.read_csv('hpg_store_info.csv'),
    'ar': pd.read_csv('air_reserve.csv'),
    'hr': pd.read_csv('hpg_reserve.csv'),
    'id': pd.read_csv('store_id_relation.csv'),
    'tes': pd.read_csv('sample_submission.csv'), # wzór danych testowych
    'hol': pd.read_csv('date_info.csv').rename(columns={'calendar_date': 'visit_date'}) # dane dotyczące dat (zmiana nazwy pierwszej kolumny)
}

# połączenie danych po id (hpg_store_id) z plików 'hpg_reserve' oraz 'store_id_relation' i aktualizacja danych w 'hpg_reserve'
data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

# pętla po danych w plikach 'air_reserve' i 'hpg_reserve'; przygotowanie dat do zmapowania
for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime']) # przykładowa data '2016-01-01 19:00:00'
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date # wycięcie godziny; przykładowa data '2016-01-01'
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1) # ilość dni od rezerwacji do wizyty
    data[df] = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[[ # grupowanie
        'reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime': 'visit_date'})

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
# 0 - poniedziałek
# 1 - wtorek
# 2 - środa
# 3 - czwartek
# 4 - piątek
# 5 - sobota
# 6 - niedziela
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek # zmiana daty na dzień tygodnia (ExtraTreesClassifier nie przyjmuje typu 'string' jako parametr)
# dalsze tymczasowe 'rozbicie' daty
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

# przykładowe id: air_00a91d42b08b08d9_2017-04-23
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2]) # odłączenie daty od właściwego id
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2])) # zapis id do zbioru testowego
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date']) # zapis daty do zbioru testowego
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

# operacje zachowania niezduplikowanych id
unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow':
    [i] * len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

# utworzenie kolumn z podstawowymi operacjami na wartościach w zbiorze treningowym
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].min().rename(columns={'visitors': 'min_visitors'}) # minimalna ilość gości
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow']) # złączenie
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].mean().rename(columns={'visitors': 'mean_visitors'}) # średnia ilość gości
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].median().rename(columns={'visitors': 'median_visitors'}) # wartość mediany ilości gości
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].max().rename(columns={'visitors': 'max_visitors'}) # maksymalna ilość gości
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
    'visitors'].count().rename(columns={'visitors': 'count_observations'}) # zliczenie
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

# testowo
stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

# utworzenie właściwych zbiorów
train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
pd.DataFrame(train).to_csv("old_train.csv", index = False)
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])
pd.DataFrame(test).to_csv("old_test.csv", index = False)

# kolumna docelowa 'visitors'
target = train.visitors.values

# przygotowanie danych w zbiorze do wytrenowania
train.drop('visitors', inplace = True, axis = 1)
train.drop('air_store_id', inplace = True, axis = 1)
train.drop('visit_date', inplace = True, axis = 1)
train.drop('year', inplace = True, axis = 1)
train.drop('month', inplace = True, axis = 1)
train.drop('air_genre_name', inplace = True, axis = 1)
train.drop('air_area_name', inplace = True, axis = 1)
train.drop('latitude', inplace = True, axis = 1)
train.drop('longitude', inplace = True, axis = 1)
# uzupełnienie pustych miejsc wartoścami -1
train.fillna(-1, inplace = True)
pd.DataFrame(train).to_csv("new_train.csv", index = True)

# 'wyczyszenie' zbioru testowego przed przewidywaniem
testids = test.id.values
test.drop(['id', 'visitors'], inplace = True, axis = 1)

# przygotowanie danych w zbiorze do przewidywania
test.drop('air_store_id', inplace = True, axis = 1)
test.drop('visit_date', inplace = True, axis = 1)
test.drop('year', inplace = True, axis = 1)
test.drop('month', inplace = True, axis = 1)
test.drop('air_genre_name', inplace = True, axis = 1)
test.drop('air_area_name', inplace = True, axis = 1)
test.drop('latitude', inplace = True, axis = 1)
test.drop('longitude', inplace = True, axis = 1)
test.fillna(1, inplace = True)
pd.DataFrame(test).to_csv("new_test.csv", index = True)

standardScaler = StandardScaler()
# zaokrąglenie danych do 4 miejsc po przecinku z dopasowanej tablicy
# fit_transform - obliczanie wartości oczekiwanej oraz obliczanie standaryzacji i dopasowanie do zbioru
train[train.columns] = np.round(standardScaler.fit_transform(train), 4)
# transform - dokonanie standaryzacji
test[test.columns] = np.round(standardScaler.transform(test), 4)

print('Start training...')

# tworzymy drzewo decyzyjne
# n_estimators - liczba drzew w lesie
# max_features - liczba kolumn w plikach ze zbiorami
# criterion = 'entropy' - analiza ekspolaracyjna (zamiennie może być stosowane z 'Gini'); sa to indeksy drzewa decyzyjnego
# min_samples_split - minimalna liczba wymagana do rozdzielenia wezla (domyślnie 2, jest to minimalna poprawna wartość parametru)
# max_depth - maksymalna głębokość drzewa (domyślnie None)
# min_samples_leaf - minimalna liczba wymagana do liścia (domyślnie 1)
# n_jobs - ile wątków (jeżeli -1 to liczba rdzeni)
extraTreesClasiifier = ExtraTreesClassifier(n_estimators = 20, max_features = 5, criterion = 'entropy', min_samples_split = 2,
                                            max_depth = 10, min_samples_leaf = 1, n_jobs = 1)
# dopasowanie drzewa
extraTreesClasiifier.fit(train, target)
# przwidywanie dla zbioru treningowego
x_pred = extraTreesClasiifier.predict(train)
# np.clip - ogranicza tablice do a_min i a_max (założenie od 1 do 100 gości)
new_x_pred = np.clip(x_pred, 1, 100)

print('Start prediction...')

# przwidywanie dla zestawu testowego
y_pred = extraTreesClasiifier.predict(test)
submission = pd.DataFrame({'id': testids,
                           'visitors': np.clip(y_pred, 1, 100)})

submission.to_csv('submission.csv', index = False)

print('Completed!')