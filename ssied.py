import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


def GetTrainAndTest():
    air_store_info = pd.read_csv(
        "air_store_info.csv");  # id restauracji_air, gatunek restauracji, nazwa restauracji, szerokość geograficzna, długość geograficzna
    air_visit_data = pd.read_csv(
        "air_visit_data.csv");  # id restauracji_air, data odbytej rezerwacji, liczba gości
    air_reserve = pd.read_csv(
        "air_reserve.csv"); # id restauracji_air, data planowanej rezerwacji, data zarezerwowania, liczba przewidywanych gości
    hpg_reserve = pd.read_csv(
        "hpg_reserve.csv");  # id restauracji_hpg, data planowanej rezerwacji, data zarezerwowania, liczba przewidywanych gości
    hpg_store_info = pd.read_csv(
        "hpg_store_info.csv");  # id restauracji_hpg, gatunek restauracji, nazwa restauracji, szerokość geograficzna, długość geograficzna
    store_id_relation = pd.read_csv(
        "store_id_relation.csv"); # id restauracji_air, id restauracji_hpg
    date_info = pd.read_csv(
        "date_info.csv"); # data kalendarzowa, dzień tygodnia, czy to dzień wakacji w Japonii (1 - tak, 0 - nie)
    sample_submission = pd.read_csv(
        "sample_submission.csv"); # id_restauracji_air + data planowanej rezerwacji, liczba gości

    # złączenie danych (historyczne rezerwacje + dane dotyczące restauracji)
    historical_results = pd.DataFrame()
    historical_results['air_store_id'] = air_visit_data.air_store_id
    historical_results['visit_date'] = air_visit_data.visit_date
    historical_results['visitors'] = air_visit_data.visitors
    train = pd.merge(left = historical_results,
                              right = air_store_info)
    train.drop(['latitude', 'longitude'], inplace=True, axis=1) # pozbycie się danych dotyczących odl. i szer. geograficznych

    pd.DataFrame(train).to_csv("train.csv", index=True) # zapis danych treningowych do pliku

    #print(len(set(store_id_relation.air_store_id) & set(air_visit_data.air_store_id))) $ wskazane nr air restauracji, które występują i w air i w hpg

    future_results = pd.DataFrame()
    future_results['air_store_id'] = air_reserve.air_store_id
    future_results['visit_date'] = air_reserve.visit_datetime
    future_results['reserve_visitors'] = air_reserve.reserve_visitors

    merged_future_results = pd.merge(left=future_results,
                              right=air_store_info)
    merged_future_results.drop(['latitude', 'longitude'], inplace=True, axis=1)

    pd.DataFrame(merged_future_results).to_csv("test.csv", index=True)  # zapis danych testowych do pliku

    # extraTreesClasiifier = ExtraTreesClassifier(n_estimators=100, max_features=27, criterion='entropy',
    #                                             min_samples_split=1,
    #                                             max_depth=50, min_samples_leaf=1, n_jobs=-1)
    # target = train
    # extraTreesClasiifier.fit(train, target)
    # x_pred = extraTreesClasiifier.predict_proba(train)
    # new_x_pred = x_pred[:, 1] * 1.721
    # new_x_pred = np.clip(new_x_pred, 1e-6, 1 - 1e-6)


GetTrainAndTest()