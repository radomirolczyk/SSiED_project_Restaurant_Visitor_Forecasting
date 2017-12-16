import pandas as pd
import numpy as np


def GetTrainAndTestAndSubmission():
    air_reserve = pd.read_csv("air_reserve.csv"); # id restauracji_air, data planowanej rezerwacji, data zarezerwowania, liczba przewidywanych gości
    air_store_info = pd.read_csv("air_store_info.csv"); # id restauracji_air, gatunek restauracji, nazwa restauracji, szerokość geograficzna, długość geograficzna
    air_visit_data = pd.read_csv("air_visit_data.csv"); # id restauracji_air, data odbytej rezerwacji, liczba gości
    hpg_reserve = pd.read_csv("hpg_reserve.csv");  # id restauracji_hpg, data planowanej rezerwacji, data zarezerwowania, liczba przewidywanych gości
    hpg_store_info = pd.read_csv("hpg_store_info.csv");  # id restauracji_hpg, gatunek restauracji, nazwa restauracji, szerokość geograficzna, długość geograficzna

    store_id_relation = pd.read_csv("store_id_relation.csv"); # id restauracji_air, id restauracji_hpg

    date_info = pd.read_csv("date_info.csv"); # data kalendarzowa, dzień tygodnia, czy to dzień wakacji w Japonii (1 - tak, 0 - nie)

    sample_submission = pd.read_csv("sample_submission.csv"); # id_restauracji_air + data planowanej rezerwacji, liczba gości

    historical_results = pd.DataFrame()

    historical_results['air_store_id'] = air_visit_data.air_store_id
    historical_results['visit_date'] = air_visit_data.visit_date
    historical_results['visitors'] = air_visit_data.visitors

    print(sorted(store_id_relation.air_store_id))


    print((air_visit_data.air_store_id))

GetTrainAndTestAndSubmission()