import pandas as pd
import numpy as np

def GetTrainAndTestAndSubmission():
    air_reserve = pd.read_csv("air_reserve.csv", names=['Air store id', 'Visit datetime', 'Reserve datetime', 'Reserve visitors']);
    print(air_reserve) # id rezerwacji, data planowanej rezerwacji, data zarezerwowania, liczba przewidywanych gości
    air_store_info = pd.read_csv("air_store_info.csv", names=['Air store id', 'Air genre name', 'Air area name', 'Latitude', 'Longitude']);
    print(air_store_info) # id rezerwacji, gatunek restauracji, nazwa restauracji, szerokość geograficzna, długość geograficzna
    air_visit_data = pd.read_csv("air_visit_data.csv", names=['Air store id', 'Visit date', 'Visitors']);
    print(air_visit_data) # id rezerwacji, data odbytej rezerwacji, liczba gości

    date_info = pd.read_csv("date_info.csv", names=['Calendar date' , 'Day of week', 'Holiday flag']);
    print(date_info) # data kalendarzowa, dzień tygodnia, czy to dzień wakacji w Japonii (1 - tak, 0 - nie)

    hpg_reserve = pd.read_csv("hpg_reserve.csv", names=['Hpg store id', 'Visit datetime', 'Reserve datetime', 'Reserve visitors']);
    print(hpg_reserve) # id restauracji, data planowanej rezerwacji, data zarezerwowania, liczba przewidywanych gości
    hpg_store_info = pd.read_csv("hpg_store_info.csv", names=['Hpg store id', 'Hpg genre name', 'Hpg area name', 'Latitude', 'Longitude']);
    print(hpg_store_info) # id restauracji, gatunek restauracji, nazwa restauracji, szerokość geograficzna, długość geograficzna
    sample_submission = pd.read_csv("sample_submission.csv", names=['Id', 'Visitors']);
    print(sample_submission) # konkatenacja Air store id i daty odbytej rezerwacji, liczba gości
    store_id_relation = pd.read_csv("store_id_relation.csv", names=['Hpg store id', 'Air store id']);
    print(store_id_relation) # id restauracji, id rezerwacji

    results = pd.DataFrame()