import numpy as np
import pandas as pd


xlsx_targets = 'D:\\Study\\asp\\thesis\\implementation\\data\\dataset_info_v2.xlsx'
data1_df = pd.read_excel(xlsx_targets, sheet_name='dataset1', engine='openpyxl')
data2_df = pd.read_excel(xlsx_targets, sheet_name='dataset2', engine='openpyxl')


def get_seizures(dataset_id, subject_id):
    assert dataset_id in [1, 2]
    assert subject_id > 0

    seizures = list()

    df = None
    if dataset_id == 1:
        global data1_df
        df = data1_df
    elif dataset_id == 2:
        global data2_df
        df = data2_df
    else:
        raise NotImplementedError

    seizures_raw = list(df.iloc[subject_id][1:])
    seizures_raw = [seizure_time for seizure_time in seizures_raw if not np.isnan(seizure_time)]
    for i in range(len(seizures_raw) // 2):
        seizures.append(
            {
                'start': seizures_raw[i * 2],
                'end': seizures_raw[i * 2 + 1],
            }
        )

    return seizures


def get_channels_info():
    sheet_names = ['ch27', 'ch28', 'ch29']

    channels_to_drop = {
        'ch27': ['ECG', 'MKR+ MKR-'],
        'ch28': ['ECG', 'MKR+ MKR-', 'Fpz'],
        'ch29': ['ECG', 'MKR+ MKR-', 'Fpz', 'EMG'],
    }
    channels_info = {
        int(sheet_name[2:]): {
            'order': list(),
            'drop': channels_to_drop[sheet_name]
        }
        for sheet_name in sheet_names
    }

    for sheet_name in sheet_names:
        df = pd.read_excel(xlsx_targets, sheet_name=sheet_name, engine='openpyxl', header=None)
        order = list(df[1])
        channels_info[int(sheet_name[2:])]['order'] = order

    return channels_info
