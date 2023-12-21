import json
import os

import scipy.io
import natsort
import numpy as np

from utils.avg_meters import AverageMeter


def check_if_in_segments(time_point, segments):
    return any([segment['start'] <= time_point <= segment['end'] for segment in segments])


def get_filenames():
    # get patient filenames in correct order
    data1_path = r'D:\Study\asp\thesis\implementation\data\data1'
    data2_path = r'D:\Study\asp\thesis\implementation\data\data2'
    filenames = [filename for filename in natsort.os_sorted(os.listdir(data1_path))]
    filenames += [filename for filename in natsort.os_sorted(os.listdir(data2_path))]

    return filenames


def change_prediction(prediction):
    prediction[np.where(prediction == 1)[0]] = 0
    prediction[np.where(prediction == -1)[0]] = 1
    return prediction


def calculate_f1_score(tp, fp, fn, recall_weight=1, precision_weight=1):
    try:
        precision = tp / (tp + fp) * recall_weight  # PPV
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp / (tp + fn) * precision_weight  # TPR
    except ZeroDivisionError:
        recall = 0

    try:
        return 2 * recall * precision / (recall + precision), recall, precision
    except ZeroDivisionError:
        return 0, 0, 0


def get_score(y, prediction, acceptance_radius):
    if -1 in prediction:
        prediction = change_prediction(prediction)

    y_seiz_idx = np.array([
        [
            el + i if el + i > 0 else el
            for i in range(-acceptance_radius, acceptance_radius + 1)
        ]
        for el in np.where(y == 1)[0]
    ])
    pred_seiz_idx = np.where(prediction == 1)[0]

    tp = 0
    for row in y_seiz_idx:
        plus = False
        for idx in row:
            if idx in pred_seiz_idx:
                plus = True
                pred_seiz_idx = pred_seiz_idx[pred_seiz_idx != idx]
        if plus == True:
            tp += 1

    fp = len(pred_seiz_idx)
    fn = len(y_seiz_idx) - tp

    # idx = 1
    # while idx < len(pred_seiz_idx) - 1:
    #     if pred_seiz_idx[idx + 1] - pred_seiz_idx[idx - 1] == 2:
    #         fp -= 2
    #         idx += 3
    #     else:
    #         idx += 1

    idx = 1
    while idx < len(pred_seiz_idx) - 1:
        if pred_seiz_idx[idx + 1] - pred_seiz_idx[idx] == 1:
            fp -= 1
        idx += 1

    score, tpr, ppv = calculate_f1_score(tp, fp, fn)
    return score, tpr, ppv


def get_metrics_for_threshold(threshold_predictions_dir, dataset_info):
    subject_file_names = get_filenames()
    subject_keys_exclude = [
        'data1/dataset11',
        'data1/dataset13',
        'data1/dataset22',
        'data1/dataset27',
        'data2/004tl Anonim-20200926_213911-20211122_144051',
        'data2/004tl Anonim-20200929_081036-20211122_144552',
        'data2/006tl Anonim-20210208_063816-20211122_154113',
        'data2/009tl Anonim-20200213_130213-20211122_160907',
        'data2/009tl Anonim-20200215_021624-20211122_161231',
        'data2/015tl Anonim-20201116_134129-20211122_161958',
        'data2/017tl Anonim-20200708_143949-20211122_163253',
        'data2/018tl Anonim-20201212_101651-20211122_163821',
        'data2/020tl Anonim-20201218_194126-20211122_171755',
        'data2/026tl Anonim-20210302_093747-20211122_175031',
        'data2/037tl Anonim-20201102_102725-20211123_003801',
        'data2/039tl Anonim-20200607_035937-20211123_005921',
    ]
    # subject_keys_exclude = list()

    # threshold_str = os.path.basename(threshold_predictions_dir)
    # threshold = float('.' + threshold_str[1:])

    targets_dir = os.path.join(os.path.dirname(os.path.abspath(threshold_predictions_dir)), 'targets')

    positive_minutes_all_subjects = 0
    total_minutes_all_subjects = 0
    f1_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    precision_avg_meter = AverageMeter()
    subject_num = len(subject_file_names)
    for subject_idx in range(subject_num):
        subject_file_name = subject_file_names[subject_idx]
        file_name = f'{subject_idx + 1}.mat'
        prediction_path = os.path.join(threshold_predictions_dir, file_name)
        prediction_data_matlab = scipy.io.loadmat(prediction_path)

        target_path = os.path.join(targets_dir, file_name)
        target_data_matlab = scipy.io.loadmat(target_path)

        # print(subject_idx, subject_file_name, prediction_data_matlab['ind'].shape, target_data_matlab['st_arr'].shape)

        subject_key = f'{"data1" if subject_file_name.endswith(".dat") else "data2"}/{os.path.splitext(subject_file_name)[0]}'
        # subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        if subject_key in subject_keys_exclude:
            continue

        y = target_data_matlab['st_arr']
        preds_idxs = prediction_data_matlab['ind']
        preds = np.array([
            1 if idx in preds_idxs else 0
            for idx in range(1, len(y) + 1)
        ])

        subject_f1, subject_recall, subject_precision = get_score(y, preds, acceptance_radius=3)

        f1_avg_meter.update(subject_f1)
        recall_avg_meter.update(subject_recall)
        precision_avg_meter.update(subject_precision)

        positive_minutes_num = np.sum(preds)
        total_minutes_num = len(preds)
        positive_minutes_all_subjects += positive_minutes_num
        total_minutes_all_subjects += total_minutes_num

        # print(f'{subject_idx + 1:02} {subject_key:50} f1 = {subject_f1:.5f} precision = {subject_precision:.5f} recall = {subject_recall:.5f}')
        # print()
    # print()
    print(f'{threshold_str} f1 = {f1_avg_meter.avg:.5f} precision = {precision_avg_meter.avg:.5f} recall = {recall_avg_meter.avg:.5f} positives = {positive_minutes_all_subjects}/{total_minutes_all_subjects} ({(positive_minutes_all_subjects / total_minutes_all_subjects) * 100:.4f}%)')
    # exit()


if __name__ == '__main__':
    base_dir = 'D:\\Study\\asp\\thesis\\implementation\\data_orig\\SVM_outliers_new_data\\results'
    dataset_info_path = r'D:\Study\asp\thesis\implementation\data\dataset_info.json'

    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    for threshold_str in os.listdir(base_dir):
        if threshold_str == 'targets':
            continue

        # if threshold_str != '003000':
        #     continue

        threshold_dir = os.path.join(base_dir, threshold_str)
        get_metrics_for_threshold(threshold_dir, dataset_info)

        # break
