import os
import pickle

import numpy as np

from SVM_outliers_new_data_predictions_imitation import get_filenames
from utils.avg_meters import AverageMeter


def trim_preds(prediction_data_ocsvm, prediction_data_nn):
    min_length = 1e9
    for key in prediction_data_ocsvm:
        if not isinstance(prediction_data_ocsvm[key], np.ndarray):
            continue
        min_length = min(min_length, len(prediction_data_ocsvm[key]), len(prediction_data_nn[key]))

    prediction_data_ocsvm_trimmed = prediction_data_ocsvm.copy()
    prediction_data_nn_trimmed = prediction_data_nn.copy()
    for key in prediction_data_ocsvm_trimmed:
        if not isinstance(prediction_data_ocsvm_trimmed[key], np.ndarray):
            continue
        prediction_data_ocsvm_trimmed[key] = prediction_data_ocsvm_trimmed[key][:min_length]
        prediction_data_nn_trimmed[key] = prediction_data_nn_trimmed[key][:min_length]

    return prediction_data_ocsvm_trimmed, prediction_data_nn_trimmed


if __name__ == '__main__':
    threshold_nn = 0.95
    exclude_16 = True
    prediction_dir_ocsvm_path = r'D:\Study\asp\thesis\implementation\experiments\SVM_outliers_new_data_000250\predictions_positive_only'
    # prediction_dir_nn_path = r'D:\Study\asp\thesis\implementation\experiments\renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug\predictions'
    prediction_dir_nn_path = r'D:\Study\asp\thesis\implementation\experiments\20231107_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NoFilt\predictions'

    import data_split
    subject_keys_exclude = data_split.get_subject_keys_exclude_16()

    negative_fail_percent_avg = AverageMeter()
    filenames = get_filenames()
    for subject_idx, subject_filename in enumerate(filenames):
        subject_name, subject_ext = os.path.splitext(subject_filename)
        subset_name = 'data1' if subject_filename.endswith('.dat') else 'data2'
        subject_key = f'{subset_name}/{subject_name}'

        if exclude_16 and subject_key in subject_keys_exclude:
            continue

        prediction_path_ocsvm = os.path.join(prediction_dir_ocsvm_path, subset_name, f'{subject_name}.pickle')
        prediction_path_nn = os.path.join(prediction_dir_nn_path, subset_name, f'{subject_name}.pickle')
        # print(os.path.exists(prediction_path_ocsvm), prediction_path_ocsvm)
        # print(os.path.exists(prediction_path_nn), prediction_path_nn)

        if not os.path.exists(prediction_path_ocsvm):
            print(f'ocsvm does not have preds for {subject_key}')
            continue

        if not os.path.exists(prediction_path_nn):
            print(f'nn does not have preds for {subject_key}')
            continue

        prediction_data_ocsvm = pickle.load(open(prediction_path_ocsvm, 'rb'))
        prediction_data_nn = pickle.load(open(prediction_path_nn, 'rb'))

        # print(len(prediction_data_ocsvm['probs']), len(prediction_data_nn['probs']))
        prediction_data_ocsvm, prediction_data_nn = trim_preds(prediction_data_ocsvm, prediction_data_nn)
        # print(len(prediction_data_ocsvm['probs']), len(prediction_data_nn['probs']))

        preds_ocsvm = prediction_data_ocsvm['probs_wo_tta']
        preds_nn = prediction_data_nn['probs_wo_tta'] > threshold_nn

        labels = prediction_data_nn['labels']
        negative_preds_ocsvm_mask = (preds_ocsvm == 0)

        picked_preds_nn = preds_nn[negative_preds_ocsvm_mask]
        negative_fail_percent = 100 * picked_preds_nn.sum() / picked_preds_nn.size
        negative_fail_percent_avg.update(negative_fail_percent)

        print(f'Progress {subject_idx + 1:02}/{len(filenames):02} {subject_key:50} negative_fail_percent = {negative_fail_percent:5.2f}% avg = {negative_fail_percent_avg.avg:5.2f}%')

        # if subject_idx > 3:
        #     break