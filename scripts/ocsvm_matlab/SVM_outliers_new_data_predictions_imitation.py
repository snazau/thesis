import json
import os
import pickle

import scipy.io
import mat73
import natsort
import numpy as np
import scipy.ndimage


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def check_if_in_segments(time_point, segments):
    return any([segment['start'] <= time_point <= segment['end'] for segment in segments])


def get_filenames():
    # get patient filenames in correct order
    data1_path = r'D:\Study\asp\thesis\implementation\data\data1'
    data2_path = r'D:\Study\asp\thesis\implementation\data\data2'
    filenames = [filename for filename in natsort.os_sorted(os.listdir(data1_path))]
    filenames += [filename for filename in natsort.os_sorted(os.listdir(data2_path))]

    return filenames


def get_metrics_for_threshold(threshold_predictions_dir, dilation_times, sfreq=128, save_json_targets=False):
    dataset_info_path = 'D:/Study/asp/thesis/implementation/data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    subject_file_names = get_filenames()

    prediction_data_dir_name = f'npy_preds_trimmed_dilation={dilation_times}' + ('' if not save_json_targets else '_v2_targets')
    prediction_data_dir = os.path.join(threshold_predictions_dir, prediction_data_dir_name)
    os.makedirs(prediction_data_dir, exist_ok=True)
    os.makedirs(os.path.join(prediction_data_dir, 'data1'), exist_ok=True)
    os.makedirs(os.path.join(prediction_data_dir, 'data2'), exist_ok=True)

    subject_num = len(subject_file_names)
    targets_dir = os.path.join(os.path.dirname(os.path.abspath(threshold_predictions_dir)), 'targets')

    positive_minutes_all_subjects = 0
    total_minutes_all_subjects = 0
    for subject_idx in range(subject_num):
        subject_file_name = subject_file_names[subject_idx]
        subject_key = f'{"data1" if subject_file_name.endswith(".dat") else "data2"}/{os.path.splitext(subject_file_name)[0]}'
        print(subject_key)

        # if subject_key != 'data2/038tl Anonim-20190822_155119-20211123_005457':
        # if subject_key != 'data2/038tl Anonim-20190822_131550-20211123_005257':
        # if subject_key != 'data2/040tl Anonim-20200421_100248-20211123_010147':
        # if subject_key != 'data2/035tl Anonim-20210324_151211-20211122_222545':
        #     continue

        try:
            file_name = f'{subject_idx + 1}.mat'
            prediction_path = os.path.join(threshold_predictions_dir, file_name)
            prediction_data_matlab = scipy.io.loadmat(prediction_path)

            target_path = os.path.join(targets_dir, file_name)
            target_data_matlab = scipy.io.loadmat(target_path)

            # print(subject_idx, subject_file_name, prediction_data_matlab['ind'].shape, target_data_matlab['st_arr'].shape)

            y = target_data_matlab['st_arr']
            preds_idxs = prediction_data_matlab['ind']
            preds = np.array([
                1 if idx in preds_idxs else 0
                for idx in range(1, len(y) + 1)
            ])

            # REMOVE ME
            # y to seizures
            positive_idxs = np.where(y.flatten() == 1)[0]
            seizures_idxs = ranges(positive_idxs.tolist())

            seizures_mat = list()
            for seizure_idx_start, seizure_idx_end in seizures_idxs:
                seizure = {
                    'start': seizure_idx_start * 60,
                    'end': seizure_idx_end * 60 + 60,
                }
                seizures_mat.append(seizure)

            seizures_json = dataset_info['subjects_info'][subject_key]['seizures']

            file_path = r'D:\Study\asp\thesis\implementation\data\final_test_data.mat'
            data = mat73.loadmat(file_path)
            # REMOVE ME

            if dilation_times > 0:
                preds = scipy.ndimage.binary_dilation(preds, iterations=dilation_times)

            positive_minutes_num = np.sum(preds)
            total_minutes_num = len(preds)
            positive_minutes_all_subjects += positive_minutes_num
            total_minutes_all_subjects += total_minutes_num

            preds_10_seconds = list()
            for minute_idx in range(len(preds)):
                pred = preds[minute_idx]
                preds_10_seconds += [pred for _ in range(6)]
            preds_10_seconds = np.array(preds_10_seconds).astype(np.uint8)
            preds_10_seconds = preds_10_seconds.astype(np.float32)

            labels_10_seconds = list()
            for minute_idx in range(len(y)):
                label = y[minute_idx][0]
                labels_10_seconds += [label for _ in range(6)]
            labels_10_seconds = np.array(labels_10_seconds)
            labels_10_seconds = labels_10_seconds.astype(np.float32)

            time_idxs_start = np.array([i * 10 * sfreq for i in range(len(labels_10_seconds))], dtype=np.float32)
            prediction_data = {
                'time_idxs_start': time_idxs_start,
                'time_idxs_end': time_idxs_start + 10 * sfreq,
                'probs_wo_tta': preds_10_seconds,
                'probs': preds_10_seconds,
                'labels': labels_10_seconds,
            }
            if save_json_targets:
                prediction_other_path = os.path.join('D:\\Study\\asp\\thesis\\implementation\\experiments\\20231107_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NoFilt', 'predictions', rf'{subject_key}.pickle')
                prediction_other_data = pickle.load(open(prediction_other_path, 'rb'))
                trim_idx = len(prediction_other_data['labels']) - (len(prediction_other_data['labels']) % 6)
                prediction_data['labels'] = prediction_other_data['labels'][:trim_idx]
                prediction_data['subject_seizures'] = prediction_other_data['subject_seizures']

                # trim long records
                min_length = 1e9
                for key in prediction_data:
                    if not isinstance(prediction_data[key], np.ndarray):
                        continue
                    min_length = min(min_length, len(prediction_data[key]))

                for key in prediction_data:
                    if not isinstance(prediction_data[key], np.ndarray):
                        continue
                    prediction_data[key] = prediction_data[key][:min_length]

            # print(subject_key)
            # prediction_data_path = os.path.join(prediction_data_dir, f'{subject_key}.pickle')
            # with open(prediction_data_path, 'wb') as fh:
            #     pickle.dump(prediction_data, fh, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(os.path.basename(threshold_predictions_dir), 'no targets_v2 for', subject_key)

    # print()
    print(f'{threshold_str} positives = {positive_minutes_all_subjects}/{total_minutes_all_subjects} ({(positive_minutes_all_subjects / total_minutes_all_subjects) * 100:.4f}%)')
    # exit()


if __name__ == '__main__':
    base_dir = 'D:\\Study\\asp\\thesis\\implementation\\data_orig\\SVM_outliers_new_data\\results'
    dataset_info_path = r'D:\Study\asp\thesis\implementation\data\dataset_info.json'

    svm_predictions_dilation_times = 2  # expand OCSVM predictions by 2 minutes left and right
    save_json_targets = True  # save targets used for NN models training (with beginning and the end of seizure) instead of targets used for OCSVM training (only minute of the beginning)

    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    for threshold_str in os.listdir(base_dir):
        if threshold_str == 'targets':
            continue

        if threshold_str != '000250':
            continue

        threshold_dir = os.path.join(base_dir, threshold_str)
        get_metrics_for_threshold(
            threshold_dir,
            svm_predictions_dilation_times,
            sfreq=128,
            save_json_targets=save_json_targets,
        )

        # break
