import copy
import gc
import os
import pickle
import random
import traceback

import numpy as np
import torch

from utils.common import filter_predictions, calc_segments_metrics, overlapping_segment, get_tp_fp_fn_segments
from scripts.visualize_errors import visualize_samples
import utils.avg_meters


def get_segments(prediction_data, threshold, filter_method='median', k_size=7, sfreq=128):
    # get filtered probs
    time_idxs_start = prediction_data['time_idxs_start']
    time_idxs_end = prediction_data['time_idxs_end']
    probs = prediction_data['probs_wo_tta'] if 'probs_wo_tta' in prediction_data else prediction_data['probs']

    # # f1_score avg =     0.5808 sum =    26.1358	precision_score avg =     0.5493 sum =    24.7175	recall_score avg =     0.6648 sum =    29.9167	fp_num avg =     1.1333 sum =    51.0000	tp_num avg =     1.0667 sum =    48.0000	fn_num avg =     0.6222 sum =    28.0000	tn_num avg =    -1.0000 sum =   -45.0000
    # name = '20231107_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NoFilt'
    # prediction_path2 = rf'D:\Study\asp\thesis\implementation\experiments\{name}\predictions\{subject_key}.pickle'
    # prediction_data2 = pickle.load(open(prediction_path2, 'rb'))
    # probs[probs > 0.95] = prediction_data2['probs_wo_tta'][probs > 0.95]

    if len(probs) == 0:
        probs = prediction_data['probs']

    probs_filtered = filter_predictions(probs, filter_method, k_size)
    preds = probs_filtered > threshold

    # # find spikes
    # spikes_num = 0
    # for idx in range(1, len(preds) - 1):
    #     if preds[idx] != preds[idx - 1] and preds[idx] != preds[idx + 1]:
    #         spikes_num += 1
    # print(f'spikes_num = {spikes_num}')

    # extract segments
    idx = 0
    segment_len = 0
    segments = list()
    normal_segments_naive, seizure_segments_naive = list(), list()
    while idx < len(preds):
        segment_start_time = time_idxs_start[idx] / sfreq
        segment_type = preds[idx]
        segment_probs, segment_probs_filtered = list(), list()
        while idx < len(preds) and preds[idx] == segment_type:
            segment_probs.append(probs[idx])
            segment_probs_filtered.append(probs_filtered[idx])

            idx += 1
            segment_len += 1
        segment_end_time = time_idxs_end[idx - 1] / sfreq if idx != (len(preds) - 1) else time_idxs_end[idx] / sfreq

        segments.append({
            'merged_segments_num': segment_len,
            'start': segment_start_time,
            'end': segment_end_time,
            'seizure_segment': bool(segment_type),
            'probs': segment_probs,
            'probs_filtered': segment_probs_filtered,
        })
        segment_len = 0

        segment = {
            'start': segment_start_time,
            'end': segment_end_time,
        }
        if segment_type:
            seizure_segments_naive.append(segment)
        else:
            normal_segments_naive.append(segment)

    # print(os.path.basename(prediction_data['subject_eeg_path']))
    # print(f'Seizures num = {len(prediction_data["subject_seizures"])}')
    # print(f'Naive merge seizure_segments_naive_num = {len(seizure_segments_naive)}')
    # print(f'Naive merge normal_segments_naive_num = {len(normal_segments_naive)}')

    # merge stochastic segments
    idx = 0
    N_max_merged_segments_num = 2
    segments_wo_stochastic = list()
    while idx < len(segments):
        segment_start_idx = idx
        while idx < len(segments) and (segments[idx]['merged_segments_num'] <= N_max_merged_segments_num or segments[idx]['seizure_segment']):
            idx += 1
        segment_end_idx = idx

        if (segment_end_idx - segment_start_idx) == 0:
            segments_wo_stochastic.append(segments[idx])
            idx += 1
        else:
            if not segments[segment_start_idx]['seizure_segment'] and segment_start_idx > 0:
                assert segments[segment_start_idx - 1]['seizure_segment']
                segment_start_idx -= 1
                segments_wo_stochastic = segments_wo_stochastic[:-1]

            if not segments[segment_end_idx - 1]['seizure_segment'] and segment_end_idx < len(segments):
                assert segments[segment_end_idx]['seizure_segment']
                segment_end_idx += 1
                idx += 1

            merged_stochastic_segment = {
                'merged_segments_num': sum([segments[segment_start_idx + shift_idx]['merged_segments_num'] for shift_idx in range(segment_end_idx - segment_start_idx)]),
                'start': segments[segment_start_idx]['start'],
                'end': segments[segment_end_idx - 1]['end'],
                'seizure_segment': True,
                'probs': sum([segments[idx]['probs'] for idx in range(segment_start_idx, segment_end_idx)], []),
                'probs_filtered': sum([segments[idx]['probs_filtered'] for idx in range(segment_start_idx, segment_end_idx)], []),
            }
            segments_wo_stochastic.append(merged_stochastic_segment)

    normal_segments = [segment for segment in segments_wo_stochastic if not segment['seizure_segment']]
    seizure_segments = [segment for segment in segments_wo_stochastic if segment['seizure_segment']]

    # print(f'Advanced merge seizure_segments_num = {len(seizure_segments)}')
    # print(f'Advanced merge normal_segments_num = {len(normal_segments)}')

    # if 'dataset16' in os.path.basename(prediction_data['subject_eeg_path']):  # CNN so-so
    #     print('debug')

    # if '020tl Anonim-20201216_073813-20211122_171341' in os.path.basename(prediction_data['subject_eeg_path']):  # CNN so-so
    #     print('debug')

    # if '020tl Anonim-20201218_071731-20211122_171454' in os.path.basename(prediction_data['subject_eeg_path']):  # CNN so-so
    #     print('debug')

    # if 'dataset1' in os.path.basename(prediction_data['subject_eeg_path']):  # CNN not ok
    #     print('debug')

    # if '040tl Anonim-20200421_100248-20211123_010147' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN so-so
    #     print('debug')

    # if '025tl Anonim-20210129_073208-20211122_173728' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN not ok
    #     print('debug')

    # if 'dataset4' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN ok
    #     print('debug')

    # if '020tl Anonim-20201216_073813-20211122_171341' in os.path.basename(prediction_data['subject_eeg_path']):  # CNN ok - illustration
    #     print('debug')

    # if 'dataset4' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN not ok
    #     print('debug')

    # if 'dataset21' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN not ok
    #     print('debug')

    # if '023tl Anonim-20210110_080440-20211122_173058' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN not ok
    #     print('debug')

    # if '030tl Anonim-20190910_110631-20211122_180335' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN not ok
    #     print('debug')

    # if '041tl Anonim-20201112_194437-20211123_010804' in os.path.basename(prediction_data['subject_eeg_path']):  # OCSVM + CNN not ok
    #     print('debug')

    return normal_segments, seizure_segments


def get_segments_from_predictions(experiment_dir, subject_keys, threshold, filter_method='median', k_size=7, sfreq=128):
    subject_key_to_pred_segments = dict()
    for subject_key in subject_keys:
        try:
            prediction_path = os.path.join(experiment_dir, 'predictions', rf'{subject_key}.pickle')
            prediction_data = pickle.load(open(prediction_path, 'rb'))
        except Exception as e:
            prediction_path = os.path.join(experiment_dir, 'predictions_positive_only', rf'{subject_key}.pickle')
            prediction_data = pickle.load(open(prediction_path, 'rb'))

        import json
        dataset_info_path = '../data/dataset_info.json'
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        time_start, time_end = 0, dataset_info['subjects_info'][subject_key]['duration_in_seconds']
        time_points = [time_start]
        for seizure in prediction_data['subject_seizures']:
            assert seizure['start'] > time_points[-1]
            time_points.append(seizure['start'])
            time_points.append(seizure['end'])
        time_points.append(time_end)
        assert len(time_points) % 2 == 0

        normal_segments = list()
        for normal_segment_idx in range(len(time_points) // 2):
            normal_segments.append({
                'start': time_points[normal_segment_idx * 2],
                'end': time_points[normal_segment_idx * 2 + 1],
            })

        normal_segments_pred, seizure_segments_pred = get_segments(prediction_data, threshold, filter_method, k_size, sfreq)
        subject_key_to_pred_segments[subject_key] = {
            'normals_pred': normal_segments_pred,
            'seizures_pred': seizure_segments_pred,
            'seizures': prediction_data['subject_seizures'],
            'normals': normal_segments,
            'record_duration': time_end / 3600,  # duration in hours
        }
        # break

    return subject_key_to_pred_segments


def find_closest_segments(curr_segment, segments_to_search):
    closest_segment = None
    closest_distance = float('inf')
    overlapping_distance = 0
    for segment_to_search in segments_to_search:
        curr_overlapping_distance = overlapping_segment(curr_segment, segment_to_search)

        if segment_to_search['start'] <= curr_segment['start'] <= segment_to_search['end']:
            closest_segment = segment_to_search
            closest_distance = 0
            overlapping_distance = curr_overlapping_distance
            break

        if segment_to_search['start'] <= curr_segment['end'] <= segment_to_search['end']:
            closest_segment = segment_to_search
            closest_distance = 0
            overlapping_distance = curr_overlapping_distance
            break

        if overlapping_distance > 0:
            closest_segment = segment_to_search
            closest_distance = 0
            overlapping_distance = curr_overlapping_distance
            break

        curr_distance = min(
            abs(curr_segment['start'] - segment_to_search['start']),
            abs(curr_segment['start'] - segment_to_search['end']),
            abs(curr_segment['end'] - segment_to_search['start']),
            abs(curr_segment['end'] - segment_to_search['end']),
        )

        if curr_distance < closest_distance:
            closest_distance = curr_distance
            closest_segment = segment_to_search

    return closest_segment, closest_distance, overlapping_distance


def visualize_predicted_segments(subject_eeg_path, seizure_segments_true, seizure_segments_pred, intersection_part_threshold):
    import eeg_reader
    import datasets.datasets_static
    import visualization

    raw = eeg_reader.EEGReader.read_eeg(subject_eeg_path, preload=True)
    datasets.datasets_static.drop_unused_channels(subject_eeg_path, raw)
    channel_names = raw.info['ch_names']
    recording_duration = raw.times.max() - raw.times.min()

    # baseline stats
    import numpy as np
    freqs = np.arange(1, 40.01, 0.1)
    baseline_mean, baseline_std = datasets.datasets_static.get_baseline_stats(
        raw,
        baseline_length_in_seconds=500,
        sfreq=raw.info['sfreq'],
        freqs=freqs,
    )

    # Noise reduction
    raw_filtered = None
    # subset_name = os.path.basename(os.path.dirname(subject_eeg_path))
    # subject_name = os.path.splitext(os.path.basename(subject_eeg_path))[0]
    # ica_dir = 'D:/Study/asp/thesis/implementation/data/ica_20231023/'
    # ica_path = os.path.join(ica_dir, subset_name, f'{subject_name}.fif')
    # if os.path.exists(ica_path):
    #     import mne
    #     import copy
    #     ica = mne.preprocessing.read_ica(ica_path)
    #     raw_filtered = ica.apply(copy.deepcopy(raw))

    padding_sec = 10
    for seizure_idx, seizure_segment_pred in enumerate(seizure_segments_pred):
        closest_segment, closest_distance, overlapping_distance = find_closest_segments(seizure_segment_pred, seizure_segments_true)

        segment_min_start_time = min(closest_segment['start'], seizure_segment_pred['start']) if closest_distance < 600 else seizure_segment_pred['start']
        segment_max_end_time = max(closest_segment['end'], seizure_segment_pred['end']) if closest_distance < 600 else seizure_segment_pred['end']

        start_time = max(0, segment_min_start_time - padding_sec)
        duration = segment_max_end_time - segment_min_start_time + padding_sec * 2
        if (start_time + duration) > recording_duration:
            duration = duration - (start_time + duration - recording_duration) - 1

        seizure_sample, _, _ = datasets.datasets_static.generate_raw_samples(
            raw,
            sample_start_times=[start_time],
            sample_duration=duration,
        )
        # print(f'{subject_key} seizure_idx = {seizure_idx} seizure_sample = {seizure_sample.shape}')

        segments_to_visualize = [
            {
                'start': seizure_segment_pred['start'] - segment_min_start_time + padding_sec,
                'end': seizure_segment_pred['end'] - segment_min_start_time + padding_sec,
            },
            {
                'start': closest_segment['start'] - segment_min_start_time + padding_sec,
                'end': closest_segment['end'] - segment_min_start_time + padding_sec,
            },
        ]
        print(seizure_idx, segments_to_visualize)

        seizure_distance = closest_segment['end'] - closest_segment['start']
        print(f'subject_key = {subject_key} seizure_idx = {seizure_idx} closest_distance = {closest_distance} seizure_distance = {seizure_distance} overlapping_distance = {overlapping_distance} overlap = {overlapping_distance / seizure_distance}')

        # TODO: fix
        set_name = 'tp' if overlapping_distance / seizure_distance > intersection_part_threshold else 'fp'
        save_name = f'{subject_key.replace("/", "_")}_seizure{int(seizure_idx)}.png'
        save_path = os.path.join(save_dir, set_name, save_name)

        # visualization.visualize_raw(
        #     seizure_sample[0],
        #     channel_names,
        #     seizure_times_list=segments_to_visualize,
        #     heatmap=None,
        #     save_path=save_path,
        # )
        print(f'Saved to {set_name}/{save_name}')

        seizure_probs = [int(seizure_idx)]
        visualize_samples(
            seizure_sample,
            seizure_probs,
            [start_time],
            channel_names,
            sfreq=128,
            baseline_mean=baseline_mean,
            set_name=set_name,
            subject_key=subject_key,
            visualization_dir=save_dir,
            seizure_times_list=segments_to_visualize,
        )

        if raw_filtered is not None:
            seizure_sample_filtered, _, _ = datasets.datasets_static.generate_raw_samples(
                raw_filtered,
                sample_start_times=[start_time],
                sample_duration=duration,
            )
            if closest_distance == 0:  # TP
                save_path = os.path.join(save_dir, 'tp_filtered', f'{subject_key.replace("/", "_")}_seizure{int(seizure_idx)}.png')
            else:  # FP
                save_path = os.path.join(save_dir, 'fp_filtered', f'{subject_key.replace("/", "_")}_seizure{int(seizure_idx)}.png')

            visualization.visualize_raw(
                seizure_sample_filtered[0],
                channel_names,
                seizure_times_list=segments_to_visualize,
                heatmap=None,
                save_path=save_path,
            )

    for seizure_idx, seizure_segment_true in enumerate(seizure_segments_true):
        overlap_distances = [
            overlapping_segment(seizure_segment_true, seizure_segment_pred)
            for seizure_segment_pred in seizure_segments_pred
        ]
        seizure_distance = seizure_segment_true['end'] - seizure_segment_true['start']
        overlap_with_pred = any([
            overlap_distance / seizure_distance > intersection_part_threshold
            for overlap_distance in overlap_distances
        ])

        print('seizure_idx_true', seizure_idx, seizure_segment_true, overlap_with_pred, overlap_distances)

        if overlap_with_pred:
            continue

        start_time = max(0, seizure_segment_true['start'] - padding_sec)
        duration = seizure_segment_true['end'] - seizure_segment_true['start'] + padding_sec * 2
        if (start_time + duration) > recording_duration:
            duration = duration - (start_time + duration - recording_duration) - 1

        seizure_sample, _, _ = datasets.datasets_static.generate_raw_samples(
            raw,
            sample_start_times=[start_time],
            sample_duration=duration,
        )
        # print(f'{subject_key} seizure_idx = {seizure_idx} seizure_sample = {seizure_sample.shape}')

        segments_to_visualize = [
            {
                'start': padding_sec,
                'end': seizure_segment_true['end'] - seizure_segment_true['start'] + padding_sec,
            },
        ]
        if len(overlap_distances) > 0:
            max_overlap_idx = np.argmax(np.array(overlap_distances))
            segments_to_visualize.append(
                {
                    'start': seizure_segments_pred[max_overlap_idx]['start'] - seizure_segment_true['start'] + padding_sec,
                    'end': seizure_segments_pred[max_overlap_idx]['end'] - seizure_segment_true['start'] + padding_sec,
                },
            )
            seizure_times_colors = ('green', 'red')
            seizure_times_ls = ('--', '-')
        else:
            seizure_times_colors = ('green', )
            seizure_times_ls = ('--', )
        print(int(seizure_idx + len(seizure_segments_pred)), segments_to_visualize)

        # FN
        set_name = 'fn'
        save_name = f'{subject_key.replace("/", "_")}_seizure{int(seizure_idx + len(seizure_segments_pred))}.png'
        save_path = os.path.join(save_dir, set_name, save_name)

        # visualization.visualize_raw(
        #     seizure_sample[0],
        #     channel_names,
        #     seizure_times_colors=('green', 'red'),
        #     seizure_times_ls=('--', '-'),
        #     seizure_times_list=segments_to_visualize,
        #     heatmap=None,
        #     save_path=save_path,
        # )
        print(f'Saved to {set_name}/{save_name}')

        seizure_probs = [int(seizure_idx + len(seizure_segments_pred))]
        visualize_samples(
            seizure_sample,
            seizure_probs,
            [start_time],
            channel_names,
            sfreq=128,
            baseline_mean=baseline_mean,
            set_name=set_name,
            subject_key=subject_key,
            visualization_dir=save_dir,
            seizure_times_list=segments_to_visualize,
            seizure_times_colors=seizure_times_colors,
            seizure_times_ls=seizure_times_ls,
        )

        if raw_filtered is not None:
            seizure_sample_filtered, _, _ = datasets.datasets_static.generate_raw_samples(
                raw_filtered,
                sample_start_times=[start_time],
                sample_duration=duration,
            )

            save_path = os.path.join(save_dir, 'fn_filtered', f'{subject_key.replace("/", "_")}_seizure{int(seizure_idx + len(seizure_segments_pred))}.png')

            visualization.visualize_raw(
                seizure_sample_filtered[0],
                channel_names,
                seizure_times_colors=seizure_times_colors,
                seizure_times_ls=seizure_times_ls,
                seizure_times_list=segments_to_visualize,
                heatmap=None,
                save_path=save_path,
            )

    del raw
    exit()


def log_problem_segment(json_path, segment_data):
    import json
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = list()

    data.append(segment_data)

    with open(json_path, 'w') as f:
        json.dump(data, f)


def visualize_predicted_segments_v2(experiment_dir, subject_eeg_path, tp_segments, fp_segments, fn_segments, seizure_segments_true_dilation, save_dir):
    import eeg_reader
    import datasets.datasets_static

    # TODO: Remove hardcode for OCSVM
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best.pth.tar')
    model_params = {
        'class_name': 'EEGResNet18Spectrum',
        'kwargs': dict(),
        'preprocessing_params': {
            'normalization': 'meanstd',
            'transform': None,
            'data_type': 'power_spectrum',
            'baseline_correction': False,
            'log': True,
        }
    }
    # TODO: Remove hardcode for OCSVM

    problem_segments_json_path = os.path.join(save_dir, 'problem_segments.json')

    raw = eeg_reader.EEGReader.read_eeg(subject_eeg_path, preload=True)
    datasets.datasets_static.drop_unused_channels(subject_eeg_path, raw)
    channel_names = raw.info['ch_names']
    recording_duration = raw.times.max() - raw.times.min()

    # baseline stats
    import numpy as np
    freqs = np.arange(1, 40.01, 0.1)
    baseline_mean, baseline_std = datasets.datasets_static.get_baseline_stats(
        raw,
        baseline_length_in_seconds=500,
        sfreq=raw.info['sfreq'],
        freqs=freqs,
        return_baseline_spectrum=False,
    )  # (C, F, 1), (C, F, 1), [(C, F, T)]

    padding_sec = 20

    # tp
    for tp_idx, tp_data in enumerate(tp_segments):
        fname_segment_idx = [tp_idx]
        save_path = os.path.join(save_dir, 'tp', f'{subject_key.replace("/", "_")}_seizure{fname_segment_idx[0]}.png')
        save_path2 = os.path.join(save_dir, 'tp', f'{subject_key.replace("/", "_")}_seizure{fname_segment_idx[0]}_V5_clip0.75.png')
        if os.path.exists(save_path) or os.path.exists(save_path2):
            print(f'Skipping tp#{tp_idx} since visual for it already exist')
            continue

        true_segment = tp_data['true']
        true_segment_dilated = tp_data['true_dilated']
        pred_segments = tp_data['preds']

        # segment limits rounded to closest 10
        # segment_min_start_time = true_segment_dilated['start']
        # segment_max_end_time = true_segment_dilated['end']
        segment_min_start_time = round(true_segment_dilated['start'] - padding_sec, -1)
        segment_max_end_time = round(true_segment_dilated['end'] + padding_sec, -1)

        start_time = max(0, segment_min_start_time - padding_sec)
        duration = segment_max_end_time - segment_min_start_time + padding_sec * 2
        if (start_time + duration) > recording_duration:
            duration = duration - (start_time + duration - recording_duration) - 1

        print(f'{os.path.basename(subject_eeg_path)} tp_idx = {tp_idx} duration = {duration} segment_min_start_time = {segment_min_start_time} segment_max_end_time = {segment_max_end_time}')

        seizure_sample, _, _ = datasets.datasets_static.generate_raw_samples(
            raw,
            sample_start_times=[start_time],
            sample_duration=duration,
        )

        segments_to_visualize = [
            {
                'start': true_segment['start'] - segment_min_start_time + padding_sec,
                'end': true_segment['end'] - segment_min_start_time + padding_sec,
            },
            {
                'start': true_segment_dilated['start'] - segment_min_start_time + padding_sec,
                'end': true_segment_dilated['end'] - segment_min_start_time + padding_sec,
            },
        ]
        segments_times_colors = ['green', 'orange']
        segments_times_ls = ['solid', 'solid']

        possible_ls = ('dotted', 'dashed', 'dashdot')
        for pred_idx, pred_segment in enumerate(pred_segments):
            segments_to_visualize.append(
                {
                    'start': pred_segment['start'] - segment_min_start_time + padding_sec,
                    'end': pred_segment['end'] - segment_min_start_time + padding_sec,
                }
            )

            color = 'red'
            ls = possible_ls[pred_idx % len(possible_ls)]
            segments_times_colors.append(color)
            segments_times_ls.append(ls)

        fname_segment_idx = [tp_idx]
        try:
            visualize_samples(
                seizure_sample,
                fname_segment_idx,
                [start_time],
                copy.deepcopy(channel_names),
                sfreq=128,
                baseline_mean=baseline_mean,
                set_name='tp',
                subject_key=subject_key,
                visualization_dir=save_dir,
                checkpoint_path=checkpoint_path,
                model_params=model_params,
                seizure_times_list=segments_to_visualize,
                seizure_times_colors=segments_times_colors,
                seizure_times_ls=segments_times_ls,
            )
        except Exception as e:
            # tp_data['segments_to_visualize'] = segments_to_visualize
            # tp_data['segment_min_start_time'] = segment_min_start_time
            # tp_data['segment_max_end_time'] = segment_max_end_time
            # tp_data['set_name'] = 'tp'
            # tp_data['eeg_path'] = subject_eeg_path
            # tp_data['error'] = traceback.format_exc()
            # tp_data['duration'] = duration
            # print('ERROR')
            # print(str(e))
            # print(traceback.format_exc())
            # print('\n')
            # log_problem_segment(problem_segments_json_path, tp_data)
            raise e
        del seizure_sample, segments_to_visualize
        gc.collect()
        torch.cuda.empty_cache()

    # # fn
    # for fn_idx, fn_data in enumerate(fn_segments):
    #     fname_segment_idx = [fn_idx + len(tp_segments)]
    #     save_path = os.path.join(save_dir, 'fn', f'{subject_key.replace("/", "_")}_seizure{fname_segment_idx[0]}.png')
    #     if os.path.exists(save_path):
    #         print(f'Skipping fn#{fn_idx} since visual for it already exist')
    #         continue
    #
    #     true_segment = fn_data['true']
    #     true_segment_dilated = fn_data['true_dilated']
    #
    #     # segment limits rounded to closest 10
    #     segment_min_start_time = round(true_segment_dilated['start'] - 10, -1)
    #     segment_max_end_time = round(true_segment_dilated['end'] + 10, -1)
    #
    #     start_time = max(0, segment_min_start_time - padding_sec)
    #     duration = segment_max_end_time - segment_min_start_time + padding_sec * 2
    #     if (start_time + duration) > recording_duration:
    #         duration = duration - (start_time + duration - recording_duration) - 1
    #
    #     print(f'{os.path.basename(subject_eeg_path)} fn_idx = {fn_idx} duration = {duration}')
    #
    #     seizure_sample, _, _ = datasets.datasets_static.generate_raw_samples(
    #         raw,
    #         sample_start_times=[start_time],
    #         sample_duration=duration,
    #     )
    #
    #     segments_to_visualize = [
    #         {
    #             'start': true_segment['start'] - segment_min_start_time + padding_sec,
    #             'end': true_segment['end'] - segment_min_start_time + padding_sec,
    #         },
    #         {
    #             'start': true_segment_dilated['start'] - segment_min_start_time + padding_sec,
    #             'end': true_segment_dilated['end'] - segment_min_start_time + padding_sec,
    #         },
    #     ]
    #     segments_times_colors = ['green', 'orange']
    #     segments_times_ls = ['solid', 'solid']
    #
    #     try:
    #         visualize_samples(
    #             seizure_sample,
    #             fname_segment_idx,
    #             [start_time],
    #             copy.deepcopy(channel_names),
    #             sfreq=128,
    #             baseline_mean=baseline_mean,
    #             set_name='fn',
    #             subject_key=subject_key,
    #             visualization_dir=save_dir,
    #             checkpoint_path=checkpoint_path,
    #             model_params=model_params,
    #             seizure_times_list=segments_to_visualize,
    #             seizure_times_colors=segments_times_colors,
    #             seizure_times_ls=segments_times_ls,
    #         )
    #     except Exception as e:
    #         # fn_data['segments_to_visualize'] = segments_to_visualize
    #         # fn_data['segment_min_start_time'] = segment_min_start_time
    #         # fn_data['segment_max_end_time'] = segment_max_end_time
    #         # fn_data['set_name'] = 'fn'
    #         # fn_data['eeg_path'] = subject_eeg_path
    #         # fn_data['error'] = traceback.format_exc()
    #         # fn_data['duration'] = duration
    #         # print('ERROR')
    #         # print(str(e))
    #         # print(traceback.format_exc())
    #         # print('\n')
    #         # log_problem_segment(problem_segments_json_path, fn_data)
    #         raise e
    #     del seizure_sample, segments_to_visualize
    #     gc.collect()
    #     torch.cuda.empty_cache()

    # # fp
    # for fp_idx, fp_data in enumerate(fp_segments):
    #     fp_idx = random.choice(list(range(len(fp_segments))))
    #
    #     # if fp_idx == 3:
    #     #     break
    #
    #     fname_segment_idx = [fp_idx + len(tp_segments) + len(fn_segments)]
    #     save_path = os.path.join(save_dir, 'fp', f'{subject_key.replace("/", "_")}_seizure{fname_segment_idx[0]}.png')
    #     if os.path.exists(save_path):
    #         print(f'Skipping fp#{fp_idx} since visual for it already exist')
    #         continue
    #
    #     pred_segment = fp_data['pred']
    #     true_segment = fp_data['true']
    #     true_segment_dilated = fp_data['true_dilated']
    #
    #     # segment limits rounded to closest 10
    #     distance = min(
    #         abs(true_segment_dilated['start'] - pred_segment['start']),
    #         abs(true_segment_dilated['start'] - pred_segment['end']),
    #         abs(true_segment_dilated['end'] - pred_segment['start']),
    #         abs(true_segment_dilated['end'] - pred_segment['end']),
    #     )
    #     segment_min_start_time = min(true_segment_dilated['start'], pred_segment['start']) if distance < 60 else pred_segment['start']
    #     segment_max_end_time = max(true_segment_dilated['end'], pred_segment['end']) if distance < 60 else pred_segment['end']
    #
    #     segment_min_start_time = round(segment_min_start_time - 10, -1)
    #     segment_max_end_time = round(segment_max_end_time + 10, -1)
    #
    #     start_time = max(0, segment_min_start_time - padding_sec)
    #     duration = segment_max_end_time - segment_min_start_time + padding_sec * 2
    #     if (start_time + duration) > recording_duration:
    #         duration = duration - (start_time + duration - recording_duration) - 1
    #
    #     print(f'{os.path.basename(subject_eeg_path)} fp_idx = {fp_idx} duration = {duration}')
    #
    #     seizure_sample, _, _ = datasets.datasets_static.generate_raw_samples(
    #         raw,
    #         sample_start_times=[start_time],
    #         sample_duration=duration,
    #     )
    #
    #     segments_to_visualize = [
    #         {
    #             'start': true_segment['start'] - segment_min_start_time + padding_sec,
    #             'end': true_segment['end'] - segment_min_start_time + padding_sec,
    #         },
    #         {
    #             'start': true_segment_dilated['start'] - segment_min_start_time + padding_sec,
    #             'end': true_segment_dilated['end'] - segment_min_start_time + padding_sec,
    #         },
    #         {
    #             'start': pred_segment['start'] - segment_min_start_time + padding_sec,
    #             'end': pred_segment['end'] - segment_min_start_time + padding_sec,
    #         },
    #     ]
    #     segments_times_colors = ['green', 'orange', 'red']
    #     segments_times_ls = ['solid', 'solid', '--']
    #
    #     try:
    #         visualize_samples(
    #             seizure_sample,
    #             fname_segment_idx,
    #             [start_time],
    #             copy.deepcopy(channel_names),
    #             sfreq=128,
    #             baseline_mean=baseline_mean,
    #             set_name='fp',
    #             subject_key=subject_key,
    #             visualization_dir=save_dir,
    #             checkpoint_path=checkpoint_path,
    #             model_params=model_params,
    #             seizure_times_list=segments_to_visualize,
    #             seizure_times_colors=segments_times_colors,
    #             seizure_times_ls=segments_times_ls,
    #         )
    #     except Exception as e:
    #         # fp_data['segments_to_visualize'] = segments_to_visualize
    #         # fp_data['segment_min_start_time'] = segment_min_start_time
    #         # fp_data['segment_max_end_time'] = segment_max_end_time
    #         # fp_data['set_name'] = 'fp'
    #         # fp_data['eeg_path'] = subject_eeg_path
    #         # fp_data['error'] = traceback.format_exc()
    #         # fp_data['duration'] = duration
    #         # print('ERROR')
    #         # print(str(e))
    #         # print(traceback.format_exc())
    #         # print('\n')
    #         # log_problem_segment(problem_segments_json_path, fp_data)
    #         raise e
    #     del seizure_sample, segments_to_visualize
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #
    #     break

    del raw, baseline_mean, baseline_std
    gc.collect()
    torch.cuda.empty_cache()


def get_segment_metrics(experiment_dir, subject_keys, seizure_segments_true_dilation, intersection_part_threshold, threshold, filter_method, k_size, verbose=1):
    subject_key_to_pred_segments = get_segments_from_predictions(
        experiment_dir,
        subject_keys,
        threshold,
        filter_method,
        k_size,
    )
    # import pprint
    # pprint.pprint(subject_key_to_pred_segments)

    # calculate metric for segments
    metric_meter = utils.avg_meters.MetricMeter()
    for subject_idx, subject_key in enumerate(subject_key_to_pred_segments.keys()):
        # if subject_key == 'data2/022tl Anonim-20201210_132636-20211122_172649':
        #     print('debug')
        # if subject_key in [  # FNs of OCSVM
        #     'data2/018tl Anonim-20201211_130036-20211122_163611',  # 0 with T = 180 // 1 with T = 120
        #     # 'data1/dataset3',  # 1 with T = 180 // 1 with T = 120
        #     # 'data1/dataset6',  # 1 with T = 180 // 1 with T = 120
        #     'data2/021tl Anonim-20201223_085255-20211122_172126',  # 0 with T = 180 // 1 with T = 120
        #     'data2/040tl Anonim-20200421_100248-20211123_010147',  # 0 with T = 180 // 1 with T = 120
        #     'data2/003tl Anonim-20200831_040629-20211122_135924',  # 1 with T = 180 // 2 with T = 120
        #     # 'data2/035tl Anonim-20210324_231349-20211122_223059',  # 1 with T = 180 // 1 with T = 120
        #     # 'data2/035tl Anonim-20210324_151211-20211122_222545',  # 2 with T = 180 // 2 with T = 120
        #     # 'data1/dataset1',  # 1 with T = 180 // 1 with T = 120
        # ]:
        #     print('debug')
        # if subject_key == 'data2/025tl Anonim-20210129_073208-20211122_173728':
        #     print('debug')
        # if subject_key == 'data1/dataset24':
        #     print('debug')

        # if subject_key == 'data1/dataset21':
        #     print('debug')

        # if subject_key == 'data2/041tl Anonim-20201112_194437-20211123_010804':
        #     print('debug')

        segments_dict = subject_key_to_pred_segments[subject_key]
        metrics_dict = calc_segments_metrics(
            segments_dict['seizures'],
            segments_dict['normals'],
            segments_dict['seizures_pred'],
            segments_dict['normals_pred'],
            intersection_part_threshold=intersection_part_threshold,
            record_duration=segments_dict['record_duration'],
            seizure_segments_true_dilation=seizure_segments_true_dilation,
        )
        metric_meter.update(metrics_dict)
        if verbose > 0:
            print(
                f'{subject_key:60}'
                f'threshold = {threshold:3.2f} '
                f'p = {metrics_dict["precision_score"]:10.4f} ({metric_meter.meters["precision_score"].avg:10.4f}) '
                f'r = {metrics_dict["recall_score"]:10.4f} ({metric_meter.meters["recall_score"].avg:10.4f}) '
                f'f1 = {metrics_dict["f1_score"]:10.4f} ({metric_meter.meters["f1_score"].avg:10.4f}) '
                f'tp = {metrics_dict["tp_num"]:6} ({metric_meter.meters["tp_num"].avg:6.2f}) '
                f'fn = {metrics_dict["fn_num"]:6} ({metric_meter.meters["fn_num"].avg:6.2f}) '
                f'fp = {metrics_dict["fp_num"]:6} ({metric_meter.meters["fp_num"].avg:6.2f}) '
                f'long_pos = {metrics_dict["long_postivies"]:6} '
                # f'fn = {metrics_dict["fn_num"]:6} ({metric_meter.meters["fn_num"].avg:6.2f}) '
                # f'duration = {metrics_dict["duration"]:6.2f}'
            )
        # print(f'#{subject_idx + 1:02} subject_key = {subject_key:60} subject_metrics {" ".join([f"{key} = {value:9.4f}" for key, value in metrics_dict.items()])}')
    # print('metric_meter\n', metric_meter)

    fp_per_hour_micro = metric_meter.meters['fp_num'].sum / metric_meter.meters['duration'].sum
    fn_per_hour_micro = metric_meter.meters['fn_num'].sum / metric_meter.meters['duration'].sum
    tp_per_hour_micro = metric_meter.meters['tp_num'].sum / metric_meter.meters['duration'].sum
    tn_per_hour_micro = metric_meter.meters['tn_num'].sum / metric_meter.meters['duration'].sum
    precision_score_micro = metric_meter.meters['tp_num'].sum / (metric_meter.meters['tp_num'].sum + metric_meter.meters['fp_num'].sum) if (metric_meter.meters['tp_num'].sum + metric_meter.meters['fp_num'].sum) > 0 else 0
    recall_score_micro = metric_meter.meters['tp_num'].sum / (metric_meter.meters['tp_num'].sum + metric_meter.meters['fn_num'].sum) if (metric_meter.meters['tp_num'].sum + metric_meter.meters['fp_num'].sum) > 0 else 0
    f1_score_micro = 2 * precision_score_micro * recall_score_micro / (precision_score_micro + recall_score_micro) if (precision_score_micro + recall_score_micro) > 0 else 0

    metric_meter.update({
        'precision_score_micro': precision_score_micro,
        'recall_score_micro': recall_score_micro,
        'f1_score_micro': f1_score_micro,
        'fp_per_h_micro': fp_per_hour_micro,
        'fn_per_h_micro': fn_per_hour_micro,
        'tp_per_h_micro': tp_per_hour_micro,
        'tn_per_h_micro': tn_per_hour_micro,
    })

    return metric_meter


def get_best_threshold_for_segment_merging(experiment_dir, subject_keys, seizure_segments_true_dilation, intersection_part_threshold, min_recall_threshold, filter_method, k_size, verbose=1):
    assert 0 < min_recall_threshold <= 1

    best_avg_recall = -1
    best_avg_precision = -1
    best_threshold = -1
    best_metric_meter = None
    threshold_range = list(np.round(np.arange(0.05, 1, 0.05), 2))
    for threshold_idx, threshold in enumerate(threshold_range):
        metric_meter = get_segment_metrics(experiment_dir, subject_keys, seizure_segments_true_dilation, intersection_part_threshold, threshold, filter_method, k_size, verbose)

        if metric_meter.meters['recall_score'].avg >= min(best_avg_recall, min_recall_threshold) and metric_meter.meters['precision_score'].avg > best_avg_precision:
            best_avg_recall = metric_meter.meters['recall_score'].avg
            best_avg_precision = metric_meter.meters['precision_score'].avg
            best_threshold = threshold
            best_metric_meter = metric_meter

        if verbose:
            # print(f'threshold = {threshold:3.2f} f1_score = {metric_meter.meters["f1_score"].avg:.4f} precision_score = {metric_meter.meters["precision_score"].avg:.4f} recall_score = {metric_meter.meters["recall_score"].avg:.4f} long_postivies = {metric_meter.meters["long_postivies"].avg:.4f} {"best_threshold" if threshold == best_threshold else ""}')
            print(f't = {threshold:3.2f} f1 = {metric_meter.meters["f1_score"].avg:.4f} p = {metric_meter.meters["precision_score"].avg:.4f} r = {metric_meter.meters["recall_score"].avg:.4f} long_pos = {metric_meter.meters["long_postivies"].sum:7.4f} best_t = {best_threshold:3.2f}')

    return best_threshold, best_metric_meter


def print_tables(
        best_threshold_10sec_val,
        metric_meter_10sec_train,
        metric_meter_10sec_val,
        metric_meter_10sec_test,
        best_threshold_segment_merging_val,
        metric_meter_segment_merging_train,
        metric_meter_segment_merging_val,
        metric_meter_segment_merging_test,
):
    # 10 sec long precision, recall, f1
    print(f'10sec long segments t = {best_threshold_10sec_val:3.2f}')
    print(f'{"":24} {"r_macro":10} {"p_macro":10} {"f1_macro":10} {"r_micro":10} {"p_micro":10} {"f1_micro":10}')
    print(
        f'{"test":20} '
        f'{metric_meter_10sec_test.meters["recall_score"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["precision_score"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["f1_score"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["recall_score_micro"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["precision_score_micro"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["f1_score_micro"].avg:10.4f} '
    )

    print(
        f'{"train":20} '
        f'{metric_meter_10sec_train.meters["recall_score"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["precision_score"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["f1_score"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["recall_score_micro"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["precision_score_micro"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["f1_score_micro"].avg:10.4f} '
    )

    print(
        f'{"val":20} '
        f'{metric_meter_10sec_val.meters["recall_score"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["precision_score"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["f1_score"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["recall_score_micro"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["precision_score_micro"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["f1_score_micro"].avg:10.4f} '
    )

    print()

    # 10 sec long FN, FP, TP
    print(f'10sec long segments t = {best_threshold_10sec_val:3.2f}')
    print(f'{"":21} {"FN_per_h_macro":15} {"FP_per_h_macro":15} {"TP_per_h_macro":15} {"TN_per_h_macro":15} {"FN_per_h_micro":15} {"FP_per_h_micro":15} {"TP_per_h_micro":15} {"TN_per_h_micro":15}')
    print(
        f'{"test":20} '
        f'{metric_meter_10sec_test.meters["fn_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_test.meters["fp_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_test.meters["tp_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_test.meters["tn_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_test.meters["fn_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_test.meters["fp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_test.meters["tp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_test.meters["tn_per_h_micro"].avg:15.4f} '
    )

    print(
        f'{"train":20} '
        f'{metric_meter_10sec_train.meters["fn_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_train.meters["fp_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_train.meters["tp_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_train.meters["tn_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_train.meters["fn_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_train.meters["fp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_train.meters["tp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_train.meters["tn_per_h_micro"].avg:15.4f} '
    )

    print(
        f'{"val":20} '
        f'{metric_meter_10sec_val.meters["fn_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_val.meters["fp_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_val.meters["tp_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_val.meters["tn_per_h"].avg:15.4f} '
        f'{metric_meter_10sec_val.meters["fn_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_val.meters["fp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_val.meters["tp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_10sec_val.meters["tn_per_h_micro"].avg:15.4f} '
    )

    print()

    # 10 sec long FN_avg, FP_avg, TP_avg, FN_avg, FP_avg, TP_avg
    print(f'10sec long segments t = {best_threshold_10sec_val:3.2f}')
    print(f'{"":24} {"FN_avg":10} {"FP_avg":10} {"TP_avg":10} {"TN_avg":10} {"FN_sum":10} {"FP_sum":10} {"TP_sum":10} {"TN_sum":10}')
    print(
        f'{"test":20} '
        f'{metric_meter_10sec_test.meters["fn_num"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["fp_num"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["tp_num"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["tn_num"].avg:10.4f} '
        f'{metric_meter_10sec_test.meters["fn_num"].sum:10} '
        f'{metric_meter_10sec_test.meters["fp_num"].sum:10} '
        f'{metric_meter_10sec_test.meters["tp_num"].sum:10} '
        f'{metric_meter_10sec_test.meters["tn_num"].sum:10} '
    )

    print(
        f'{"train":20} '
        f'{metric_meter_10sec_train.meters["fn_num"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["fp_num"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["tp_num"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["tn_num"].avg:10.4f} '
        f'{metric_meter_10sec_train.meters["fn_num"].sum:10} '
        f'{metric_meter_10sec_train.meters["fp_num"].sum:10} '
        f'{metric_meter_10sec_train.meters["tp_num"].sum:10} '
        f'{metric_meter_10sec_train.meters["tn_num"].sum:10} '
    )

    print(
        f'{"val":20} '
        f'{metric_meter_10sec_val.meters["fn_num"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["fp_num"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["tp_num"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["tn_num"].avg:10.4f} '
        f'{metric_meter_10sec_val.meters["fn_num"].sum:10} '
        f'{metric_meter_10sec_val.meters["fp_num"].sum:10} '
        f'{metric_meter_10sec_val.meters["tp_num"].sum:10} '
        f'{metric_meter_10sec_val.meters["tn_num"].sum:10} '
    )

    print()

    # Arbitrary long precision, recall, f1
    print(f'Arbitrary long segments t = {best_threshold_segment_merging_val:3.2f}')
    print(f'{"":24} {"r_macro":10} {"p_macro":10} {"f1_macro":10} {"r_micro":10} {"p_micro":10} {"f1_micro":10}')
    print(
        f'{"test":20} '
        f'{metric_meter_segment_merging_test.meters["recall_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["precision_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["f1_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["recall_score_micro"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["precision_score_micro"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["f1_score_micro"].avg:10.4f} '
    )

    print(
        f'{"train":20} '
        f'{metric_meter_segment_merging_train.meters["recall_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["precision_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["f1_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["recall_score_micro"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["precision_score_micro"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["f1_score_micro"].avg:10.4f} '
    )

    print(
        f'{"val":20} '
        f'{metric_meter_segment_merging_val.meters["recall_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["precision_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["f1_score"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["recall_score_micro"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["precision_score_micro"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["f1_score_micro"].avg:10.4f} '
    )

    print()

    # Arbitrary long FN, FP, TP
    print(f'Arbitrary long segments t = {best_threshold_segment_merging_val:3.2f}')
    print(f'{"":21} {"FN_per_h_macro":15} {"FP_per_h_macro":15} {"TP_per_h_macro":15} {"TN_per_h_macro":15} {"FN_per_h_micro":15} {"FP_per_h_micro":15} {"TP_per_h_micro":15} {"TN_per_h_micro":15}')
    print(
        f'{"test":20} '
        f'{metric_meter_segment_merging_test.meters["fn_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_test.meters["fp_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_test.meters["tp_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_test.meters["tn_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_test.meters["fn_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_test.meters["fp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_test.meters["tp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_test.meters["tn_per_h_micro"].avg:15.4f} '
    )

    print(
        f'{"train":20} '
        f'{metric_meter_segment_merging_train.meters["fn_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_train.meters["fp_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_train.meters["tp_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_train.meters["tn_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_train.meters["fn_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_train.meters["fp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_train.meters["tp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_train.meters["tn_per_h_micro"].avg:15.4f} '
    )

    print(
        f'{"val":20} '
        f'{metric_meter_segment_merging_val.meters["fn_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_val.meters["fp_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_val.meters["tp_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_val.meters["tn_per_h"].avg:15.4f} '
        f'{metric_meter_segment_merging_val.meters["fn_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_val.meters["fp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_val.meters["tp_per_h_micro"].avg:15.4f} '
        f'{metric_meter_segment_merging_val.meters["tn_per_h_micro"].avg:15.4f} '
    )

    print()

    # Arbitrary long FN_avg, FP_avg, TP_avg, FN_avg, FP_avg, TP_avg
    print(f'Arbitrary long segments t = {best_threshold_segment_merging_val:3.2f}')
    print(f'{"":24} {"FN_avg":10} {"FP_avg":10} {"TP_avg":10} {"TN_avg":10} {"FN_sum":10} {"FP_sum":10} {"TP_sum":10} {"TN_sum":10}')
    print(
        f'{"test":20} '
        f'{metric_meter_segment_merging_test.meters["fn_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["fp_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["tp_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["tn_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_test.meters["fn_num"].sum:10} '
        f'{metric_meter_segment_merging_test.meters["fp_num"].sum:10} '
        f'{metric_meter_segment_merging_test.meters["tp_num"].sum:10} '
        f'{metric_meter_segment_merging_test.meters["tn_num"].sum:10} '
    )

    print(
        f'{"train":20} '
        f'{metric_meter_segment_merging_train.meters["fn_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["fp_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["tp_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["tn_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_train.meters["fn_num"].sum:10} '
        f'{metric_meter_segment_merging_train.meters["fp_num"].sum:10} '
        f'{metric_meter_segment_merging_train.meters["tp_num"].sum:10} '
        f'{metric_meter_segment_merging_train.meters["tn_num"].sum:10} '
    )

    print(
        f'{"val":20} '
        f'{metric_meter_segment_merging_val.meters["fn_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["fp_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["tp_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["tn_num"].avg:10.4f} '
        f'{metric_meter_segment_merging_val.meters["fn_num"].sum:10} '
        f'{metric_meter_segment_merging_val.meters["fp_num"].sum:10} '
        f'{metric_meter_segment_merging_val.meters["tp_num"].sum:10} '
        f'{metric_meter_segment_merging_val.meters["tn_num"].sum:10} '
    )

    print()


if __name__ == '__main__':
    # parameters
    # experiment_name = '20231012_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded'
    # experiment_name = '20231005_EEGResNetCustomRaw_MixUp_TimeSeriesAug_raw_16excluded'
    # experiment_name = '20231005_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded_wo_baseline_correction'
    # experiment_name = '20231024_EEGResNet18Raw_MixUp_TimeSeriesAug_raw_16excluded'
    # experiment_name = '20231025_EEGResNet18Raw_MixUp_TimeSeriesAug_raw_16excluded_wo_baseline_correction'
    # experiment_name = 'renset18_2nd_stage_MixUp_SpecTimeFlipEEGFlipAug'
    # experiment_name = 'SVM_outliers_new_data_000250_dilation=0'  # OCSVM
    # experiment_name = 'renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug'  # NN
    experiment_name = '20231107_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NoFilt'  # NN + NN
    # experiment_name = '20231213_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_OCSVM_positive_only_16excluded'  # NN + OCSVM
    # experiment_name = '20231107_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NoFilt_pos_only'  # NN + NN + OCSVM
    # experiment_name = '20240110_EEGResNet18Spectrum_Default_NoiseBaseline_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NN'  # NN + NN + Noise
    # experiment_name = '20231225_EEGResNet18Spectrum_MixUp_NoiseBaseline_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NN'  # NN + NN + Noise + MixUp
    # experiment_name = 'SVM_outliers_new_data_000250'
    # experiment_name = 'SVM_outliers_new_data_000250_dilation=2'  # identical to SVM_outliers_new_data_000250
    # experiment_name = '20240118_EEGResNet18Spectrum_Default_NoiseBaseline_SpecTimeFlipEEGFlipAug_meanstd_norm_602020splitV1'
    # experiment_name = '20240129_EEGResNet18Spectrum_Default_NoiseBaseline_SpecTimeFlipEEGFlipAug_meanstd_norm_602020splitV2'
    experiment_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name)

    # global settings
    split_name = 'base'
    # split_name = '602020_v1'
    # split_name = '602020_v2'
    visualize_segments = True
    exclude_16 = True
    filter = True
    verbose = 1
    min_recall_threshold = 0.9
    intersection_part_threshold = 0.51
    seizure_segments_true_dilation = 60 * 1

    if filter:
        filter_method = 'median'
        k_size = 7
    else:
        filter_method = None
        k_size = -1

    print(experiment_name)
    print(f'split_name = {split_name}')
    print(f'filter={filter_method} k={k_size} exclude_16={exclude_16} seizure_segments_true_dilation={seizure_segments_true_dilation} intersection_part_threshold={intersection_part_threshold} min_recall_threshold={min_recall_threshold}')
    print()

    # evaluating val to find best_threshold for 10sec preds
    print('evaluating val to find best_threshold for 10sec preds')
    import data_split
    subject_keys = data_split.get_subject_keys_val()
    # subject_keys = data_split.get_subject_keys('val', split_name)
    subject_keys_exclude = data_split.get_subject_keys_exclude_16()
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    from scripts.find_threshold import get_best_threshold, get_metrics
    best_threshold_10sec_val, metric_meter_10sec_val = get_best_threshold(
        experiment_dir=experiment_dir,
        subject_keys=subject_keys,
        filter_method=filter_method,
        k_size=k_size,
        verbose=verbose,
    )
    print(f'val best_threshold = {best_threshold_10sec_val}')
    print(f'metric_meter_10sec_val:\n{metric_meter_10sec_val}')
    print()

    # evaluating test performance with best_threshold_10sec_val for 10sec preds
    print('evaluating test performance with best_threshold_10sec_val for 10sec preds')
    subject_keys = data_split.get_subject_keys_test()
    # subject_keys = data_split.get_subject_keys('test', split_name)
    subject_keys_exclude = data_split.get_subject_keys_exclude_16()
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    metric_meter_10sec_test = get_metrics(
        experiment_dir=experiment_dir,
        subject_keys=subject_keys,
        threshold=best_threshold_10sec_val,
        filter_method=filter_method,
        k_size=k_size,
        verbose=verbose,
    )

    print(f'test threshold = {best_threshold_10sec_val}')
    print(f'metric_meter_10sec_test:\n{metric_meter_10sec_test}')
    print()

    # evaluating train performance with best_threshold_10sec_val for 10sec preds
    print('evaluating train performance with best_threshold_10sec_val for 10sec preds')
    subject_keys = data_split.get_subject_keys_train()
    # subject_keys = data_split.get_subject_keys('train', split_name)
    subject_keys_exclude = data_split.get_subject_keys_exclude_16()
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    metric_meter_10sec_train = get_metrics(
        experiment_dir=experiment_dir,
        subject_keys=subject_keys,
        threshold=best_threshold_10sec_val,
        filter_method=filter_method,
        k_size=k_size,
        verbose=verbose,
    )

    print(f'train threshold = {best_threshold_10sec_val}')
    print(f'metric_meter_10sec_train:\n{metric_meter_10sec_train}')
    print()

    # evaluating val to find best_threshold_segment for merging of 10sec preds into arbitrary long segments
    print('evaluating val to find best_threshold_segment for merging of 10sec preds into arbitrary long segments')
    subject_keys = data_split.get_subject_keys_val()
    # subject_keys = data_split.get_subject_keys('val', split_name)
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    best_threshold_segment_merging_val, metric_meter_segment_merging_val = get_best_threshold_for_segment_merging(
        experiment_dir=experiment_dir,
        subject_keys=subject_keys,
        seizure_segments_true_dilation=seizure_segments_true_dilation,
        intersection_part_threshold=intersection_part_threshold,
        min_recall_threshold=min_recall_threshold,
        filter_method=filter_method,
        k_size=k_size,
        verbose=verbose,
    )
    # best_threshold_segment_merging_val = 0.95
    print(f'val best_threshold_segment_merging_val = {best_threshold_segment_merging_val}')
    print(f'metric_meter_segment_merging_val:\n{metric_meter_segment_merging_val}')

    # evaluating test performance with arbitrary long segments and best_threshold_segment_merging_val
    print('evaluating test performance with arbitrary long segments and best_threshold_segment_merging_val')
    subject_keys = data_split.get_subject_keys_test()
    # subject_keys = data_split.get_subject_keys('test', split_name)
    subject_keys_exclude = data_split.get_subject_keys_exclude_16()
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    metric_meter_segment_merging_test = get_segment_metrics(
        experiment_dir=experiment_dir,
        subject_keys=subject_keys,
        seizure_segments_true_dilation=seizure_segments_true_dilation,
        intersection_part_threshold=intersection_part_threshold,
        threshold=best_threshold_segment_merging_val,
        filter_method=filter_method,
        k_size=k_size,
        # verbose=verbose,
        verbose=2,
    )
    print(f'test t = {best_threshold_segment_merging_val:3.2f} f1 = {metric_meter_segment_merging_test.meters["f1_score"].avg:.4f} p = {metric_meter_segment_merging_test.meters["precision_score"].avg:.4f} r = {metric_meter_segment_merging_test.meters["recall_score"].avg:.4f} fp = {metric_meter_segment_merging_test.meters["fp_num"].avg:7.4f} ({metric_meter_segment_merging_test.meters["fp_num"].sum:7.4f}) fn = {metric_meter_segment_merging_test.meters["fn_num"].avg:7.4f} ({metric_meter_segment_merging_test.meters["fn_num"].sum:7.4f}) tp = {metric_meter_segment_merging_test.meters["tp_num"].avg:7.4f} ({metric_meter_segment_merging_test.meters["tp_num"].sum:7.4f}) p_micro = {metric_meter_segment_merging_test.meters["precision_score_micro"].avg:.4f} r_micro = {metric_meter_segment_merging_test.meters["recall_score_micro"].avg:.4f}')
    print(f'test t = {best_threshold_segment_merging_val:3.2f} {metric_meter_segment_merging_test}')
    print('\n')

    # evaluating train performance with arbitrary long segments and best_threshold_segment_merging_val
    print('evaluating train performance with arbitrary long segments and best_threshold_segment_merging_val')
    subject_keys = data_split.get_subject_keys_train()
    # subject_keys = data_split.get_subject_keys('train', split_name)
    subject_keys_exclude = data_split.get_subject_keys_exclude_16()
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    metric_meter_segment_merging_train = get_segment_metrics(
        experiment_dir=experiment_dir,
        subject_keys=subject_keys,
        seizure_segments_true_dilation=seizure_segments_true_dilation,
        intersection_part_threshold=intersection_part_threshold,
        threshold=best_threshold_segment_merging_val,
        filter_method=filter_method,
        k_size=k_size,
        verbose=verbose,
        # verbose=2,
    )
    print(f'train t = {best_threshold_segment_merging_val:3.2f} f1 = {metric_meter_segment_merging_train.meters["f1_score"].avg:.4f} p = {metric_meter_segment_merging_train.meters["precision_score"].avg:.4f} r = {metric_meter_segment_merging_train.meters["recall_score"].avg:.4f} fp = {metric_meter_segment_merging_train.meters["fp_num"].avg:7.4f} ({metric_meter_segment_merging_train.meters["fp_num"].sum:7.4f}) fn = {metric_meter_segment_merging_train.meters["fn_num"].avg:7.4f} ({metric_meter_segment_merging_train.meters["fn_num"].sum:7.4f}) tp = {metric_meter_segment_merging_train.meters["tp_num"].avg:7.4f} ({metric_meter_segment_merging_train.meters["tp_num"].sum:7.4f}) p_micro = {metric_meter_segment_merging_train.meters["precision_score_micro"].avg:.4f} r_micro = {metric_meter_segment_merging_train.meters["recall_score_micro"].avg:.4f}')
    print(f'train t = {best_threshold_segment_merging_val:3.2f} {metric_meter_segment_merging_train}')
    print('\n')

    print(experiment_name)
    print(f'split_name = {split_name} seizure_segments_true_dilation = {seizure_segments_true_dilation}')
    print(f'filter={filter_method} k={k_size} exclude_16={exclude_16} seizure_segments_true_dilation={seizure_segments_true_dilation} intersection_part_threshold={intersection_part_threshold} min_recall_threshold={min_recall_threshold}')
    print_tables(
        best_threshold_10sec_val,
        metric_meter_10sec_train,
        metric_meter_10sec_val,
        metric_meter_10sec_test,
        best_threshold_segment_merging_val,
        metric_meter_segment_merging_train,
        metric_meter_segment_merging_val,
        metric_meter_segment_merging_test,
    )

    # visualize segments
    if visualize_segments:
        data_dir = r'D:\Study\asp\thesis\implementation\data'

        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'visualizations_segment')
        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'visual_GRADCAM_20241011_upd')
        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'visual_GRADCAM_20241012_upd')
        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'visual_GRADCAM_20241015_upd')
        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'visual_GRADCAM_20241019_upd')
        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'vis_20241020')
        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'vis_20241024')
        # save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'vis_20241031')
        save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'vis_20241104')
        os.makedirs(save_dir, exist_ok=True)

        import json

        dataset_info_path = r'D:\Study\asp\thesis\implementation\data\dataset_info.json'
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        # subject_keys = data_split.get_subject_keys_test_debug()
        # subject_keys = data_split.get_subject_keys_test() + data_split.get_subject_keys_val()`
        subject_keys = data_split.get_subject_keys_test()
        # subject_keys = data_split.get_subject_keys('test', split_name) + data_split.get_subject_keys('val', split_name)
        subject_keys_exclude = data_split.get_subject_keys_exclude_16()
        if exclude_16:
            subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]
        subject_key_to_pred_segments = get_segments_from_predictions(
            experiment_dir,
            subject_keys,
            best_threshold_segment_merging_val,
            filter_method,
            k_size,
        )

        # visualize pred lengths distribution
        all_seizure_durations = [
            seizure['end'] - seizure['start']
            for subject_key in subject_key_to_pred_segments.keys()
            for seizure in subject_key_to_pred_segments[subject_key]['seizures']
        ]

        all_seizure_pred_durations = [
            seizure['end'] - seizure['start']
            for subject_key in subject_key_to_pred_segments.keys()
            for seizure in subject_key_to_pred_segments[subject_key]['seizures_pred']
        ]

        # save_json_path = r'D:\Study\asp\thesis\publications\unknown_2024\assets\durations.json'
        # if os.path.exists(save_json_path):
        #     with open(save_json_path, 'r') as f:
        #         durations_dict = json.load(f)
        # else:
        #     durations_dict = dict()
        #
        # with open(save_json_path, 'w') as f:
        #     # durations_dict['GT_test'] = all_seizure_durations
        #     durations_dict['Error-awareCNN_test'] = all_seizure_pred_durations
        #     json.dump(durations_dict, f)
        # exit()

        all_seizure_durations = np.array(all_seizure_durations)
        all_seizure_pred_durations = np.array(all_seizure_pred_durations)

        max_pred_duration = max(all_seizure_pred_durations)
        max_true_duration = max(all_seizure_durations)
        print(f'max_pred_duration = {max_pred_duration:.4f} max_true_duration= {max_true_duration:.4f}')
        print(f'len(all_seizure_durations) = {len(all_seizure_durations)}')
        print(f'len(all_seizure_pred_durations) = {len(all_seizure_pred_durations)}')

        # import matplotlib.pyplot as plt
        # bins = np.linspace(10, 300, 25)
        # step = bins[1] - bins[0]
        # print(step)
        # # bins = np.linspace(10, max_pred_duration, int((max_pred_duration - 10) / step + 1))
        # bins = [10 + i * step for i in range(int((max_pred_duration + 50) / step) + 1)]
        #
        # density = False
        # plt.hist(all_seizure_durations, color='green', alpha=0.5, bins=bins, label='true', density=density)
        # plt.axvline(x=all_seizure_durations.mean(), color='green', linewidth=2)
        #
        # plt.hist(all_seizure_pred_durations, color='red', alpha=0.5, bins=bins, label='pred', density=density)
        # plt.axvline(x=all_seizure_pred_durations.mean(), color='red', linewidth=2)
        #
        # plt.title('Distribution of seizure duration', fontsize=18)
        # plt.xlabel('Seizure duration, sec', fontsize=18)
        # plt.ylabel('P' if density else 'amount', fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # # plt.show()
        # # exit()

        tp_num, fp_num, fn_num = 0, 0, 0
        for subject_idx, subject_key in enumerate(subject_key_to_pred_segments.keys()):
            print(f'{subject_idx}/{len(subject_key_to_pred_segments)} subject_key = {subject_key}')

            # if subject_idx < 30:  # baseline
            if subject_idx < 31:
                continue

            # if subject_key not in [
            #     'data1/dataset3',
            # ]:
            #     continue

            # if subject_key not in [
            #     'data1/dataset1',
            #     'data2/022tl Anonim-20201210_132636-20211122_172649',
            #     'data1/dataset12',
            # ]:
            #     continue

            # 14 failed  # in NN + OCSVM
            # if subject_idx < 32:
            #     continue

            # if subject_key != 'data1/dataset12':
            #     continue

            # if subject_key != 'data2/026tl Anonim-20210301_013744-20211122_174658':
            #     continue

            # if subject_key != 'data2/038tl Anonim-20190822_131550-20211123_005257':
            #     continue

            segments_dict = subject_key_to_pred_segments[subject_key]
            tp_segments, fp_segments, fn_segments = get_tp_fp_fn_segments(
                segments_dict['seizures'],
                segments_dict['normals'],
                segments_dict['seizures_pred'],
                segments_dict['normals_pred'],
                intersection_part_threshold=intersection_part_threshold,
                record_duration=segments_dict['record_duration'],
                seizure_segments_true_dilation=seizure_segments_true_dilation,
            )
            tp_num = tp_num + len(tp_segments)
            fp_num = fp_num + len(fp_segments)
            fn_num = fn_num + len(fn_segments)

            print(f'subject_key = {subject_key} tp_num = {len(tp_segments)} fp_num = {len(fp_segments)} fn_num = {len(fn_segments)}')
            # continue

            try:
                subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))
                # visualize_predicted_segments(subject_eeg_path, segments_dict['seizures'], segments_dict['seizures_pred'], intersection_part_threshold)
                visualize_predicted_segments_v2(
                    experiment_dir,
                    subject_eeg_path,
                    tp_segments,
                    fp_segments,
                    fn_segments,
                    seizure_segments_true_dilation,
                    save_dir,
                )
            except Exception as e:
                # print(f'Smth went wrong with {subject_key}')
                # print(traceback.format_exc())
                # print('\n\n\n')
                gc.collect()
                torch.cuda.empty_cache()
                raise e
            # exit(0)
        print('Visuazlization finished\n\n\n')
        print(f'tp_num = {tp_num} fp_num = {fp_num} fn_num = {fn_num}')
