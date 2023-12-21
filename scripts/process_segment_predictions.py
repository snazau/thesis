import os
import pickle

import numpy as np

from utils.common import filter_predictions, calc_segments_metrics, overlapping_segment
from scripts.visualize_errors import visualize_samples


def get_segments(prediction_data, threshold, filter_method='median', k_size=7, sfreq=128):
    normal_segments, seizure_segments = list(), list()

    # get giltered probs
    time_idxs_start = prediction_data['time_idxs_start']
    time_idxs_end = prediction_data['time_idxs_end']
    probs = prediction_data['probs_wo_tta'] if 'probs_wo_tta' in prediction_data else prediction_data['probs']

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
    while idx < len(preds):
        segment_start_time = time_idxs_start[idx] / sfreq
        segment_type = preds[idx]
        while idx < len(preds) and preds[idx] == segment_type:
            idx += 1
            segment_len += 1
        segment_end_time = time_idxs_end[idx - 1] / sfreq if idx != (len(preds) - 1) else time_idxs_end[idx]

        segments.append({
            'merged_segments_num': segment_len,
            'start': segment_start_time,
            'end': segment_end_time,
            'seizure_segment': bool(segment_type),
        })
        segment_len = 0

        segment = {
            'start': segment_start_time,
            'end': segment_end_time,
        }
        if segment_type:
            seizure_segments.append(segment)
        else:
            normal_segments.append(segment)

    # merge stochastic segments
    idx = 0
    segments_wo_stochastic = list()
    while idx < len(segments):
        segment_start_idx = idx
        while idx < len(segments) and (segments[idx]['merged_segments_num'] <= 2 or segments[idx]['seizure_segment']):
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
            }
            segments_wo_stochastic.append(merged_stochastic_segment)

    normal_segments = [segment for segment in segments_wo_stochastic if not segment['seizure_segment']]
    seizure_segments = [segment for segment in segments_wo_stochastic if segment['seizure_segment']]

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
            'seizures':  prediction_data['subject_seizures'],
            'normals':  normal_segments,
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


if __name__ == '__main__':
    # parameters
    experiment_name = '20231012_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded'
    # experiment_name = '20231005_EEGResNetCustomRaw_MixUp_TimeSeriesAug_raw_16excluded'
    # experiment_name = '20231005_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded_wo_baseline_correction'
    experiment_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name)

    visualize_segments = False
    exclude_16 = True

    # filter_method = None
    # k_size = -1
    filter_method = 'median'
    k_size = 7

    print(experiment_name)
    print(f'filter={filter_method} k={k_size} exclude_16={exclude_16}')
    print()

    # find best threshold
    subject_keys = [
        # stage_1 train
        'data2/011tl Anonim-20200118_041022-20211122_161616',
        'data2/016tl Anonim-20210127_214910-20211122_162657',
        'data2/022tl Anonim-20201209_132645-20211122_172422',
        'data2/002tl Anonim-20200826_044516-20211122_135439',
        'data2/004tl Anonim-20200929_081036-20211122_144552', 'data1/dataset29',
        'data2/026tl Anonim-20210227_214223-20211122_174442', 'data1/dataset19', 'data1/dataset21',
        'data2/020tl Anonim-20201218_194126-20211122_171755',
        'data2/034tl Anonim-20210304_071124-20211122_222211', 'data1/dataset17',
        'data2/033tl Anonim-20200114_085935-20211122_180917', 'data1/dataset10',
        'data1/dataset9',
        'data2/028tl Anonim-20191014_212520-20211122_175854',
        'data2/030tl Anonim-20190910_110631-20211122_180335',
        'data2/016tl Anonim-20210128_054911-20211122_163013',
        'data2/023tl Anonim-20210110_080440-20211122_173058',
        'data1/dataset15',
        'data2/008tl Anonim-20210204_211327-20211122_160546',
        'data2/037tl Anonim-20191020_110036-20211122_223805', 'data1/dataset7',
        'data2/006tl Anonim-20210209_144403-20211122_155146',
        'data2/006tl Anonim-20210208_144401-20211122_154504',
        # 'data2/009tl Anonim-20200215_021624-20211122_161231',  # 1000 sec seizure in the end
        'data2/035tl Anonim-20210326_231343-20211122_223404',
        'data2/009tl Anonim-20200213_130213-20211122_160907',
        'data2/041tl Anonim-20201112_194437-20211123_010804', 'data1/dataset26',
        'data2/018tl Anonim-20201212_101651-20211122_163821',
        'data2/037tl Anonim-20201102_102725-20211123_003801',
        # 'data2/004tl Anonim-20200926_213911-20211122_144051',  # 1000 sec seizure in the end
        'data1/dataset8',
        'data1/dataset18',
        'data2/026tl Anonim-20210302_093747-20211122_175031',
    ]
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
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    from scripts.find_threshold import get_best_threshold
    best_threshold, best_metric_meter = get_best_threshold(
        experiment_dir,
        subject_keys,
        filter_method,
        k_size,
        verbose=1,
    )
    print(f'\nbest_threshold = {best_threshold} best_metric_meter:\n{best_metric_meter}')
    print()

    # get segments pred
    subject_keys = [
        # 'data2/038tl Anonim-20190821_113559-20211123_004935'

        # # 'data2/038tl Anonim-20190821_113559-20211123_004935'
        # # 'data2/008tl Anonim-20210204_131328-20211122_160417'
        #
        # 'data2/018tl Anonim-20201211_130036-20211122_163611',
        # 'data2/006tl Anonim-20210208_063816-20211122_154113',
        # 'data1/dataset20',

        # stage_1
        # part1
        'data2/038tl Anonim-20190821_113559-20211123_004935',  # val
        'data2/027tl Anonim-20200309_195746-20211122_175315',  # val
        'data1/dataset27',  # val
        'data1/dataset14',  # val
        'data2/036tl Anonim-20201224_124349-20211122_181415',  # val
        'data2/041tl Anonim-20201115_222025-20211123_011114',  # val
        'data1/dataset24',  # val
        'data2/026tl Anonim-20210301_013744-20211122_174658',  # val

        'data2/020tl Anonim-20201218_071731-20211122_171454', 'data1/dataset13',
        'data2/018tl Anonim-20201211_130036-20211122_163611',
        'data2/038tl Anonim-20190822_155119-20211123_005457',
        'data2/025tl Anonim-20210128_233211-20211122_173425',
        'data2/015tl Anonim-20201116_134129-20211122_161958',

        # part2
        'data1/dataset3',
        'data2/027tl Anonim-20200310_035747-20211122_175503',
        'data2/002tl Anonim-20200826_124513-20211122_135804', 'data1/dataset23',
        'data2/022tl Anonim-20201210_132636-20211122_172649',
        'data1/dataset6', 'data1/dataset11',
        'data2/021tl Anonim-20201223_085255-20211122_172126', 'data1/dataset28',

        # part3
        'data2/008tl Anonim-20210204_131328-20211122_160417',
        'data2/003tl Anonim-20200831_120629-20211122_140327',
        'data2/025tl Anonim-20210129_073208-20211122_173728',
        'data2/038tl Anonim-20190822_131550-20211123_005257', 'data1/dataset2',

        'data1/dataset22',
        'data2/040tl Anonim-20200421_100248-20211123_010147',
        'data2/020tl Anonim-20201216_073813-20211122_171341',
        'data2/019tl Anonim-20201213_072025-20211122_165918',

        'data2/003tl Anonim-20200831_040629-20211122_135924',
        'data2/006tl Anonim-20210208_063816-20211122_154113', 'data1/dataset4', 'data1/dataset20',
        'data2/035tl Anonim-20210324_231349-20211122_223059', 'data1/dataset16',
        'data2/035tl Anonim-20210324_151211-20211122_222545',
        'data2/038tl Anonim-20190822_203419-20211123_005705', 'data1/dataset25', 'data1/dataset5',
        'data2/018tl Anonim-20201215_022951-20211122_165644',
        'data1/dataset1',
        'data1/dataset12',

        # # stage_2
        # 'data2/003tl Anonim-20200831_120629-20211122_140327',
        # 'data1/dataset12',
        # 'data2/025tl Anonim-20210129_073208-20211122_173728',
        # 'data2/038tl Anonim-20190822_131550-20211123_005257', 'data1/dataset2',
        # 'data1/dataset22',
        # 'data2/040tl Anonim-20200421_100248-20211123_010147',
        # 'data2/020tl Anonim-20201216_073813-20211122_171341',
        # 'data2/019tl Anonim-20201213_072025-20211122_165918',
        # 'data2/003tl Anonim-20200831_040629-20211122_135924',
        # 'data2/006tl Anonim-20210208_063816-20211122_154113', 'data1/dataset4', 'data1/dataset20',
        # 'data2/035tl Anonim-20210324_231349-20211122_223059', 'data1/dataset16',
        # 'data2/035tl Anonim-20210324_151211-20211122_222545',
        # 'data2/038tl Anonim-20190822_203419-20211123_005705', 'data1/dataset25', 'data1/dataset5',
        # 'data2/018tl Anonim-20201215_022951-20211122_165644',
    ]
    if exclude_16:
        subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]
    subject_key_to_pred_segments = get_segments_from_predictions(
        experiment_dir,
        subject_keys,
        best_threshold,
        filter_method,
        k_size,
    )
    # import pprint
    # pprint.pprint(subject_key_to_pred_segments)

    # calculate metric for segments
    import utils.avg_meters
    metric_meter = utils.avg_meters.MetricMeter()
    for subject_idx, subject_key in enumerate(subject_key_to_pred_segments.keys()):
        segments_dict = subject_key_to_pred_segments[subject_key]
        metrics_dict = calc_segments_metrics(
            segments_dict['seizures'],
            segments_dict['normals'],
            segments_dict['seizures_pred'],
            segments_dict['normals_pred'],
            intersection_part_threshold=0.51,
        )
        metric_meter.update(metrics_dict)
        print(f'#{subject_idx + 1:02} subject_key = {subject_key:60} subject_metrics {" ".join([f"{key} = {value:9.4f}" for key, value in metrics_dict.items()])}')
    print('metric_meter\n', metric_meter)

    # visualize segments
    if visualize_segments:
        data_dir = r'D:\Study\asp\thesis\implementation\data'

        save_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name, 'visualizations_segment')
        os.makedirs(save_dir, exist_ok=True)

        import json

        dataset_info_path = r'D:\Study\asp\thesis\implementation\data\dataset_info.json'
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        for subject_idx, subject_key in enumerate(subject_key_to_pred_segments.keys()):
            print(f'{subject_idx}/{len(subject_key_to_pred_segments)} subject_key = {subject_key}')

            segments_dict = subject_key_to_pred_segments[subject_key]

            try:
                subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))
                visualize_predicted_segments(subject_eeg_path, segments_dict['seizures'], segments_dict['seizures_pred'], intersection_part_threshold)
            except Exception as e:
                import traceback

                print(f'Smth went wrong with {subject_key}')
                print(traceback.format_exc())
                print('\n\n\n')
            # exit(0)
        print('Visuazlization finished\n\n\n')
