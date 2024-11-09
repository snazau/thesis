import os
import pickle
import random

import cv2
import mne
import numpy as np
import torch

import eeg_reader
import datasets.datasets_static
import utils.neural.training
import visualization
from utils.common import get_min_deviation_from_seizure

from models.resnet import EEGResNet18Spectrum, EEGResNet18Raw
from models.efficientnet import EEGEfficientNetB0Spectrum, EEGEfficientNetB0Raw


def find_best_fp(seizures, time_idxs_start, labels, probs, threshold, sfreq, min_start_time=900, min_deviation=30):
    """
    Finds FP prediction such that:
    (1) its not in the beginning (its start_time (sec) > min_start_time (secs))
    (2) its not close to seizure (at least min_deviation secs away from seizure)
    (3) it has highest prob among all FPs which satisfy (1) and (2)
    :return: idx of found FP, prob of found FP
    """

    fp_mask = ((labels == 0) & (probs > threshold))

    counter = 0
    best_fp_time_idx_start, best_fp_prob = -1, -1
    for sample_idx in range(len(fp_mask)):
        if not fp_mask[sample_idx]:
            continue

        counter += 1
        fp_prob = probs[sample_idx]
        fp_time = time_idxs_start[sample_idx] / sfreq
        min_fp_deviation = get_min_deviation_from_seizure(seizures, fp_time)
        if fp_prob > best_fp_prob and min_fp_deviation > min_deviation and fp_time > min_start_time:
            best_fp_time_idx_start = time_idxs_start[sample_idx]
            best_fp_prob = fp_prob

    return best_fp_time_idx_start, best_fp_prob


def find_best_fn(seizures, time_idxs_start, labels, probs, threshold, sfreq, min_start_time=900):
    """
    Finds FN prediction such that:
    (1) its not in the beginning (its start_time (sec) > min_start_time (secs))
    (2) it belongs to seizure segment (min_fn_deviation must be equal to -1)
    (3) it has lowest prob among all FNs which satisfy (1) and (2)
    :return: idx of found FN, prob of found FN
    """

    fn_mask = ((labels == 1) & (probs <= threshold))

    counter = 0
    best_fn_time_idx_start, best_fn_prob = -1, 1
    for sample_idx in range(len(fn_mask)):
        if not fn_mask[sample_idx]:
            continue

        counter += 1
        fn_prob = probs[sample_idx]
        fn_time = time_idxs_start[sample_idx] / sfreq
        min_fn_deviation = get_min_deviation_from_seizure(seizures, fn_time)
        if fn_prob < best_fn_prob and min_fn_deviation == -1 and fn_time > min_start_time:
            best_fn_time_idx_start = time_idxs_start[sample_idx]
            best_fn_prob = fn_prob

    return best_fn_time_idx_start, best_fn_prob


def find_best_tp(seizures, time_idxs_start, labels, probs, threshold, sfreq):
    """
    Finds TP prediction such that:
    (1) Its random for now
    :return: idx of found TP, prob of found TP
    """

    tp_mask = ((labels == 1) & (probs > threshold))

    tp_idxs = list()
    for sample_idx in range(len(tp_mask)):
        if not tp_mask[sample_idx]:
            continue

        tp_idxs.append(sample_idx)

    try:  # in case there is no TP => tp_idxs is empty => random choice will cause an error
        random_idx = random.choice(tp_idxs)
        best_tp_time_idx_start = time_idxs_start[random_idx]
        best_tp_prob = probs[random_idx]
    except Exception as e:
        best_tp_time_idx_start = -1
        best_tp_prob = -1

    return best_tp_time_idx_start, best_tp_prob


def select_most_important_channels_v1(channel_importance, important_num, channel_names):
    topk_channel_idxs_over_10sec = torch.topk(
        torch.from_numpy(channel_importance),
        k=important_num,
        dim=1
    ).indices  # (N_10, k)
    channel_idxs, channel_idxs_counts = np.unique(topk_channel_idxs_over_10sec.numpy().flatten(), return_counts=True)
    channel_idxs_counts_sort_idxs = np.argsort(-channel_idxs_counts)
    topk_channel_idxs_overall = channel_idxs[channel_idxs_counts_sort_idxs]
    channels_to_show = [
        channel_names[channel_idx].replace('EEG ', '')
        for channel_idx in topk_channel_idxs_overall[:important_num]
    ]
    channels_to_show = sorted(channels_to_show)
    return channels_to_show


def select_most_important_channels_v3(channel_importance, important_num, channel_names):
    # channel_importance.shape = (N_10, 25)

    # calc mean importance_score for each EEG channel over time of interest
    channel_importance_score = np.mean(channel_importance, axis=0)

    # sorted idxs from min importance_score to max importance_score
    channel_importance_score_sorted_idxs = np.argsort(channel_importance_score)

    # we are selecting channels with highest importance_score
    important_channel_idxs = channel_importance_score_sorted_idxs[-important_num:]

    channels_to_show = [
        channel_names[channel_idx].replace('EEG ', '')
        for channel_idx in important_channel_idxs
    ]
    channels_to_show = sorted(channels_to_show)
    return channels_to_show


def select_most_important_channels_v2(
        channel_importance,
        important_num,
        channel_names,
        prediction_type,
        segment_of_int_idx_start,
        segment_of_int_idx_end,
):
    # channel_importance.shape = (N_10, 25)

    prediction_type = prediction_type.lower().strip()
    assert prediction_type in ['tp', 'fp', 'fn']

    # select segments from segment_of_int_idx_start to segment_of_int_idx_end (inclusively)
    channel_importance_of_int = channel_importance[segment_of_int_idx_start:segment_of_int_idx_end + 1]  # (N_10_int, 25)

    # calc mean importance_score for each EEG channel over time of interest
    channel_importance_score = np.mean(channel_importance_of_int, axis=0)

    # sorted idxs from min importance_score to max importance_score
    channel_importance_score_sorted_idxs = np.argsort(channel_importance_score)

    if prediction_type == 'fn':
        # In case of FN prediction we are interested in those channels
        # which will increase the predicted prob if we change them a bit
        # so we are selecting channels with highest importance_score (mean deriv of pred w. r. t. channel)
        important_channel_idxs = channel_importance_score_sorted_idxs[-important_num:]
    else:
        # In case of TP or FP prediction we are interested in those channels
        # which will decrease the predicted prob if we change them a bit
        # so we are selecting channels with lowest importance_score (mean deriv of pred w. r. t. channel)
        important_channel_idxs = channel_importance_score_sorted_idxs[:important_num]

    channels_to_show = [
        channel_names[channel_idx].replace('EEG ', '')
        for channel_idx in important_channel_idxs
    ]
    channels_to_show = sorted(channels_to_show)

    return channels_to_show


def get_gradcam(
        power_spectrum,
        model_params,
        checkpoint_path,
        train_segment_duration_in_sec,
        segment_of_int_idx_start,
        segment_of_int_idx_end,
        sfreq,
        device,
):
    # power_spectrum.shape = (1, C, F, T)
    # T should be the multiple of (sfreq * train_segment_duration_in_sec)

    assert power_spectrum.shape[0] == 1

    # reshape power_spectrum into the 10sec segments as model input
    _, channels, freq_dim, time_dim = power_spectrum.shape[:4]
    # assert time_dim % (sfreq * train_segment_duration_in_sec) == 0

    batch_input = torch.split(power_spectrum, train_segment_duration_in_sec * sfreq, dim=3)  # list of (1, C, F, 1280)
    batch_input = torch.cat(batch_input, dim=0)  # (B, C, F, 1280)
    print(f'batch_input[0] = {batch_input[0].shape} len = {len(batch_input)}')

    # prepare input
    log = model_params['preprocessing_params']['log']
    baseline_correction = model_params['preprocessing_params']['baseline_correction']
    if log:
        batch_input = np.log(batch_input)
    elif baseline_correction:
        raise NotImplementedError

    normalization = model_params['preprocessing_params']['normalization']
    if normalization == 'minmax':
        batch_input = (batch_input - batch_input.min()) / (batch_input.max() - batch_input.min())
    elif normalization == 'meanstd':
        batch_input = (batch_input - batch_input.mean()) / batch_input.std()
    elif normalization == 'cwt_meanstd':
        raise NotImplementedError

    # load model
    model_class_name = model_params.get('class_name', 'EEGResNet18Spectrum')
    model_kwargs = model_params.get('kwargs', dict())
    model = globals()[model_class_name](**model_kwargs)

    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model']['state_dict']
    try:
        model.load_state_dict(state_dict)
    except Exception as e:  # might be bad 20241006
        new_state_dict = {f'model.{key}': value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # inference
    samples_num = batch_input.shape[0]
    batch_input = batch_input.to(device)

    with torch.no_grad():
        batch_outputs = torch.squeeze(model(batch_input))  # (B, )
        batch_probs = torch.sigmoid(batch_outputs)  # (B, )

    # calc Grad-CAM
    heatmaps, fmaps, grads = list(), list(), list()
    for sample_idx in range(samples_num):
        sample_heatmap, sample_fmaps, sample_grads = model.interpret(
            batch_input[sample_idx:sample_idx + 1], return_fm=True
        )  # (1, 1, F, 1280), (B, 512, F / 32, 40), (B, 512, F / 32, 40)
        # sample_heatmap = model.cameras(batch_input[sample_idx:sample_idx + 1])  # (1, 1, F, 1280)
        heatmaps.append(sample_heatmap)
        fmaps.append(sample_fmaps)
        grads.append(sample_grads)
    batch_heatmap = torch.cat(heatmaps, dim=0)  # (B, 1, F, 1280)
    batch_fmap = torch.cat(fmaps, dim=0)  # (B, 512, F / 32, 40)
    batch_grad = torch.cat(grads, dim=0)  # (B, 512, F / 32, 40)

    # calc channel importance by occlusion
    power_spectrum_batched = torch.cat(torch.split(power_spectrum, train_segment_duration_in_sec * sfreq, dim=3), dim=0)  # (B, C, F, 1280)
    occlusion_idx = torch.argmin(
        torch.sum(power_spectrum_batched[max(segment_of_int_idx_start - 6, 0):segment_of_int_idx_start], dim=(1, 2, 3))
    )
    occlusion_idx = occlusion_idx + max(segment_of_int_idx_start - 6, 0)
    occlusion = batch_input[occlusion_idx:occlusion_idx + 1]  # (1, C, F, 1280)

    channel_importance_occluded = torch.zeros((batch_input.shape[0], batch_input.shape[1]), dtype=torch.float32)
    with torch.no_grad():
        for channel_idx in range(batch_input.shape[1]):
            batch_input_occluded = batch_input[segment_of_int_idx_start:segment_of_int_idx_end + 1]  # (N_10_soi, C, F, 1280)
            batch_input_occluded[:, channel_idx] = occlusion[:, channel_idx]

            batch_outputs_occluded = torch.squeeze(model(batch_input_occluded))  # (N_10_soi, )
            batch_probs_occluded = torch.sigmoid(batch_outputs_occluded)  # (N_10_soi, )

            channel_importance_occluded[segment_of_int_idx_start:segment_of_int_idx_end + 1, channel_idx] = torch.abs(
                batch_probs[segment_of_int_idx_start:segment_of_int_idx_end + 1] - batch_probs_occluded,
            )

    # calc freq_importance - avg over time of Grad-CAM
    freq_importance = torch.mean(batch_heatmap, dim=3, keepdim=True)  # (B, 1, F, 1)
    freq_importance = torch.squeeze(freq_importance)  # (B, F)

    # calc channel_importance - derivate of a prediction with respect to 25 input channels
    dpred_dinput = model.channel_importance(batch_input)  # (B, 25, F, 1280)
    channel_importance = torch.sum(dpred_dinput, dim=(2, 3))  # (B, 25)

    # concat dpred_dinput for each 10sec segment along time dim
    dpred_dinput = torch.cat(
        [
            dpred_dinput[sample_idx:sample_idx + 1]
            for sample_idx in range(dpred_dinput.shape[0])
        ],
        dim=3,
    )  # (1, 25, F, B * 1280)
    print(f'dpred_dinput = {dpred_dinput.shape}')

    # concat Grad-CAM for each 10sec segment along time dim
    batch_heatmap = torch.cat(
        [
            batch_heatmap[sample_idx:sample_idx + 1]
            for sample_idx in range(batch_heatmap.shape[0])
        ],
        dim=3,
    )  # (1, 1, F, B * 1280)
    print(f'batch_heatmap = {batch_heatmap.shape}')

    batch_fmap = torch.cat(
        [
            batch_fmap[sample_idx:sample_idx + 1]
            for sample_idx in range(batch_fmap.shape[0])
        ],
        dim=3,
    )  # (1, 512, F, B * 40)
    print(f'batch_fmap = {batch_fmap.shape}')

    batch_grad = torch.cat(
        [
            batch_grad[sample_idx:sample_idx + 1]
            for sample_idx in range(batch_grad.shape[0])
        ],
        dim=3,
    )  # (1, 512, F, B * 40)
    print(f'batch_grad = {batch_grad.shape}')

    return (
        batch_probs.cpu().numpy(),
        batch_heatmap.cpu().numpy(),
        channel_importance.cpu().numpy(),
        channel_importance_occluded.cpu().numpy(),
        freq_importance.cpu().numpy(),
        dpred_dinput.cpu().numpy(),
        batch_fmap.cpu().numpy(),
        batch_grad.cpu().numpy(),
    )


def visualize_samples(
        samples,
        probs,
        time_starts,
        channel_names,
        sfreq,
        baseline_mean,
        set_name,
        subject_key,
        visualization_dir,
        checkpoint_path=None,
        model_params=None,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
):
    # samples.shape = (1, C, T)
    freqs = np.arange(1, 40.01, 0.1)

    print('channel_names', channel_names)
    channels_to_show = ['F3', 'F4', 'T5', 'T6', 'P3', 'P4', 'Cz']

    freq_ranges = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 30),
        'gamma': (30, 40),
    }
    channel_groups = {
        'frontal': {
            'channel_names': ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10']])
            ],
        },
        'central': {
            'channel_names': ['C3', 'Cz', 'C4'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['C3', 'Cz', 'C4']])
            ],
        },
        'perietal-occipital': {
            'channel_names': ['P3', 'Pz', 'P4', 'O1', 'O2'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['P3', 'Pz', 'P4', 'O1', 'O2']])
            ],
        },
        'temporal-left': {
            'channel_names': ['T9', 'T3', 'P9', 'T5'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['T9', 'T3', 'P9', 'T5']])
            ],
        },
        'temporal-right': {
            'channel_names': ['T10', 'T4', 'P10', 'T6'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['T10', 'T4', 'P10', 'T6']])
            ],
        },
    }

    # power_spectrum.shape = (1, C, F, T)
    power_spectrum = mne.time_frequency.tfr_array_morlet(
        samples,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=freqs,
        output='power',
        n_jobs=-1
    )

    segments_num = int(samples.shape[-1] / sfreq / 10)
    if set_name == 'fn':
        # idxs of segments with GT seizures
        segment_of_int_idx_start = int(seizure_times_list[0]['start'] / 10)
        segment_of_int_idx_end = int(seizure_times_list[0]['end'] / 10) - 1
    else:
        # idxs of segments with PRED seizures
        segment_of_int_idx_start = int(seizure_times_list[2]['start'] / 10)
        segment_of_int_idx_end = int(seizure_times_list[2]['end'] / 10) - 1
    segment_of_int_idx_end = min(segments_num - 1, segment_of_int_idx_end)

    heatmap = None
    pred_prob = None
    dpred_dinput, fmap, grad = None, None, None
    channel_importance, channel_importance_occluded, freq_importance = None, None, None
    importance_matrices, freq_range_names, channel_group_names = None, None, None
    if model_params is not None and checkpoint_path is not None and os.path.exists(checkpoint_path):
        pred_prob, heatmap, channel_importance, channel_importance_occluded, freq_importance, dpred_dinput, fmap, grad = get_gradcam(
            torch.from_numpy(power_spectrum).float(),
            model_params,
            checkpoint_path,
            train_segment_duration_in_sec=10,
            segment_of_int_idx_start=segment_of_int_idx_start,
            segment_of_int_idx_end=segment_of_int_idx_end,
            sfreq=128,
            device=torch.device('cpu'),
        )
        # N_10 = T // 1280
        # pred_prob.shape = (N_10, )
        # heatmap.shape = (1, 1, F, T)
        # channel_importance.shape = (N_10, 25)
        # channel_importance_occluded.shape = (N_10, 25)
        # freq_importance.shape = (N_10, F)
        # dpred_dinput.shape = (1, 25, F, T)
        # fmap.shape = (1, 512, F / 32, T / 32)
        # grad.shape = (1, 512, F / 32, T / 32)

        save_path = os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[0])}_gcam.npy')
        save_dict = {
            'fmap': fmap,
            'grad': grad,
            'segment_of_int_idx_start': segment_of_int_idx_start,
            'segment_of_int_idx_end': segment_of_int_idx_end,
            'segments_num': segments_num,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, save_dict)

        # # # most important channels V1
        # channels_to_show = select_most_important_channels_v1(
        #     channel_importance,
        #     important_num=7,
        #     channel_names=channel_names,
        # )
        # # # most important channels V1

        # # # # most important channels V2
        # channels_to_show = select_most_important_channels_v2(
        #     channel_importance,
        #     important_num=7,
        #     channel_names=channel_names,
        #     prediction_type=set_name,
        #     segment_of_int_idx_start=segment_of_int_idx_start,
        #     segment_of_int_idx_end=segment_of_int_idx_end,
        # )
        # # # # most important channels V2

        # # # most important channels V3
        channels_to_show = select_most_important_channels_v3(
            channel_importance_occluded,
            important_num=7,
            channel_names=channel_names,
        )
        # # # most important channels V3

        # heatmap = None
        channel_names.append('__heatmap__')
        channels_to_show.append('__heatmap__')

        importance_matrices, freq_range_names, channel_group_names = visualization.get_importance_matrices(
            heatmap[0], freq_ranges, channel_importance_occluded, channel_groups, segment_len_in_sec=10, sfreq=sfreq,
        )  # (N_10, 5, 5), list with row names, list with col names

    set_dir = os.path.join(visualization_dir, set_name, 'power_specturm')
    os.makedirs(set_dir, exist_ok=True)
    for idx in range(power_spectrum.shape[0]):
        # save to csv
        if channel_importance is not None:
            csv_data = list()
            for segment_of_int_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1):
                data = {
                    'prob': pred_prob[segment_of_int_idx],
                    'start_rel': segment_of_int_idx * 10,
                    'end_rel': segment_of_int_idx * 10 + 10,
                    'start_abs': time_starts[idx] + segment_of_int_idx * 10,
                    'end_abs': time_starts[idx] + segment_of_int_idx * 10 + 10,
                    'importance_matrix': importance_matrices[segment_of_int_idx],
                }

                for freq_idx in range(freq_importance.shape[1]):
                    name = f'freq_{freqs[freq_idx]:.1f}_importance'
                    value = freq_importance[segment_of_int_idx, freq_idx]
                    data[name] = value

                for channel_idx in range(channel_importance.shape[1]):
                    name = f'{channel_names[channel_idx].replace("EEG ", "")}_importance'
                    value = channel_importance[segment_of_int_idx, channel_idx]
                    data[name] = value

                csv_data.append(data)

            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}_stats.csv')
            df.to_csv(csv_path, index=False)

        power_spectrum_corrected = (power_spectrum[idx] - baseline_mean) / baseline_mean

        # power_spectrum_visualization = visualization.visualize_spectrum_channels(
        #     power_spectrum_corrected,
        #     channel_names,
        # )
        # visualization.visualize_raw_with_spectrum_data(
        #     power_spectrum_corrected,
        #     samples[idx],
        #     channel_names,
        #     save_dir=os.path.join(set_dir, subject_key),
        #     baseline_correction=True,
        # )
        # visualization.visualize_raw(
        #     samples[idx],
        #     channel_names,
        #     # save_dir=os.path.join(set_dir, subject_key),  # TODO: change save_dir to save_path
        #     save_path=os.path.join(set_dir, subject_key, f'raw.png')
        # )

        # visualization.visualize_raw_with_spectrum_data_v2(
        #     power_spectrum_corrected,
        #     samples[idx],
        #     channel_names,
        #     heatmap=heatmap[idx] if heatmap is not None else None,
        #     preds=pred_prob if pred_prob is not None else None,
        #     save_path=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}.png'),
        #     sfreq=128,
        #     time_shift=time_starts[idx],
        #     channels_to_show=channels_to_show,
        #     seizure_times_list=seizure_times_list,
        #     seizure_times_colors=seizure_times_colors,
        #     seizure_times_ls=seizure_times_ls,
        #     max_spectrum_value=100,
        # )

        visualization.visualize_raw_with_spectrum_data_v5(
            freq_ranges,
            channel_groups,
            power_spectrum_corrected,
            # power_spectrum[idx],
            samples[idx],
            heatmap[idx] if heatmap is not None else None,
            channel_importance_occluded,
            channel_names=channel_names,
            channels_to_show=channels_to_show,
            segment_of_int_idx_start=segment_of_int_idx_start,
            segment_of_int_idx_end=segment_of_int_idx_end,
            save_path=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}_V5.png'),
            sfreq=128,
            time_shift=time_starts[idx],
            seizure_times_list=seizure_times_list,
            seizure_times_colors=seizure_times_colors,
            seizure_times_ls=seizure_times_ls,
            max_spectrum_value=100,
            min_importance_value=0.75,
            min_importance_matrix_value='min',
            max_importance_matrix_value='max',
        )

        # visualization.visualize_raw_with_spectrum_data_v4(
        #     power_spectrum_corrected,
        #     # power_spectrum[idx],
        #     samples[idx],
        #     heatmap[idx] if heatmap is not None else None,
        #     channel_importance_occluded,
        #     channel_names=channel_names,
        #     channels_to_show=channels_to_show,
        #     segment_of_int_idx_start=segment_of_int_idx_start,
        #     segment_of_int_idx_end=segment_of_int_idx_end,
        #     save_path=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}_V4.png'),
        #     sfreq=128,
        #     time_shift=time_starts[idx],
        #     seizure_times_list=seizure_times_list,
        #     seizure_times_colors=seizure_times_colors,
        #     seizure_times_ls=seizure_times_ls,
        #     max_spectrum_value=100,
        #     min_importance_value=0.75,
        # )

        # visualization.visualize_raw_with_spectrum_data_v3(
        #     power_spectrum_corrected,
        #     # power_spectrum[idx],
        #     samples[idx],
        #     heatmap[idx] if heatmap is not None else None,
        #     channel_names=channel_names,
        #     channels_to_show=channels_to_show,
        #     segment_of_int_idx_start=segment_of_int_idx_start,
        #     segment_of_int_idx_end=segment_of_int_idx_end,
        #     save_path=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}_V3.png'),
        #     sfreq=128,
        #     time_shift=time_starts[idx],
        #     seizure_times_list=seizure_times_list,
        #     seizure_times_colors=seizure_times_colors,
        #     seizure_times_ls=seizure_times_ls,
        #     max_spectrum_value=100,
        # )

        # if dpred_dinput is not None:
        #     visualization.visualize_raw_with_spectrum_data_v2(
        #         dpred_dinput[idx],
        #         samples[idx].copy(),
        #         channel_names,
        #         heatmap=heatmap[idx] if heatmap is not None else None,
        #         preds=pred_prob if pred_prob is not None else None,
        #         save_path=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}_dpred_dinput.png'),
        #         sfreq=128,
        #         time_shift=time_starts[idx],
        #         channels_to_show=channels_to_show,
        #         seizure_times_list=seizure_times_list,
        #         seizure_times_colors=seizure_times_colors,
        #         seizure_times_ls=seizure_times_ls,
        #     )

        # if fmap is not None:
        #     visualization.visualize_feature_maps(
        #         fmap[idx],
        #         original_resolution=power_spectrum_corrected.shape[-2:],
        #         save_dir=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}'),
        #         sfreq=128,
        #         time_shift=time_starts[idx],
        #         seizure_times_list=seizure_times_list,
        #         seizure_times_colors=seizure_times_colors,
        #         seizure_times_ls=seizure_times_ls,
        #     )

        # if channel_importance is not None:
        #     visualization.visualize_channel_importance(
        #         channel_importance,
        #         time_starts[idx],
        #         channel_names[:-1] if heatmap is not None else channel_names,  # exclude '__heatmap__' if needed
        #         time_step_sec=10,
        #         save_path=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(probs[idx])}_CI.png'),
        #     )

        # visualization_path = os.path.join(
        #     set_dir,
        #     subject_key,
        #     f'p={probs[idx]:7.6f}_{idx:04}_{subject_key}.png'
        # )
        # cv2.imwrite(visualization_path, (power_spectrum_visualization * 255).astype(np.uint8))
        # print(f'Saved {visualization_path}')


def save_errors(subject_key, raw_data, prediction_data, threshold, channel_names, sfreq, visualization_dir):
    # get data from prediction_data
    time_idxs_start = prediction_data['time_idxs_start']
    time_idxs_end = prediction_data['time_idxs_end']
    seizures = prediction_data['subject_seizures']
    probs = prediction_data['probs_wo_tta']
    labels = prediction_data['labels']

    # baseline correction preparation
    freqs = np.arange(1, 40.01, 0.1)
    baseline_mean, baseline_std = datasets.datasets_static.get_baseline_stats(
        raw_data,
        baseline_length_in_seconds=500,
        sfreq=raw_data.info['sfreq'],
        freqs=freqs,
    )

    # # REMOVE ME
    # tp_start_times = [17550.0]
    # tp_probs = [1.0]
    # tp_samples, _, _ = datasets.datasets_static.generate_raw_samples(raw_data, tp_start_times, sample_duration=10)
    #
    # visualize_samples(
    #     tp_samples,
    #     tp_probs,
    #     tp_start_times,
    #     channel_names,
    #     sfreq,
    #     baseline_mean,
    #     set_name='tp',
    #     subject_key=subject_key,
    #     visualization_dir=visualization_dir,
    # )
    # exit()
    # # REMOVE ME

    # FP visualization
    best_fp_time_idx_start, best_fp_prob = find_best_fp(
        seizures,
        time_idxs_start,
        labels,
        probs,
        threshold,
        sfreq,
        min_start_time=900,
        min_deviation=30,
    )

    if best_fp_time_idx_start != -1:
        fp_start_times = [best_fp_time_idx_start / sfreq]
        fp_probs = [best_fp_prob]
        fp_samples, _, _ = datasets.datasets_static.generate_raw_samples(raw_data, fp_start_times, sample_duration=10)

        visualize_samples(
            fp_samples,
            fp_probs,
            fp_start_times,
            channel_names,
            sfreq,
            baseline_mean,
            set_name='fp',
            subject_key=subject_key,
            visualization_dir=visualization_dir,
        )

    # FN visualization
    best_fn_time_idx_start, best_fn_prob = find_best_fn(
        seizures,
        time_idxs_start,
        labels,
        probs,
        threshold,
        sfreq,
        min_start_time=900,
    )

    if best_fn_time_idx_start != -1:
        fn_start_times = [best_fn_time_idx_start / sfreq]
        fn_probs = [best_fn_prob]
        fn_samples, _, _ = datasets.datasets_static.generate_raw_samples(raw_data, fn_start_times, sample_duration=10)

        visualize_samples(
            fn_samples,
            fn_probs,
            fn_start_times,
            channel_names,
            sfreq,
            baseline_mean,
            set_name='fn',
            subject_key=subject_key,
            visualization_dir=visualization_dir,
        )

    # TP visualization
    best_tp_time_idx_start, best_tp_prob = find_best_tp(
        seizures,
        time_idxs_start,
        labels,
        probs,
        threshold,
        sfreq,
    )

    if best_tp_time_idx_start != -1:
        tp_start_times = [best_tp_time_idx_start / sfreq]
        tp_probs = [best_tp_prob]
        tp_samples, _, _ = datasets.datasets_static.generate_raw_samples(raw_data, tp_start_times, sample_duration=10)

        visualize_samples(
            tp_samples,
            tp_probs,
            tp_start_times,
            channel_names,
            sfreq,
            baseline_mean,
            set_name='tp',
            subject_key=subject_key,
            visualization_dir=visualization_dir,
        )


if __name__ == '__main__':
    utils.neural.training.set_seed(8, deterministic=True)

    data_dir = r'D:\Study\asp\thesis\implementation\data'
    experiment_name = 'renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug'
    # visualizations_dir = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\errors_v4_relchange_optimal_cwt_cmp'
    # visualizations_dir = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\errors_paper_review2'
    visualizations_dir = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\errors_v4_reconstruction_20231030'
    os.makedirs(visualizations_dir, exist_ok=True)

    subject_keys = [
        'data2/027tl Anonim-20200309_195746-20211122_175315'
        # 'data2/038tl Anonim-20190821_113559-20211123_004935'
        # 'data2/008tl Anonim-20210204_131328-20211122_160417'

        # # part1
        # 'data2/038tl Anonim-20190821_113559-20211123_004935',  # val
        # 'data2/027tl Anonim-20200309_195746-20211122_175315',  # val
        # 'data1/dataset27',  # val
        # 'data1/dataset14',  # val
        # 'data2/036tl Anonim-20201224_124349-20211122_181415',  # val
        # 'data2/041tl Anonim-20201115_222025-20211123_011114',  # val
        # # 'data1/dataset24',  # val
        # 'data2/026tl Anonim-20210301_013744-20211122_174658',  # val

        # 'data2/020tl Anonim-20201218_071731-20211122_171454', 'data1/dataset13',
        # 'data2/018tl Anonim-20201211_130036-20211122_163611',
        # 'data2/038tl Anonim-20190822_155119-20211123_005457',
        # 'data2/025tl Anonim-20210128_233211-20211122_173425',
        # 'data2/015tl Anonim-20201116_134129-20211122_161958',

        # # part2
        # 'data1/dataset3',
        # 'data2/027tl Anonim-20200310_035747-20211122_175503',
        # 'data2/002tl Anonim-20200826_124513-20211122_135804', 'data1/dataset23',
        # 'data2/022tl Anonim-20201210_132636-20211122_172649', 'data1/dataset6', 'data1/dataset11',
        # 'data2/021tl Anonim-20201223_085255-20211122_172126', 'data1/dataset28',
        #
        # # part3
        # 'data1/dataset1',
        # 'data2/008tl Anonim-20210204_131328-20211122_160417',
        # 'data2/003tl Anonim-20200831_120629-20211122_140327', 'data1/dataset12',
        # 'data2/025tl Anonim-20210129_073208-20211122_173728',
        # 'data2/038tl Anonim-20190822_131550-20211123_005257', 'data1/dataset2',
        #
        # 'data1/dataset22',
        # 'data2/040tl Anonim-20200421_100248-20211123_010147',
        # 'data2/020tl Anonim-20201216_073813-20211122_171341',
        # 'data2/019tl Anonim-20201213_072025-20211122_165918',
        #
        # 'data2/003tl Anonim-20200831_040629-20211122_135924',
        # 'data2/006tl Anonim-20210208_063816-20211122_154113', 'data1/dataset4', 'data1/dataset20',
        # 'data2/035tl Anonim-20210324_231349-20211122_223059', 'data1/dataset16',
        # 'data2/035tl Anonim-20210324_151211-20211122_222545',
        # 'data2/038tl Anonim-20190822_203419-20211123_005705', 'data1/dataset25', 'data1/dataset5',
        # 'data2/018tl Anonim-20201215_022951-20211122_165644',
    ]
    for subject_idx, subject_key in enumerate(subject_keys):
        print(f'Progress {subject_idx + 1}/{len(subject_keys)} {subject_key}')
        prediction_path = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\predictions\{subject_key}.pickle'
        prediction_data = pickle.load(open(prediction_path, 'rb'))

        eeg_file_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))
        raw_data = eeg_reader.EEGReader.read_eeg(eeg_file_path)
        datasets.datasets_static.drop_unused_channels(eeg_file_path, raw_data)
        channel_names = raw_data.info['ch_names']

        set_visualizations_dir = os.path.join(visualizations_dir)
        os.makedirs(set_visualizations_dir, exist_ok=True)

        try:
            save_errors(subject_key[6:], raw_data, prediction_data, threshold=0.95, channel_names=channel_names, sfreq=128, visualization_dir=set_visualizations_dir)
        except Exception as e:
            print(f'Smth went wrong with {subject_key}')
            print(f'Exception = {e}')
            # raise e
        # break
