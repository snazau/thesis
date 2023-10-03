import os
import pickle

import cv2
import mne
import numpy as np

import eeg_reader
import datasets.datasets_static
import torch
import utils.neural.training
import visualization

import scripts.visualize_errors

experiments_dir = r'D:\Study\asp\thesis\implementation\experiments'
experiment_name = '08082023_efficientnet_b0_all_subjects_MixUp_TimeSeriesAug_raw'
experiment_dir = os.path.join(experiments_dir, experiment_name)
checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best.pth.tar')

state_dict = torch.load(checkpoint_path)['model']['state_dict']
model = utils.neural.training.get_model('efficientnet_b0_1channel', model_kwargs=dict(pretrained=True))
model.load_state_dict(state_dict)


def visualize_samples_raw(samples, probs, channel_names, sfreq, baseline_mean, set_name, subject_key, visualization_dir):
    # samples.shape = (1, C, T)
    set_dir = os.path.join(visualization_dir, set_name, 'raw')
    os.makedirs(set_dir, exist_ok=True)
    for idx in range(samples.shape[0]):
        tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(samples[idx]), dim=0), dim=0).float()
        heatmap = model.interpret(tensor)
        heatmap = heatmap[0, 0].detach().cpu().numpy()

        visualization.visualize_raw(
            samples[idx],
            channel_names,
            heatmap=heatmap,
            save_path=os.path.join(set_dir, subject_key, f'{set_name}_with_heatmap.png'),
        )

        visualization_path = os.path.join(
            set_dir,
            subject_key,
            f'p={probs[idx]:7.6f}_{idx:04}_{subject_key}.png'
        )
        # cv2.imwrite(visualization_path, (power_spectrum_visualization * 255).astype(np.uint8))
        print(f'Saved {visualization_path}')


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

    # FP visualization
    best_fp_time_idx_start, best_fp_prob = scripts.visualize_errors.find_best_fp(
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

        visualize_samples_raw(
            fp_samples,
            fp_probs,
            channel_names,
            sfreq,
            baseline_mean,
            set_name='fp',
            subject_key=subject_key,
            visualization_dir=visualization_dir,
        )

    # FN visualization
    best_fn_time_idx_start, best_fn_prob = scripts.visualize_errors.find_best_fn(
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

        visualize_samples_raw(
            fn_samples,
            fn_probs,
            channel_names,
            sfreq,
            baseline_mean,
            set_name='fn',
            subject_key=subject_key,
            visualization_dir=visualization_dir,
        )

    # TP visualization
    best_tp_time_idx_start, best_tp_prob = scripts.visualize_errors.find_best_tp(
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

        visualize_samples_raw(
            tp_samples,
            tp_probs,
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
    experiment_name = '08082023_efficientnet_b0_all_subjects_MixUp_TimeSeriesAug_raw'
    visualizations_dir = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\errors'
    os.makedirs(visualizations_dir, exist_ok=True)

    subject_keys = [
        # 'data2/027tl Anonim-20200309_195746-20211122_175315'
        # 'data2/038tl Anonim-20190821_113559-20211123_004935'
        # 'data2/008tl Anonim-20210204_131328-20211122_160417'

        # # part1
        'data2/038tl Anonim-20190821_113559-20211123_004935',  # val
        # 'data2/027tl Anonim-20200309_195746-20211122_175315',  # val
        # 'data1/dataset27',  # val
        # 'data1/dataset14',  # val
        'data2/036tl Anonim-20201224_124349-20211122_181415',  # val
        'data2/041tl Anonim-20201115_222025-20211123_011114',  # val
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
            save_errors(subject_key[6:], raw_data, prediction_data, threshold=0.5, channel_names=channel_names, sfreq=128, visualization_dir=set_visualizations_dir)
        except Exception as e:
            print(f'Smth went wrong with {subject_key}')
            print(f'Exception = {e}')
            raise e
        # break
