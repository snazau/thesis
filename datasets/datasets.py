import os
import pickle

import mne
import numpy as np
import torch
from torch.utils.data._utils import collate

import custom_distribution
import eeg_reader
import scripts.visualize_errors

mne.set_log_level('error')


def check_if_in_segments(time_point, segments):
    return any([segment['start'] <= time_point <= segment['end'] for segment in segments])


def calc_time_to_closest_seizure(time_point, seizures):
    is_inside = False
    time_to_closest_seizure = 1e9
    for seizure in seizures:
        time_to_seizure = abs(time_point - seizure['start'])
        if time_to_seizure < time_to_closest_seizure:
            time_to_closest_seizure = time_to_seizure
            is_inside = seizure['start'] <= time_point <= seizure['end']

        time_to_seizure = abs(time_point - seizure['end'])
        if time_to_seizure < time_to_closest_seizure:
            time_to_closest_seizure = time_to_seizure
            is_inside = seizure['start'] <= time_point <= seizure['end']

    return time_to_closest_seizure, is_inside


def calc_signed_time_to_closest_seizure(time_point, seizures):
    time_to_closest_seizure, is_inside = calc_time_to_closest_seizure(time_point, seizures)
    time_to_closest_seizure = -abs(time_to_closest_seizure) if is_inside else abs(time_to_closest_seizure)
    return time_to_closest_seizure


def generate_raw_samples(raw_eeg, sample_start_times, sample_duration):
    # if mask is not None:
    #     my_annot = mne.Annotations(
    #         onset=[sample_start_time for sample_start_time in sample_start_times],
    #         duration=[sample_duration for _ in sample_start_times],
    #         description=[f'seizure' if is_seizure_sample else f'normal' for idx, is_seizure_sample in enumerate(mask)]
    #     )
    #     raw_eeg.set_annotations(my_annot)
    #     raw_eeg.plot()
    #     import matplotlib.pyplot as plt
    #     plt.show()

    # events, event_id = mne.events_from_annotations(self.raw)
    # epochs = mne.Epochs(self.raw, events, tmin=0, tmax=self.sample_duration, baseline=None, event_repeated='drop')
    # self.raw_samples = epochs.get_data()
    # self.raw_samples = self.raw_samples[:, :, :self.sample_duration * 128]

    raw_data = raw_eeg.get_data()

    samples_num = len(sample_start_times)
    frequency = raw_eeg.info['sfreq']
    sample_len_in_idxs = int(sample_duration * frequency)
    channels_num = len(raw_eeg.info['ch_names'])

    samples = np.zeros((samples_num, channels_num, sample_len_in_idxs))
    time_idxs_start = np.zeros((samples_num, ))
    time_idxs_end = np.zeros((samples_num, ))
    for sample_idx, sample_start_time in enumerate(sample_start_times):
        start_idx = int(frequency * sample_start_time)
        end_idx = start_idx + sample_len_in_idxs
        samples[sample_idx] = raw_data[:, start_idx:end_idx]
        time_idxs_start[sample_idx] = start_idx
        time_idxs_end[sample_idx] = end_idx

    return samples, time_idxs_start, time_idxs_end


def drop_unused_channels(eeg_file_path, raw_file):
    # drop unnecessary channels
    if 'data1' in eeg_file_path:
        channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
    elif 'data2' in eeg_file_path:
        channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
    else:
        raise NotImplementedError

    channels_num = len(raw_file.info['ch_names'])
    channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
    raw_file.drop_channels(channels_to_drop)


def extract_fp_times(prediction_data, threshold, sfreq):
    time_idxs_start = prediction_data['time_idxs_start']
    probs = prediction_data['probs_wo_tta']
    labels = prediction_data['labels']

    fp_mask = ((labels == 0) & (probs > threshold))
    fp_start_times = time_idxs_start[fp_mask] / sfreq

    return fp_start_times


def filter_fp_times(fp_start_times, seizures, min_deviation=30, min_start_time=900):
    acceptable_fp_start_times = list()
    for fp_start_time in fp_start_times:
        min_fp_deviation = scripts.visualize_errors.get_min_deviation_from_seizure(seizures, fp_start_time)
        if min_fp_deviation > min_deviation and fp_start_time > min_start_time:
            acceptable_fp_start_times.append(fp_start_time)
    acceptable_fp_start_times = np.array(acceptable_fp_start_times)
    return acceptable_fp_start_times


def extract_fn_times(prediction_data, threshold, sfreq):
    time_idxs_start = prediction_data['time_idxs_start']
    probs = prediction_data['probs_wo_tta']
    labels = prediction_data['labels']

    fn_mask = ((labels == 1) & (probs <= threshold))
    fn_start_times = time_idxs_start[fn_mask] / sfreq

    return fn_start_times


def filter_fn_times(fn_start_times, seizures, min_deviation=-1, min_start_time=900):
    acceptable_fn_start_times = list()
    for fn_start_time in fn_start_times:
        min_fn_deviation = scripts.visualize_errors.get_min_deviation_from_seizure(seizures, fn_start_time)
        if min_fn_deviation <= min_deviation and fn_start_time > min_start_time:
            acceptable_fn_start_times.append(fn_start_time)
    acceptable_fn_start_times = np.array(acceptable_fn_start_times)
    return acceptable_fn_start_times


def get_baseline_stats(raw_data, baseline_length_in_seconds=500, sfreq=128, freqs=np.arange(1, 40.01, 0.1)):
    min_time, max_time = raw_data.times.min(), raw_data.times.max()

    # sample_start_times = np.arange(min_time, max_time - baseline_length_in_seconds, baseline_length_in_seconds // 2)
    sample_start_times = np.arange(min_time, max_time - baseline_length_in_seconds, max(baseline_length_in_seconds, max_time // baseline_length_in_seconds))
    samples, _, _ = generate_raw_samples(
        raw_data,
        sample_start_times,
        baseline_length_in_seconds
    )

    samples_std = np.std(samples, axis=2)  # std over time
    samples_avg_std = np.mean(samples_std, axis=1)  # mean over channels
    baseline_idx = np.argmin(samples_avg_std)
    baseline_segment = samples[baseline_idx:baseline_idx + 1]

    baseline_power_spectrum = mne.time_frequency.tfr_array_morlet(
        baseline_segment,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=freqs,
        output='power',
        n_jobs=-1
    )
    baseline_mean = np.mean(baseline_power_spectrum[0], axis=2, keepdims=True)
    baseline_std = np.std(baseline_power_spectrum[0], axis=2, keepdims=True)

    return baseline_mean, baseline_std


class SubjectRandomDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            eeg_file_path,
            seizures,
            samples_num,
            prediction_data_path=None,
            stats_path=None,
            sample_duration=60,
            normal_samples_fraction=0.5,
            normalization=None,
            data_type='power_spectrum',
            baseline_correction=False,
            transform=None,
    ):
        self.eeg_file_path = eeg_file_path
        self.raw = eeg_reader.EEGReader.read_eeg(self.eeg_file_path)
        self.seizures = seizures
        self.samples_num = samples_num
        self.prediction_data = None if prediction_data_path is None else pickle.load(open(prediction_data_path, 'rb'))
        self.stats_data = None if stats_path is None else pickle.load(open(stats_path, 'rb'))
        self.sample_duration = sample_duration
        self.data_type = data_type
        self.normal_samples_fraction = normal_samples_fraction
        self.normalization = normalization
        self.baseline_correction = baseline_correction
        self.transform = transform

        # trim last seizure
        if self.seizures[-1]['end'] + self.sample_duration >= self.raw.times.max():
            print('Trimming last seizure')
            self.seizures[-1]['end'] = self.raw.times.max() - self.sample_duration - 1e-3

        # my_annot = mne.Annotations(
        #     onset=[seizure['start'] for seizure in self.seizures],
        #     duration=[seizure['end'] - seizure['start'] for seizure in self.seizures],
        #     description=[f'seizure' for seizure in self.seizures]
        # )
        # self.raw.set_annotations(my_annot)
        # self.raw.plot()
        # import matplotlib.pyplot as plt
        # plt.show()

        # drop unnecessary channels
        if 'data1' in self.eeg_file_path:
            channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
        elif 'data2' in self.eeg_file_path:
            channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
        else:
            raise NotImplementedError

        channels_num = len(self.raw.info['ch_names'])
        channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
        self.raw.drop_channels(channels_to_drop)

        self.channel_name_to_idx = {
            channel_name.replace('EEG ', ''): channel_idx for channel_idx, channel_name in enumerate(self.raw.info['ch_names'])
        }

        # calc stuff for baseline correction
        self.freqs = np.arange(1, 40.01, 0.1)
        if self.baseline_correction:
            self.baseline_mean, self.baseline_std = get_baseline_stats(
                self.raw,
                baseline_length_in_seconds=500,
                sfreq=self.raw.info['sfreq'],
                freqs=self.freqs,
            )
            self.baseline_mean = self.baseline_mean[np.newaxis]
            self.baseline_std = self.baseline_std[np.newaxis]
        else:
            self.baseline_mean = [0]
            self.baseline_std = [1]

        # set montage
        # print(self.raw.get_data().min(), self.raw.get_data().mean(), self.raw.get_data().max())
        # montage = mne.channels.make_standard_montage('standard_1020')
        # self.raw.set_montage(montage, on_missing='ignore', match_alias={ch: ch.replace('EEG ', '').strip() for ch in self.raw.info['ch_names']})
        # self.raw.plot_sensors(kind='topomap');
        # import matplotlib.pyplot as plt
        # plt.show()
        # exit()

        # generate normal_segments using seizures and time limits of eeg file
        time_start, time_end = self.raw.times.min(), self.raw.times.max()
        time_points = [time_start]
        for seizure in self.seizures:
            assert seizure['start'] > time_points[-1]
            time_points.append(seizure['start'])
            time_points.append(seizure['end'])
        time_points.append(time_end - self.sample_duration)
        assert len(time_points) % 2 == 0

        self.normal_segments = list()
        for normal_segment_idx in range(len(time_points) // 2):
            self.normal_segments.append({
                'start': time_points[normal_segment_idx * 2],
                'end': time_points[normal_segment_idx * 2 + 1],
            })

        # print(f'seizures = {self.seizures}')
        # print(f'normal_segments = {self.normal_segments}')

        (
            self.mask,
            self.sample_start_times,
            self.targets,
            self.raw_samples,
            self.time_idxs_start,
            self.time_idxs_end,
            self.times_to_closest_seizure
        ) = self._generate_data()
        # print(f'self.raw_samples.shape = {self.raw_samples.shape}')

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        target = self.targets[idx]
        raw_sample = self.raw_samples[idx:idx + 1]

        # raw_mean = np.mean(raw_sample, axis=(0, 2), keepdims=True)
        # raw_std = np.std(raw_sample, axis=(0, 2), keepdims=True)
        # raw_sample = (raw_sample - raw_mean) / raw_std
        # raw_mean = np.mean(raw_sample, axis=(0, 2), keepdims=True)
        # raw_std = np.std(raw_sample, axis=(0, 2), keepdims=True)

        if self.data_type == 'raw':
            sample_data = np.expand_dims(raw_sample, axis=1)
        elif self.data_type == 'power_spectrum':
            # wavelet (morlet) transform
            power_spectrum = mne.time_frequency.tfr_array_morlet(
                raw_sample,
                sfreq=self.raw.info['sfreq'],
                freqs=self.freqs,
                n_cycles=self.freqs,
                output='power',
                n_jobs=1
            )

            if self.baseline_correction:
                power_spectrum = (power_spectrum - self.baseline_mean) / self.baseline_mean
            else:
                power_spectrum = np.log(power_spectrum)


            sample_data = power_spectrum
        else:
            raise NotImplementedError

        if self.normalization == 'minmax':
            sample_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
        elif self.normalization == 'meanstd':
            sample_data = (sample_data - sample_data.mean()) / sample_data.std()

        sample = {
            'data': torch.from_numpy(sample_data[0]).float(),
            'raw': torch.from_numpy(raw_sample[0]).float(),
            'target': target,
            'start_time': self.sample_start_times[idx],
            'eeg_file_path': self.eeg_file_path,
            'channel_name_to_idx': self.channel_name_to_idx,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'time_to_closest_seizure': self.times_to_closest_seizure[idx],
        }

        if self.stats_data is not None:
            sample['cwt_mean'] = np.expand_dims(np.expand_dims(self.stats_data['mean'], axis=0), axis=3)
            sample['cwt_std'] = np.expand_dims(np.expand_dims(self.stats_data['std'], axis=0), axis=3)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _generate_data(self):
        # get start time for samples
        seizure_times = custom_distribution.unifrom_segments_sample(size=self.samples_num, segments=self.seizures)
        assert all(
            [
                any([seizure['start'] < seizure_time < seizure['end'] for seizure in self.seizures])
                for seizure_time in seizure_times
            ]
        )

        normal_times = custom_distribution.unifrom_segments_sample(size=self.samples_num, segments=self.normal_segments)
        assert all(
            [
                any(
                    [normal_segment['start'] < normal_time < normal_segment['end'] for normal_segment in
                     self.normal_segments]
                )
                for normal_time in normal_times
            ]
        )

        # generate samples
        mask = np.random.uniform(size=self.samples_num) > self.normal_samples_fraction
        targets = mask.astype(int)
        sample_start_times = mask * seizure_times + (1 - mask) * normal_times

        if self.prediction_data is not None:
            fp_start_times = extract_fp_times(self.prediction_data, threshold=0.70, sfreq=self.raw.info['sfreq'])
            fn_start_times = extract_fn_times(self.prediction_data, threshold=0.99, sfreq=self.raw.info['sfreq'])

            fp_start_times = filter_fp_times(fp_start_times, self.seizures, min_deviation=300, min_start_time=900)
            fn_start_times = filter_fn_times(fn_start_times, self.seizures, min_deviation=-1, min_start_time=900)

            fn_num_to_pick = min(len(fn_start_times), self.samples_num // 2)
            fp_num_to_pick = min(len(fp_start_times), self.samples_num // 2, fn_num_to_pick)

            fp_picked_start_times = np.random.choice(fp_start_times, size=fp_num_to_pick, replace=False)
            fn_picked_start_times = np.random.choice(fn_start_times, size=fn_num_to_pick, replace=False)

            false_preds_start_times = np.append(fp_picked_start_times, fn_picked_start_times)
            false_preds_labels = np.append(np.zeros_like(fp_picked_start_times), np.ones_like(fn_picked_start_times))

            random_permutation = np.random.permutation(len(false_preds_labels))
            false_preds_start_times = false_preds_start_times[random_permutation]
            false_preds_labels = false_preds_labels[random_permutation]

            sample_start_times = np.append(sample_start_times, false_preds_start_times)
            targets = np.append(targets, false_preds_labels)

            sample_start_times = sample_start_times[-self.samples_num:]
            targets = targets[-self.samples_num:]

            random_permutation = np.random.permutation(len(sample_start_times))
            sample_start_times = sample_start_times[random_permutation]
            targets = targets[random_permutation]

        # get time to closest seizure
        times_to_closest_seizure = [
            calc_signed_time_to_closest_seizure(sample_start_time, self.seizures)
            for sample_start_time in sample_start_times
        ]

        raw_samples, time_idxs_start, time_idxs_end = generate_raw_samples(self.raw, sample_start_times, self.sample_duration)

        return mask, sample_start_times, targets, raw_samples, time_idxs_start, time_idxs_end, times_to_closest_seizure

    def renew_data(self):
        (
            self.mask,
            self.sample_start_times,
            self.targets,
            self.raw_samples,
            self.time_idxs_start,
            self.time_idxs_end,
            self.times_to_closest_seizure,
        ) = self._generate_data()


class SubjectSequentialDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            eeg_file_path,
            seizures,
            stats_path=None,
            sample_duration=60,
            shift=None,
            normalization=None,
            data_type='power_spectrum',
            baseline_correction=False,
            transform=None,
    ):
        self.eeg_file_path = eeg_file_path
        self.raw = eeg_reader.EEGReader.read_eeg(self.eeg_file_path)
        self.seizures = seizures
        self.stats_data = None if stats_path is None else pickle.load(open(stats_path, 'rb'))
        self.sample_duration = sample_duration
        self.shift = shift if shift is not None else self.sample_duration
        self.data_type = data_type
        self.normalization = normalization
        self.baseline_correction = baseline_correction
        self.transform = transform

        # trim last seizure
        if self.seizures[-1]['end'] + self.sample_duration >= self.raw.times.max():
            print('Trimming last seizure')
            self.seizures[-1]['end'] = self.raw.times.max() - self.sample_duration - 1e-3

        # drop unnecessary channels
        if 'data1' in self.eeg_file_path:
            channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
        elif 'data2' in self.eeg_file_path:
            channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
        else:
            raise NotImplementedError

        channels_num = len(self.raw.info['ch_names'])
        channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
        self.raw.drop_channels(channels_to_drop)

        self.channel_name_to_idx = {
            channel_name.replace('EEG ', ''): channel_idx for channel_idx, channel_name in
            enumerate(self.raw.info['ch_names'])
        }

        # calc stuff for baseline correction
        self.freqs = np.arange(1, 40.01, 0.1)
        self.baseline_mean, self.baseline_std = get_baseline_stats(
            self.raw,
            baseline_length_in_seconds=500,
            sfreq=self.raw.info['sfreq'],
            freqs=self.freqs,
        )
        self.baseline_mean = self.baseline_mean[np.newaxis]
        self.baseline_std = self.baseline_std[np.newaxis]

        # get time limits of eeg file in seconds
        time_start, time_end = self.raw.times.min(), self.raw.times.max()

        # get start time for samples
        self.sample_start_times = [start_time for start_time in np.arange(time_start, time_end - self.sample_duration, self.shift)]

        # generate targets
        self.targets = np.array([
            1 if check_if_in_segments(start_time, self.seizures) or check_if_in_segments(start_time + self.sample_duration, self.seizures) else 0
            for start_time in self.sample_start_times
        ])

        # generate raw samples
        self.raw_samples, self.time_idxs_start, self.time_idxs_end = generate_raw_samples(self.raw, self.sample_start_times, self.sample_duration)

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        target = self.targets[idx]
        raw_sample = self.raw_samples[idx:idx + 1]
        if self.data_type == 'raw':
            sample_data = np.expand_dims(raw_sample, axis=1)
        elif self.data_type == 'power_spectrum':
            # wavelet (morlet) transform
            power_spectrum = mne.time_frequency.tfr_array_morlet(
                raw_sample,
                sfreq=self.raw.info['sfreq'],
                freqs=self.freqs,
                n_cycles=self.freqs,
                output='power',
                n_jobs=1
            )

            if self.baseline_correction:
                power_spectrum = (power_spectrum - self.baseline_mean) / self.baseline_mean
            else:
                power_spectrum = np.log(power_spectrum)

            sample_data = power_spectrum
        else:
            raise NotImplementedError

        if self.normalization == 'minmax':
            sample_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
        elif self.normalization == 'meanstd':
            sample_data = (sample_data - sample_data.mean()) / sample_data.std()

        sample = {
            'data': torch.from_numpy(sample_data[0]).float(),
            'raw': torch.from_numpy(raw_sample[0]).float(),
            'target': target,
            'start_time': self.sample_start_times[idx],
            'eeg_file_path': self.eeg_file_path,
            'channel_name_to_idx': self.channel_name_to_idx,
            'time_idx_start': self.time_idxs_start[idx],
            'time_idx_end': self.time_idxs_end[idx],
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
        }

        if self.stats_data is not None:
            sample['cwt_mean'] = np.expand_dims(np.expand_dims(self.stats_data['mean'], axis=0), axis=3)
            sample['cwt_std'] = np.expand_dims(np.expand_dims(self.stats_data['std'], axis=0), axis=3)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class SubjectInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file_path, sample_start_times, sample_duration=60, normalization=None, data_type='power_spectrum', transform=None):
        self.eeg_file_path = eeg_file_path
        self.raw = eeg_reader.EEGReader.read_eeg(self.eeg_file_path)
        self.sample_duration = sample_duration
        self.data_type = data_type
        self.normalization = normalization
        self.transform = transform
        self.sample_start_times = sample_start_times

        # drop unnecessary channels
        if 'data1' in self.eeg_file_path:
            channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
        elif 'data2' in self.eeg_file_path:
            channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
        else:
            raise NotImplementedError

        channels_num = len(self.raw.info['ch_names'])
        channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
        self.raw.drop_channels(channels_to_drop)

        self.channel_name_to_idx = {
            channel_name.replace('EEG ', ''): channel_idx for channel_idx, channel_name in enumerate(self.raw.info['ch_names'])
        }

        self.raw_samples, self.time_idxs_start, self.time_idxs_end = generate_raw_samples(
            self.raw,
            self.sample_start_times,
            self.sample_duration,
        )
        self.freqs = np.arange(1, 40.01, 0.1)
        # print(f'self.raw_samples.shape = {self.raw_samples.shape}')

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        raw_sample = self.raw_samples[idx:idx + 1]

        if self.data_type == 'raw':
            sample_data = np.expand_dims(raw_sample, axis=1)
        elif self.data_type == 'power_spectrum':
            # wavelet (morlet) transform
            power_spectrum = mne.time_frequency.tfr_array_morlet(
                raw_sample,
                sfreq=self.raw.info['sfreq'],
                freqs=self.freqs,
                n_cycles=self.freqs,
                output='power',
                n_jobs=1
            )
            power_spectrum = np.log(power_spectrum)

            sample_data = power_spectrum
        else:
            raise NotImplementedError

        if self.normalization == 'minmax':
            sample_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
        elif self.normalization == 'meanstd':
            sample_data = (sample_data - sample_data.mean()) / sample_data.std()

        sample = {
            'data': torch.from_numpy(sample_data[0]).float(),
            'raw': torch.from_numpy(raw_sample[0]).float(),
            'start_time': self.sample_start_times[idx],
            'eeg_file_path': self.eeg_file_path,
            'channel_name_to_idx': self.channel_name_to_idx,
            'time_idx_start': self.time_idxs_start[idx],
            'time_idx_end': self.time_idxs_end[idx],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class SubjectPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_dir, seizures=None, sfreq=128, sample_duration=10, log=False, normalization=None, transform=None):
        self.sample_duration = sample_duration
        self.sfreq = sfreq
        self.preprocessed_dir = preprocessed_dir

        self.filenames = [
            filename
            for filename in sorted(os.listdir(self.preprocessed_dir))
            if filename.endswith('.npy')
        ]
        self.file_paths = [os.path.join(self.preprocessed_dir, filename) for filename in self.filenames]

        self.seizures = seizures

        self.log = log
        self.normalization = normalization
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sample_data = np.load(self.file_paths[idx])

        filename = self.filenames[idx]
        filename = os.path.splitext(filename)[0]
        start_time = int(filename.split('start_time=')[1])
        time_idx_start = int((filename.split('time_idx_start=')[1]).split('_')[0])

        target = 1 if check_if_in_segments(start_time, self.seizures) or check_if_in_segments(start_time + self.sample_duration, self.seizures) else 0

        if self.log:
            sample_data = np.log(sample_data)

        if self.normalization == 'minmax':
            sample_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
        elif self.normalization == 'meanstd':
            sample_data = (sample_data - sample_data.mean()) / sample_data.std()

        sample = {
            'data': torch.from_numpy(sample_data).float(),
            'target': target,
            'start_time': start_time,
            'time_idx_start': time_idx_start,
            'time_idx_end': time_idx_start + self.sfreq * self.sample_duration,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def custom_collate_function(batch, data_type='power_spectrum', normalization=None, freqs=np.arange(1, 40.01, 0.1), sfreq=128, baseline_correction=False, log=True, transform=None):
    if data_type == 'power_spectrum':
        # wavelet (morlet) transform
        raw_data = torch.cat([sample_data['data'] for sample_data in batch], dim=0)
        power_spectrum = mne.time_frequency.tfr_array_morlet(
            raw_data.numpy(),
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=freqs,
            output='power',
            n_jobs=-1
        )

        if baseline_correction:
            baseline_mean = np.concatenate([sample_data['baseline_mean'] for sample_data in batch], axis=0)
            power_spectrum = (power_spectrum - baseline_mean) / baseline_mean
        elif log:
            power_spectrum = np.log(power_spectrum)

        if normalization == 'minmax':
            power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())
        elif normalization == 'meanstd':
            power_spectrum = (power_spectrum - power_spectrum.mean()) / power_spectrum.std()
        elif normalization == 'cwt_meanstd':
            cwt_mean = np.concatenate([sample_data['cwt_mean'] for sample_data in batch], axis=0)
            cwt_std = np.concatenate([sample_data['cwt_std'] for sample_data in batch], axis=0)
            power_spectrum = (power_spectrum - cwt_mean) / cwt_std

        for sample_idx in range(power_spectrum.shape[0]):
            batch[sample_idx]['data'] = torch.from_numpy(power_spectrum[sample_idx]).float()
            if transform is not None:
                batch[sample_idx] = transform(batch[sample_idx])
            # batch[sample_idx]['data'] = torch.from_numpy(power_spectrum[sample_idx])
        # batch['data'] = torch.from_numpy(sample_data).float()
    elif data_type == 'raw':
        raw_data = torch.cat([sample_data['data'] for sample_data in batch], dim=0)
        if transform is not None:
            raw_data = transform({'data': raw_data})['data']

        if normalization == 'minmax':
            raw_data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())
        elif normalization == 'meanstd':
            raw_data = (raw_data - raw_data.mean()) / raw_data.std()

        for sample_idx in range(raw_data.shape[0]):
            batch[sample_idx]['data'] = torch.unsqueeze(raw_data[sample_idx], dim=0).float()

    batch = torch.utils.data.dataloader.default_collate(batch)

    return batch


def tta_collate_function(batch, tta_augs=tuple(), data_type='power_spectrum', normalization=None, freqs=np.arange(1, 40.01, 0.1), sfreq=128, baseline_correction=False, log=True):
    if data_type == 'power_spectrum':
        # wavelet (morlet) transform
        raw_data = torch.cat([sample_data['data'] for sample_data in batch], dim=0)
        power_spectrum = mne.time_frequency.tfr_array_morlet(
            raw_data.numpy(),
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=freqs,
            output='power',
            n_jobs=-1
        )

        if baseline_correction:
            baseline_mean = np.concatenate([sample_data['baseline_mean'] for sample_data in batch], axis=0)
            power_spectrum = (power_spectrum - baseline_mean) / baseline_mean
        elif log:
            power_spectrum = np.log(power_spectrum)

        if normalization == 'minmax':
            power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())
        elif normalization == 'meanstd':
            power_spectrum = (power_spectrum - power_spectrum.mean()) / power_spectrum.std()

        for sample_idx in range(power_spectrum.shape[0]):
            batch[sample_idx]['data'] = torch.from_numpy(power_spectrum[sample_idx]).float()
    elif data_type == 'raw':
        pass  # TODO: Add TTA for raw data

    for sample_idx in range(len(batch)):
        original_data = batch[sample_idx]['data'].clone()
        for aug_idx, aug_func in enumerate(tta_augs):
            transformed_sample = aug_func(batch[sample_idx])
            batch[sample_idx][f'data_aug{aug_idx:03}'] = transformed_sample['data']
            batch[sample_idx]['data'] = original_data.clone()
    batch = torch.utils.data.dataloader.default_collate(batch)

    return batch


if __name__ == '__main__':
    import json

    dataset_info_path = '../data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = '../data'
    # subject_key = 'data1/dataset28'
    # subject_key = 'data1/dataset2'
    subject_key = 'data2/038tl Anonim-20190821_113559-20211123_004935'
    # subject_key = 'data2/003tl Anonim-20200831_040629-20211122_135924'
    subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
    subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))
    stats_path = os.path.join(data_dir, 'cwt_log_stats_v2', subject_key + '.npy')

    # import augmentations.flip
    # import augmentations.spec_augment
    # import torchvision.transforms
    # subject_dataset = SubjectPreprocessedDataset(
    #     # preprocessed_dir=f'D:\\Study\\asp\\thesis\\implementation\\data_ocsvm_cwt\\{subject_key.split("/")[1]}',
    #     preprocessed_dir=f'C:\\data\\{subject_key.split("/")[1]}',
    #     seizures=subject_seizures,
    #     normalization='meanstd',
    #     transform=torchvision.transforms.Compose([
    #         augmentations.flip.TimeFlip(p=0.5),
    #         augmentations.spec_augment.SpecAugment(
    #             p_aug=0.5,
    #             p_mask=0.5,
    #             max_freq_mask_width=40,
    #             max_time_mask_width=128,
    #             num_masks_per_channel=1,
    #             replace_with_zero=False,
    #         ),
    #     ]),
    # )
    # sample = subject_dataset[0]
    # # exit()

    import time
    import utils.neural.training
    device = torch.device('cuda:0')
    # model = utils.neural.training.get_model(model_name='resnet18_1channel', model_kwargs={'pretrained': True}).to(device)
    model = utils.neural.training.get_model(model_name='resnet18', model_kwargs={'pretrained': True}).to(device)
    model.eval()

    # loader = torch.utils.data.DataLoader(subject_dataset, batch_size=16, collate_fn=torch.utils.data.dataloader.default_collate)
    # torch.cuda.synchronize()
    # time_start = time.time()
    # for batch_idx, batch in enumerate(loader):
    #     with torch.no_grad():
    #         output = model(batch['data'].to(device))
    #     print(f'\rProgress {batch_idx + 1}/{len(loader)}', end='')
    #     if batch_idx > 20:
    #         break
    # torch.cuda.synchronize()
    # time_elapsed = time.time() - time_start
    # print(f'\ntime_elapsed = {time_elapsed}')
    # exit()

    # subject_dataset = SubjectRandomDataset(
    #     subject_eeg_path,
    #     subject_seizures,
    #     samples_num=100,
    #     # prediction_data_path=r'D:\Study\asp\thesis\implementation\experiments\renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug\predictions\data2\003tl Anonim-20200831_040629-20211122_135924.pickle',
    #     sample_duration=10,
    #     # data_type='raw',
    #     data_type='power_spectrum',
    #     baseline_correction=True
    # )
    # print(subject_dataset[0]['data'].shape)
    # exit()
    # # subject_dataset = SubjectSequentialDataset(subject_eeg_path, subject_seizures, sample_duration=10)
    # loader = torch.utils.data.DataLoader(subject_dataset, batch_size=16)
    # torch.cuda.synchronize()
    # time_start = time.time()
    # for batch_idx, batch in enumerate(loader):
    #     with torch.no_grad():
    #         output = model(batch['data'].to(device))
    #     print(f'\rProgress {batch_idx + 1}/{len(loader)}', end='')
    #     if batch_idx > 20:
    #         break
    # torch.cuda.synchronize()
    # time_elapsed = time.time() - time_start
    # print(f'\ntime_elapsed = {time_elapsed}')

    subject_dataset = SubjectRandomDataset(
        subject_eeg_path,
        subject_seizures,
        stats_path=stats_path,
        samples_num=100,
        sample_duration=10,
        data_type='raw',
        baseline_correction=True,
    )
    # subject_dataset = SubjectSequentialDataset(subject_eeg_path, subject_seizures, sample_duration=10, data_type='raw')
    from functools import partial
    # collate_fn = partial(custom_collate_function, data_type='power_spectrum', normalization=None, baseline_correction=True)
    import torchvision
    import augmentations.ts_augments
    collate_fn = partial(
        custom_collate_function,
        data_type='power_spectrum',
        normalization='cwt_meanstd',
        baseline_correction=False,
        transform=torchvision.transforms.Compose([
            # augmentations.ts_augments.AddNoise(p=0.5),
            # augmentations.ts_augments.TimeWarp(p=0.5),
            # augmentations.ts_augments.Drift(p=0.5),
            # augmentations.ts_augments.Crop(p=0.5),
        ])
    )
    loader = torch.utils.data.DataLoader(subject_dataset, batch_size=16, collate_fn=collate_fn)
    torch.cuda.synchronize()
    time_start = time.time()
    for batch_idx, batch in enumerate(loader):
        with torch.no_grad():
            output = model(batch['data'].to(device))
        print(f'\rProgress {batch_idx + 1}/{len(loader)}', end='')
        if batch_idx > 20:
            break
    torch.cuda.synchronize()
    time_elapsed = time.time() - time_start
    print(f'\ntime_elapsed = {time_elapsed}')
    exit()

    subject_dataset = SubjectSequentialDataset(subject_eeg_path, subject_seizures, sample_duration=10)
    for idx in range(1):
        dataset_sample = subject_dataset[idx]
        import visualization

        print(dataset_sample['target'])
        visualization.plot_spectrum_averaged(np.exp(dataset_sample['data'].cpu().numpy()), subject_dataset.freqs)
        visualization.plot_spectrum_channels(dataset_sample['data'].cpu().numpy(), time_idx_from=0, time_idx_to=128 * 9)
        for key in dataset_sample.keys():
            print_value = dataset_sample[key].shape if hasattr(dataset_sample[key], "shape") and len(dataset_sample[key].shape) > 1 else dataset_sample[key]
            print(f'dataset_sample[{key}] {print_value}')
        print()
