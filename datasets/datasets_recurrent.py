import mne
import numpy as np
import torch
from torch.utils.data._utils import collate

import custom_distribution
from datasets.datasets_static import get_baseline_stats_raw, check_if_in_segments
import eeg_reader

mne.set_log_level('error')


def generate_raw_samples(raw_eeg, sample_sequence_start_times, sample_duration, sample_sequence_length):
    # sample_sequence_start_times.shape = (samples_num, sample_sequence_length)

    raw_data = raw_eeg.get_data()

    samples_num = len(sample_sequence_start_times)
    frequency = raw_eeg.info['sfreq']
    sample_len_in_idxs = int(sample_duration * frequency)
    channels_num = len(raw_eeg.info['ch_names'])

    sample_sequences = np.zeros((samples_num, sample_sequence_length, channels_num, sample_len_in_idxs), dtype=np.float32)
    sequence_time_idxs_start = np.zeros_like(sample_sequence_start_times, dtype=np.float32)
    sequence_time_idxs_end = np.zeros_like(sample_sequence_start_times, dtype=np.float32)
    for sample_idx in range(samples_num):
        sample_start_time = sample_sequence_start_times[sample_idx][0]

        start_idx = int(frequency * sample_start_time)
        end_idx = start_idx + sample_len_in_idxs
        for element_idx in range(sample_sequence_length):
            sample_sequences[sample_idx, element_idx] = raw_data[:, start_idx:end_idx]
            sequence_time_idxs_start[sample_idx] = start_idx
            sequence_time_idxs_end[sample_idx] = end_idx

            start_idx += sample_len_in_idxs
            end_idx += sample_len_in_idxs

    return sample_sequences, sequence_time_idxs_start, sequence_time_idxs_end


def read_raw(eeg_file_path):
    raw = eeg_reader.EEGReader.read_eeg(eeg_file_path)

    # drop unnecessary channels
    if 'data1' in eeg_file_path:
        channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
    elif 'data2' in eeg_file_path:
        channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
    else:
        raise NotImplementedError

    channels_num = len(raw.info['ch_names'])
    channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
    raw.drop_channels(channels_to_drop)

    return raw


class SubjectRandomRecurrentDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            eeg_file_path,
            seizures,
            samples_num,
            sample_duration=60,
            sample_sequence_length=6,
            normal_samples_fraction=0.5,
            normalization=None,
            data_type='raw',
            baseline_correction=False,
            transform=None,
    ):
        self.eeg_file_path = eeg_file_path
        self.seizures = seizures
        self.samples_num = samples_num
        self.sample_duration = sample_duration
        self.sample_sequence_length = sample_sequence_length
        self.data_type = data_type
        self.normal_samples_fraction = normal_samples_fraction
        self.normalization = normalization
        self.baseline_correction = baseline_correction
        self.transform = transform

        self.raw = read_raw(self.eeg_file_path)

        # trim last seizure
        if self.seizures[-1]['end'] + self.sample_duration >= self.raw.times.max():
            print('Trimming last seizure')
            self.seizures[-1]['end'] = self.raw.times.max() - self.sample_duration - 1e-3

        self.channel_name_to_idx = {
            channel_name.replace('EEG ', ''): channel_idx for channel_idx, channel_name in enumerate(self.raw.info['ch_names'])
        }

        # calc stuff for baseline correction
        self.freqs = np.arange(1, 40.01, 0.1)
        if self.baseline_correction:
            self.baseline_mean, self.baseline_std = get_baseline_stats_raw(
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

        # generate normal_segments using seizures and time limits of eeg file
        self.time_start, self.time_end = self.raw.times.min(), self.raw.times.max()
        time_points = [self.time_start]
        for seizure in self.seizures:
            assert seizure['start'] > time_points[-1]
            time_points.append(seizure['start'] - self.sample_duration * self.sample_sequence_length)
            time_points.append(seizure['end'])
        time_points.append(self.time_end - self.sample_duration * self.sample_sequence_length)
        assert len(time_points) % 2 == 0

        self.normal_segments = list()
        for normal_segment_idx in range(len(time_points) // 2):
            segment_duration = time_points[normal_segment_idx * 2 + 1] - time_points[normal_segment_idx * 2]
            if segment_duration <= 0:
                continue

            self.normal_segments.append({
                'start': time_points[normal_segment_idx * 2],
                'end': time_points[normal_segment_idx * 2 + 1],
            })

        (
            self.mask,
            self.sample_start_times,
            self.targets,
            self.raw_samples,
            self.time_idxs_start,
            self.time_idxs_end,
        ) = self._generate_data()

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        target_sequence = self.targets[idx]
        raw_sample = self.raw_samples[idx]
        target = int(target_sequence.sum() > 0)

        if self.data_type == 'raw':
            sample_data = np.expand_dims(raw_sample, axis=1)
            if self.baseline_correction:
                sample_data = (sample_data - self.baseline_mean) / self.baseline_mean
        else:
            raise NotImplementedError

        if self.normalization == 'minmax':
            sample_min = np.min(sample_data, axis=(0, 3), keepdims=True)
            sample_max = np.max(sample_data, axis=(0, 3), keepdims=True)
            sample_data = (sample_data - sample_min) / (sample_max - sample_min)
        elif self.normalization == 'meanstd':
            sample_mean = np.mean(sample_data, axis=(0, 3), keepdims=True)
            sample_std = np.std(sample_data, axis=(0, 3), keepdims=True)
            sample_data = (sample_data - sample_mean) / sample_std

        sample = {
            'data': torch.from_numpy(sample_data).float(),
            'raw': torch.from_numpy(raw_sample).float(),
            'target': target,
            'target_sequence': target_sequence,
            'start_time': self.sample_start_times[idx],
            'eeg_file_path': self.eeg_file_path,
            'channel_name_to_idx': self.channel_name_to_idx,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _generate_data(self):
        # get times of seizure points
        seizure_times = custom_distribution.unifrom_segments_sample(size=self.samples_num, segments=self.seizures)  # (samples_num, )
        assert all(
            [
                any([seizure['start'] < seizure_time < seizure['end'] for seizure in self.seizures])
                for seizure_time in seizure_times
            ]
        )

        seizure_positions = np.random.randint(0, self.sample_sequence_length, size=self.samples_num)  # (samples_num, )
        seizure_sequence_times = [
            [
                seizure_times[seizure_sample_idx] + (idx - seizure_positions[seizure_sample_idx]) * self.sample_duration
                for idx in range(0, self.sample_sequence_length)
            ]
            for seizure_sample_idx in range(self.samples_num)
        ]
        seizure_sequence_times = np.array(seizure_sequence_times)  # (samples_num, sample_sequence_length)

        # TODO: filter out
        for sample_idx in range(self.samples_num):
            if seizure_sequence_times[sample_idx][0] < 0:
                seizure_sequence_times[sample_idx] += abs(seizure_sequence_times[sample_idx][0])

            if seizure_sequence_times[sample_idx][self.sample_sequence_length - 1] > (self.time_end - self.sample_duration):
                seizure_sequence_times[sample_idx] -= abs(seizure_sequence_times[sample_idx][self.sample_sequence_length - 1] + self.sample_duration - self.time_end)

        seizure_sequence_targets = [
            [
                1 if check_if_in_segments(seizure_sequence_times[seizure_sample_idx][idx], self.seizures) else 0
                for idx in range(0, self.sample_sequence_length)
            ]
            for seizure_sample_idx in range(self.samples_num)
        ]
        seizure_sequence_targets = np.array(seizure_sequence_targets)  # (samples_num, sample_sequence_length)
        assert all([np.any(seizure_sequence_target == 1) for seizure_sequence_target in seizure_sequence_targets])

        # get times of normal points
        normal_times = custom_distribution.unifrom_segments_sample(size=self.samples_num, segments=self.normal_segments)  # (samples_num, )
        assert all(
            [
                any([normal_segment['start'] < normal_time < normal_segment['end'] for normal_segment in self.normal_segments])
                for normal_time in normal_times
            ]
        )

        normal_sequence_times = [
            [
                normal_times[normal_sample_idx] + idx * self.sample_duration
                for idx in range(0, self.sample_sequence_length)
            ]
            for normal_sample_idx in range(self.samples_num)
        ]
        normal_sequence_times = np.array(normal_sequence_times)  # (samples_num, sample_sequence_length)
        normal_sequence_times[0] = 0  # TODO: REMOVE!!!

        # TODO: filter out
        for sample_idx in range(self.samples_num):
            offset_to_left = 0
            for element_idx in range(self.sample_sequence_length):
                normal_time = normal_sequence_times[sample_idx][element_idx]
                if check_if_in_segments(normal_time, self.seizures):
                    offset_to_left = (self.sample_sequence_length - element_idx) * self.sample_duration
                    break

            normal_sequence_times[sample_idx] -= offset_to_left

        normal_sequence_targets = [
            [
                1 if check_if_in_segments(normal_sequence_times[normal_sample_idx][idx], self.seizures) else 0
                for idx in range(0, self.sample_sequence_length)
            ]
            for normal_sample_idx in range(self.samples_num)
        ]
        normal_sequence_targets = np.array(normal_sequence_targets)  # (samples_num, sample_sequence_length)
        assert np.all(normal_sequence_targets == 0)

        # generate samples
        mask = np.random.uniform(size=self.samples_num) > self.normal_samples_fraction
        mask = mask[..., np.newaxis]  # (samples_num, 1)
        sample_sequence_times = mask * seizure_sequence_times + (1 - mask) * normal_sequence_times  # (samples_num, sample_sequence_length)
        sample_sequence_targets = mask * seizure_sequence_targets + (1 - mask) * normal_sequence_targets  # (samples_num, sample_sequence_length)

        (
            sample_sequence_raws,
            sample_sequence_time_idxs_start,
            sample_sequence_time_idxs_end
        ) = generate_raw_samples(self.raw, sample_sequence_times, self.sample_duration, self.sample_sequence_length)

        del self.raw

        return (
            mask,
            sample_sequence_times,
            sample_sequence_targets,
            sample_sequence_raws,
            sample_sequence_time_idxs_start,
            sample_sequence_time_idxs_end,
        )

    def renew_data(self):
        if not hasattr(self, 'raw'):
            self.raw = read_raw(self.eeg_file_path)

        (
            self.mask,
            self.sample_start_times,
            self.targets,
            self.raw_samples,
            self.time_idxs_start,
            self.time_idxs_end,
        ) = self._generate_data()


if __name__ == '__main__':
    import os
    import json
    from utils.neural.training import set_seed

    set_seed(8, deterministic=True)

    dataset_info_path = '../data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = '../data'
    # subject_key = 'data1/dataset28'
    # subject_key = 'data1/dataset2'
    subject_key = 'data2/038tl Anonim-20190821_113559-20211123_004935'
    # subject_key = 'data2/037tl Anonim-20191020_110036-20211122_223805'
    # subject_key = 'data2/003tl Anonim-20200831_040629-20211122_135924'
    subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
    subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))
    stats_path = os.path.join(data_dir, 'cwt_log_stats_v2', subject_key + '.npy')

    subject_dataset = SubjectRandomRecurrentDataset(
        subject_eeg_path,
        subject_seizures,
        samples_num=100,
        sample_duration=10,
        sample_sequence_length=6,
        normalization=None,
        data_type='raw',
        baseline_correction=False,
    )
    sample = subject_dataset[0]

    # inference debug
    import utils.neural.training
    loader = utils.neural.training.get_loader(
        subject_dataset,
        loader_kwargs={'batch_size': 16, 'shuffle': False},
    )



