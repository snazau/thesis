import mne
import numpy as np
import torch

import custom_distribution
import eeg_reader

# TODO: move code duplication into separate functions


def check_if_in_segments(time_point, segments):
    return any([segment['start'] <= time_point <= segment['end'] for segment in segments])


def generate_raw_samples(raw_eeg, sample_start_times, sample_duration):
    # my_annot = mne.Annotations(
    #     onset=[sample_start_time for sample_start_time in sample_start_times],
    #     duration=[self.sample_duration for _ in sample_start_times],
    #     description=[f'seizure' if is_seizure_sample else f'normal' for idx, is_seizure_sample in enumerate(self.mask)]
    # )
    # self.raw.set_annotations(my_annot)
    # self.raw.plot()
    # import matplotlib.pyplot as plt
    # plt.show()

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
    for sample_idx, sample_start_time in enumerate(sample_start_times):
        start_idx = int(frequency * sample_start_time)
        end_idx = start_idx + sample_len_in_idxs
        samples[sample_idx] = raw_data[:, start_idx:end_idx]

    return samples


class SubjectRandomDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file_path, seizures, samples_num, sample_duration=60, normalization=None, data_type='power_spectrum', transform=None):
        self.eeg_file_path = eeg_file_path
        self.raw = eeg_reader.EEGReader.read_eeg(self.eeg_file_path)
        self.seizures = seizures
        self.samples_num = samples_num
        self.sample_duration = sample_duration
        self.data_type = data_type
        self.normalization = normalization
        self.transform = transform

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

        # set montage
        # print(self.raw.get_data().min(), self.raw.get_data().mean(), self.raw.get_data().max())
        # montage = mne.channels.make_standard_montage('standard_1020')
        # self.raw.set_montage(montage, on_missing='ignore', match_alias={ch: ch.replace('EEG ', '').strip() for ch in self.raw.info['ch_names']})
        # self.raw.plot_sensors(kind='topomap');

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

        # get start time for samples
        seizure_times = custom_distribution.unifrom_segments_sample(size=self.samples_num, segments=self.seizures)
        assert all([
            any([seizure['start'] < seizure_time < seizure['end'] for seizure in self.seizures])
            for seizure_time in seizure_times
        ])

        normal_times = custom_distribution.unifrom_segments_sample(size=self.samples_num, segments=self.normal_segments)
        assert all([
            any([normal_segment['start'] < normal_time < normal_segment['end'] for normal_segment in self.normal_segments])
            for normal_time in normal_times
        ])

        # generate samples
        self.mask = np.random.uniform(size=self.samples_num) > 0.5
        self.sample_start_times = self.mask * seizure_times + (1 - self.mask) * normal_times
        # print(f'len(sample_start_times) = {len(sample_start_times)}')
        # sample_start_times[0] = 23860

        self.targets = self.mask.astype(np.int)
        self.raw_samples = generate_raw_samples(self.raw, self.sample_start_times, self.sample_duration)
        self.freqs = np.arange(1, 40.01, 0.1)
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
            power_spectrum = np.log(power_spectrum)

            if self.normalization == 'minmax':
                power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())
            elif self.normalization == 'meanstd':
                power_spectrum = (power_spectrum - power_spectrum.mean()) / power_spectrum.std()

            sample_data = power_spectrum
        else:
            raise NotImplementedError

        sample = {
            'data': torch.from_numpy(sample_data).float(),
            'raw': torch.from_numpy(raw_sample).float(),
            'target': target,
            'start_time': self.sample_start_times[idx],
            'eeg_file_path': self.eeg_file_path,
        }
        return sample


class SubjectSequentialDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file_path, seizures, sample_duration=60, normalization=None, data_type='power_spectrum', transform=None):
        self.eeg_file_path = eeg_file_path
        self.raw = eeg_reader.EEGReader.read_eeg(self.eeg_file_path)
        self.seizures = seizures
        self.sample_duration = sample_duration
        self.data_type = data_type
        self.normalization = normalization
        self.transform = transform

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

        # get time limits of eeg file in seconds
        time_start, time_end = self.raw.times.min(), self.raw.times.max()

        # get start time for samples
        self.sample_start_times = [start_time for start_time in np.arange(time_start, time_end - self.sample_duration, self.sample_duration)]

        # generate targets
        self.targets = np.array([
            1 if check_if_in_segments(start_time, self.seizures) or check_if_in_segments(start_time + self.sample_duration, self.seizures) else 0
            for start_time in self.sample_start_times
        ])

        # generate raw samples
        self.raw_samples = generate_raw_samples(self.raw, self.sample_start_times, self.sample_duration)
        self.freqs = np.arange(1, 40.01, 0.1)

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
            power_spectrum = np.log(power_spectrum)

            if self.normalization == 'minmax':
                power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())
            elif self.normalization == 'meanstd':
                power_spectrum = (power_spectrum - power_spectrum.mean()) / power_spectrum.std()

            sample_data = power_spectrum
        else:
            raise NotImplementedError

        sample = {
            'data': torch.from_numpy(sample_data).float(),
            'raw': torch.from_numpy(raw_sample).float(),
            'target': target,
            'start_time': self.sample_start_times[idx],
            'eeg_file_path': self.eeg_file_path,
        }
        return sample


if __name__ == '__main__':
    import json
    import os

    dataset_info_path = './data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = './data'
    # subject_key = 'data1/dataset28'
    # subject_key = 'data1/dataset2'
    subject_key = 'data2/038tl Anonim-20190821_113559-20211123_004935'
    subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
    subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

    # subject_dataset = SubjectRandomDataset(subject_eeg_path, subject_seizures, samples_num=100, sample_duration=10)
    subject_dataset = SubjectSequentialDataset(subject_eeg_path, subject_seizures, sample_duration=10)
    for idx in range(1):
        dataset_sample = subject_dataset[idx]
        import visualization

        print(dataset_sample['target'])
        visualization.plot_spectrum_averaged(np.exp(dataset_sample['data'][0].cpu().numpy()), subject_dataset.freqs)
        visualization.plot_spectrum_channels(dataset_sample['data'][0].cpu().numpy(), time_idx_from=0, time_idx_to=128 * 9)
        for key in dataset_sample.keys():
            print_value = dataset_sample[key].shape if hasattr(dataset_sample[key], "shape") and len(dataset_sample[key].shape) > 1 else dataset_sample[key]
            print(f'dataset_sample[{key}] {print_value}')
        print()
