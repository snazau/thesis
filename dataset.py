import mne
import numpy as np
import torch

import custom_distribution
import eeg_reader


class SubjectDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file_path, seizures, samples_num, sample_duration=60, normalization=None, data_type='power_spectrum', transform=None):
        self.raw = eeg_reader.EEGReader.read_eeg(eeg_file_path)
        self.seizures = seizures
        self.samples_num = samples_num
        self.sample_duration = sample_duration
        self.data_type = data_type
        self.normalization = normalization
        self.transform = transform

        # drop unnecessary channels
        if 'data1' in eeg_file_path:
            channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
        elif 'data2' in subject_key:
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
        sample_start_times = self.mask * seizure_times + (1 - self.mask) * normal_times
        # print(f'len(sample_start_times) = {len(sample_start_times)}')
        # print(f'len(list(set(sample_start_times))) = {len(list(set(sample_start_times)))}')
        # sample_start_times[0] = 23860

        self.targets = self.mask.astype(np.int)
        self.raw_samples = self.generate_raw_samples(sample_start_times)
        # print(f'self.raw_samples.shape = {self.raw_samples.shape}')

        if self.data_type == 'raw':
            self.samples = np.expand_dims(self.raw_samples, axis=1)
            return

        if self.data_type == 'power_spectrum':
            # wavelet (morlet) transform
            freqs = np.arange(1, 40.01, 0.1)
            power_spectrum = mne.time_frequency.tfr_array_morlet(self.raw_samples, sfreq=self.raw.info['sfreq'], freqs=freqs, n_cycles=freqs, output='power', n_jobs=-1)
            # print(power.shape, power.min(), power.mean(), power.max())

            # TODO: done normalization correctly
            #  Support link: https://colab.research.google.com/github/enzokro/clck10/blob/master/_notebooks/2020-09-10-Normalizing-spectrograms-for-deep-learning.ipynb
            if self.normalization == 'log':
                power_spectrum = np.log(power_spectrum)
            elif self.normalization == 'minmax':
                power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())
            elif self.normalization == 'meanstd':
                power_spectrum = (power_spectrum - power_spectrum.mean()) / power_spectrum.std()

            self.samples = power_spectrum

            import visualization
            for idx in range(len(self.targets)):
                print(self.targets[idx])
                visualization.plot_spectrum_averaged(power_spectrum[idx], freqs)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        sample = {
            'data': self.samples[idx],
            'target': self.target[idx],
        }
        return sample

    def generate_raw_samples(self, sample_start_times):
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

        raw_data = self.raw.get_data()

        samples_num = len(sample_start_times)
        frequency = self.raw.info['sfreq']
        sample_len_in_idxs = int(self.sample_duration * frequency)
        channels_num = len(self.raw.info['ch_names'])

        samples = np.zeros((samples_num, channels_num, sample_len_in_idxs))
        for sample_idx, sample_start_time in enumerate(sample_start_times):
            start_idx = int(frequency * sample_start_time)
            end_idx = start_idx + sample_len_in_idxs
            samples[sample_idx] = raw_data[:, start_idx:end_idx]

        return samples


if __name__ == '__main__':
    import json
    import os

    dataset_info_path = './data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = './data'
    # subject_key = 'data1/dataset28'
    subject_key = 'data1/dataset2'
    # subject_key = 'data2/038tl Anonim-20190821_113559-20211123_004935'
    subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
    subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

    subject_dataset = SubjectDataset(subject_eeg_path, subject_seizures, samples_num=100, sample_duration=10)
    dataset_sample = subject_dataset[0]
    for key in dataset_sample.keys():
        print(f'dataset_sample[{key}] {dataset_sample[key].shape}')
