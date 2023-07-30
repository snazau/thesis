import pickle

import mne
import numpy as np
import torch
from torch.utils.data._utils import collate

import custom_distribution
import eeg_reader

mne.set_log_level('error')


def check_if_in_segments(time_point, segments):
    return any([segment['start'] <= time_point <= segment['end'] for segment in segments])


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


def extract_fn_times(prediction_data, threshold, sfreq):
    time_idxs_start = prediction_data['time_idxs_start']
    probs = prediction_data['probs_wo_tta']
    labels = prediction_data['labels']

    fn_mask = ((labels == 1) & (probs <= threshold))
    fn_start_times = time_idxs_start[fn_mask] / sfreq

    return fn_start_times


class SubjectRandomDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file_path, seizures, samples_num, prediction_data_path=None, sample_duration=60, normalization=None, data_type='power_spectrum', transform=None):
        self.eeg_file_path = eeg_file_path
        self.raw = eeg_reader.EEGReader.read_eeg(self.eeg_file_path)
        self.seizures = seizures
        self.samples_num = samples_num
        self.prediction_data = None if prediction_data_path is None else pickle.load(open(prediction_data_path, 'rb'))
        self.sample_duration = sample_duration
        self.data_type = data_type
        self.normalization = normalization
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
            channel_name.replace('EEG ', ''): channel_idx for channel_idx, channel_name in enumerate(self.raw.info['ch_names'])
        }

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

        self.mask, self.sample_start_times, self.targets, self.raw_samples, self.time_idxs_start, self.time_idxs_end = self._generate_data()
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
        }

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
        mask = np.random.uniform(size=self.samples_num) > 0.5
        targets = mask.astype(np.int)
        sample_start_times = mask * seizure_times + (1 - mask) * normal_times

        if self.prediction_data is not None:
            fp_start_times = extract_fp_times(self.prediction_data, threshold=0.95, sfreq=self.raw.info['sfreq'])
            fn_start_times = extract_fn_times(self.prediction_data, threshold=0.95, sfreq=self.raw.info['sfreq'])

            fp_num_to_pick = min(len(fp_start_times), self.samples_num // 2)
            fn_num_to_pick = min(len(fn_start_times), self.samples_num // 2)

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

        raw_samples, time_idxs_start, time_idxs_end = generate_raw_samples(self.raw, sample_start_times, self.sample_duration)

        return mask, sample_start_times, targets, raw_samples, time_idxs_start, time_idxs_end

    def renew_data(self):
        (
            self.mask,
            self.sample_start_times,
            self.targets,
            self.raw_samples,
            self.time_idxs_start,
            self.time_idxs_end
        ) = self._generate_data()


class SubjectSequentialDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file_path, seizures, sample_duration=60, shift=None, normalization=None, data_type='power_spectrum', transform=None):
        self.eeg_file_path = eeg_file_path
        self.raw = eeg_reader.EEGReader.read_eeg(self.eeg_file_path)
        self.seizures = seizures
        self.sample_duration = sample_duration
        self.shift = shift if shift is not None else self.sample_duration
        self.data_type = data_type
        self.normalization = normalization
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
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def custom_collate_function(batch, data_type='power_spectrum', normalization=None, freqs=np.arange(1, 40.01, 0.1), sfreq=128, transform=None):
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
        power_spectrum = np.log(power_spectrum)

        if normalization == 'minmax':
            power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())
        elif normalization == 'meanstd':
            power_spectrum = (power_spectrum - power_spectrum.mean()) / power_spectrum.std()

        for sample_idx in range(power_spectrum.shape[0]):
            batch[sample_idx]['data'] = torch.from_numpy(power_spectrum[sample_idx]).float()
            if transform is not None:
                batch[sample_idx] = transform(batch[sample_idx])
            # batch[sample_idx]['data'] = torch.from_numpy(power_spectrum[sample_idx])
        # batch['data'] = torch.from_numpy(sample_data).float()

    batch = torch.utils.data.dataloader.default_collate(batch)

    return batch


def tta_collate_function(batch, tta_augs=tuple(), data_type='power_spectrum', normalization=None, freqs=np.arange(1, 40.01, 0.1), sfreq=128):
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
        power_spectrum = np.log(power_spectrum)

        if normalization == 'minmax':
            power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min())
        elif normalization == 'meanstd':
            power_spectrum = (power_spectrum - power_spectrum.mean()) / power_spectrum.std()

        for sample_idx in range(power_spectrum.shape[0]):
            batch[sample_idx]['data'] = torch.from_numpy(power_spectrum[sample_idx]).float()

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
    import os

    dataset_info_path = './data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = './data'
    # subject_key = 'data1/dataset28'
    # subject_key = 'data1/dataset2'
    # subject_key = 'data2/038tl Anonim-20190821_113559-20211123_004935'
    subject_key = 'data2/003tl Anonim-20200831_040629-20211122_135924'
    subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
    subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

    import time
    import utils.neural.training
    device = torch.device('cuda:0')
    model = utils.neural.training.get_model(model_name='resnet18', model_kwargs={'pretrained': True}).to(device)
    model.eval()

    subject_dataset = SubjectRandomDataset(
        subject_eeg_path,
        subject_seizures,
        samples_num=100,
        prediction_data_path=r'D:\Study\asp\thesis\implementation\experiments\renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug\predictions\data2\003tl Anonim-20200831_040629-20211122_135924.pickle',
        sample_duration=10,
        data_type='raw',
    )
    print(subject_dataset[0]['data'].shape)
    exit()
    # subject_dataset = SubjectSequentialDataset(subject_eeg_path, subject_seizures, sample_duration=10)
    loader = torch.utils.data.DataLoader(subject_dataset, batch_size=16)
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

    subject_dataset = SubjectRandomDataset(subject_eeg_path, subject_seizures, samples_num=100, sample_duration=10, data_type='raw')
    # subject_dataset = SubjectSequentialDataset(subject_eeg_path, subject_seizures, sample_duration=10, data_type='raw')
    from functools import partial
    collate_fn = partial(custom_collate_function, data_type='power_spectrum', normalization=None)
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
