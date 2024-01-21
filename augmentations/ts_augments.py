import random

import tsaug
import torch


class TimeWarp:
    def __init__(self, p, n_speed_change=5, max_speed_ratio=3):
        self.p = p
        self.aug = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)

    def __call__(self, sample):
        # data.shape = (1, C, T)
        if random.random() < self.p:
            data = torch.permute(sample['data'], (0, 2, 1)).cpu().numpy()
            data_augmented = self.aug.augment(data)
            data_augmented = torch.permute(torch.from_numpy(data_augmented), (0, 2, 1))
            sample['data'] = data_augmented
        return sample


class Drift:
    def __init__(self, p, max_drift=(0.0, 0.05), n_drift_points=3, normalize=True):
        self.p = p
        self.aug = tsaug.Drift(max_drift=max_drift, n_drift_points=n_drift_points, normalize=normalize)

    def __call__(self, sample):
        # data.shape = (1, C, T)
        if random.random() < self.p:
            data = torch.permute(sample['data'], (0, 2, 1)).cpu().numpy()
            data_augmented = self.aug.augment(data)
            data_augmented = torch.permute(torch.from_numpy(data_augmented), (0, 2, 1))
            sample['data'] = data_augmented
        return sample


class Crop:
    def __init__(self, p, size=1000, resize=1280):
        self.p = p
        self.aug = tsaug.Crop(size=size, resize=resize)

    def __call__(self, sample):
        # data.shape = (1, C, T)
        if random.random() < self.p:
            data = torch.permute(sample['data'], (0, 2, 1)).cpu().numpy()
            data_augmented = self.aug.augment(data)
            data_augmented = torch.permute(torch.from_numpy(data_augmented), (0, 2, 1))
            sample['data'] = data_augmented
        return sample


class AddNoise:
    def __init__(self, p, loc=0, scale=(5e-7, 2e-6), normalize=False):
        self.p = p
        self.aug = tsaug.AddNoise(loc=loc, scale=scale, normalize=normalize)

    def __call__(self, sample):
        # data.shape = (1, C, T)
        if random.random() < self.p:
            data = torch.permute(sample['data'], (0, 2, 1)).cpu().numpy()
            data_augmented = self.aug.augment(data)
            data_augmented = torch.permute(torch.from_numpy(data_augmented), (0, 2, 1))
            sample['data'] = data_augmented
        return sample


class AddNoiseBaseline:
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        assert 'baseline_mean' in sample

        # data.shape = (1, C, T)
        if random.random() < self.p:
            data = torch.permute(sample['data'], (0, 2, 1)).cpu().numpy()

            loc = np.zeros_like(data)
            scale = np.zeros_like(data) + np.abs(sample['baseline_mean'][..., 0])
            scale /= 10

            noise = np.random.normal(loc, scale)
            data_augmented = data + noise

            data_augmented = torch.permute(torch.from_numpy(data_augmented), (0, 2, 1))
            sample['data'] = data_augmented
        return sample


if __name__ == "__main__":
    import numpy as np
    import utils.neural.training
    utils.neural.training.set_seed(8, deterministic=True)

    try:
        tsaug_series = np.load('tsaug_sample.npy')
        # raise NotImplementedError
    except Exception as e:
        import json
        import os

        dataset_info_path = '../data/dataset_info.json'
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        data_dir = '../data'
        # subject_key = 'data1/dataset28'
        # subject_key = 'data1/dataset2'
        # subject_key = 'data2/038tl Anonim-20190821_113559-20211123_004935'
        subject_key = 'data2/037tl Anonim-20191020_110036-20211122_223805'
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        import datasets.datasets_static
        subject_dataset = datasets.datasets_static.SubjectRandomDataset(subject_eeg_path, subject_seizures, samples_num=100, sample_duration=10, normalization=None, data_type='raw')

        sample = subject_dataset[0]  # (B, C, T)
        print('orig', sample['data'].shape)

        tsaug_series = np.transpose(sample['data'].detach().cpu().numpy(), (0, 2, 1))
        with open('tsaug_sample.npy', 'wb') as f:
            np.save(f, tsaug_series)

    print(f'tsaug_series = {tsaug_series.shape}')
    tsaug_series = np.concatenate([tsaug_series, tsaug_series, tsaug_series], axis=0)
    print(f'tsaug_series = {tsaug_series.shape}')

    import matplotlib.pyplot as plt
    from tsaug.visualization import plot
    plot(tsaug_series[..., :2])
    # plt.show()

    tsaug_series_aug = tsaug.AddNoise(loc=0, scale=(5e-7, 2e-6), normalize=False).augment(tsaug_series)
    # tsaug_series_aug = tsaug.AddNoise(loc=0, scale=(0.001, 0.01), normalize=True).augment(tsaug_series)
    plot(tsaug_series_aug[..., :2])

    # tsaug_series_aug = tsaug.Crop(size=1000, resize=1280).augment(tsaug_series)
    # plot(tsaug_series_aug[..., :2])
    # plt.show()

    # tsaug_series_aug = tsaug.Drift(max_drift=(0.0, 0.05), n_drift_points=3, normalize=True).augment(tsaug_series)
    # plot(tsaug_series_aug[..., :2])
    # plt.show()

    # tsaug_series_aug = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(tsaug_series)
    # plot(tsaug_series_aug[..., :2])
    # plt.show()

    print(f'tsaug_series_aug = {tsaug_series_aug.shape}')

    aug = AddNoise(p=1)
    sample = {'data': torch.permute(torch.from_numpy(tsaug_series), (0, 2, 1))}
    sample_aug = aug(sample)
    print('augmented', sample_aug['data'].shape)
    plt.show()
