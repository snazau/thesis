import random

import numpy as np


class SpecAugment:
    def __init__(self, p_aug, p_mask, max_freq_mask_width, max_time_mask_width, num_masks_per_channel, replace_with_zero):
        assert 0 <= p_aug <= 1
        assert 0 <= p_mask <= 1

        self.p_aug = p_aug
        self.p_mask = p_mask
        self.max_freq_mask_width = max_freq_mask_width
        self.max_time_mask_width = max_time_mask_width
        self.num_masks_per_channel = num_masks_per_channel
        self.replace_with_zero = replace_with_zero

    def __call__(self, sample):
        if random.random() < self.p_aug:
            data = sample['data']
            data = self.freq_mask(data)
            data = self.time_mask(data)
            sample['data'] = data
        return sample

    def freq_mask(self, data):
        # data.shape = (C, F, T)

        freq_num = data.shape[1]
        channels_num = data.shape[0]

        cloned = data.clone()
        for channel_num in range(channels_num):
            for mask_idx in range(self.num_masks_per_channel):
                if random.random() >= self.p_mask:
                    continue

                mask_width = int(np.random.uniform(0, self.max_freq_mask_width))  # [0, F)
                mask_start_idx = random.randint(0, freq_num - mask_width)  # [0, v - f)
                if self.replace_with_zero is True:
                    cloned[channel_num, mask_start_idx:mask_start_idx + mask_width] = 0
                else:
                    cloned[channel_num, mask_start_idx:mask_start_idx + mask_width] = cloned.mean()

        return cloned

    def time_mask(self, spec):
        # spec.shape = (C, F, T)

        time_num = spec.shape[2]
        channels_num = spec.shape[0]

        cloned = spec.clone()
        for channel_num in range(channels_num):
            for mask_idx in range(self.num_masks_per_channel):
                if random.random() >= self.p_mask:
                    continue

                mask_width = int(np.random.uniform(0, self.max_time_mask_width))  # [0, T)
                mask_start_idx = random.randint(0, time_num - mask_width)  # [0, tau - t)
                if self.replace_with_zero is True:
                    cloned[channel_num, :, mask_start_idx:mask_start_idx + mask_width] = 0
                else:
                    cloned[channel_num, :, mask_start_idx:mask_start_idx + mask_width] = cloned.mean()
        return cloned


if __name__ == "__main__":
    import json
    import os

    dataset_info_path = '../data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = '../data'
    # subject_key = 'data1/dataset28'
    # subject_key = 'data1/dataset2'
    subject_key = 'data2/038tl Anonim-20190821_113559-20211123_004935'
    # subject_key = 'data2/037tl Anonim-20191020_110036-20211122_223805'
    subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
    subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

    import datasets
    subject_dataset = datasets.SubjectRandomDataset(subject_eeg_path, subject_seizures, samples_num=100, sample_duration=10)

    sample = subject_dataset[0]
    spec_aug = SpecAugment(
        p_aug=1,
        p_mask=1,
        max_freq_mask_width=40 * 2,
        max_time_mask_width=128 * 2,
        num_masks_per_channel=1,
        replace_with_zero=False,
    )
    sample = spec_aug(sample)

    import visualization
    # visualization.plot_spectrum_averaged(np.exp(sample['data'].cpu().numpy()), subject_dataset.freqs)
    # visualization.plot_spectrum_channels(sample['data'].cpu().numpy(), time_idx_from=0, time_idx_to=128 * 10)
    visualization.plot_spectrum_channel(sample['data'].cpu().numpy(), channel_idx=0, time_idx_from=0, time_idx_to=128 * 10)
