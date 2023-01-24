import random

import torch


class TimeFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        # data.shape = (C, F, T)
        if random.random() < self.p:
            print(sample['data'].shape)
            sample['data'] = torch.flip(sample['data'], dims=(2, ))
        return sample


class EEGChannelsFlip:
    def __init__(self, p):
        self.p = p
        self.channels_to_swap = [
            ('Fp1', 'Fp2'),
            ('F7', 'F8'),
            ('F3', 'F4'),
            ('T3', 'T4'),
            ('C3', 'C4'),
            ('T5', 'T6'),
            ('P3', 'P4'),
            ('O1', 'O2'),
            ('F9', 'F10'),
            ('P9', 'P10'),
            ('T9', 'T10'),
        ]

    def __call__(self, sample):
        # import numpy as np
        # sample['data'] = np.zeros_like(sample['data'])
        # for i in range(sample['data'].shape[0]):
        #     sample['data'][i] = torch.from_numpy(draw_text(sample['data'][i], x=50, y=300, text=f'{i:02}'))

        if random.random() < self.p:
            channel_name_to_idx = sample['channel_name_to_idx']
            for channels_pair in self.channels_to_swap:

                channel_name_one = channels_pair[0]
                channel_name_two = channels_pair[1]
                channel_idx_one = channel_name_to_idx[channel_name_one]
                channel_idx_two = channel_name_to_idx[channel_name_two]

                # channel_temp = sample['data'][channel_idx_one].copy()
                channel_temp = sample['data'][channel_idx_one].clone()
                sample['data'][channel_idx_one] = sample['data'][channel_idx_two]
                sample['data'][channel_idx_two] = channel_temp

                # print(f'swapped channel_idx_one = {channel_idx_one} ({channel_name_one}) channel_idx_two = {channel_idx_two} ({channel_name_two})')
        return sample


def draw_text(image, x, y, text):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (x, y)
    fontScale = 12
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(image, text, position, font, fontScale, color, thickness, cv2.LINE_AA)
    return image


if __name__ == "__main__":
    import utils.neural.training
    utils.neural.training.set_seed(8, deterministic=True)

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

    import datasets
    subject_dataset = datasets.SubjectRandomDataset(subject_eeg_path, subject_seizures, samples_num=100, sample_duration=10, normalization='meanstd')

    sample = subject_dataset[0]
    # aug = TimeFlip(p=1)
    aug = EEGChannelsFlip(p=0)
    sample = aug(sample)

    import numpy as np
    import visualization
    # visualization.plot_spectrum_averaged(np.exp(sample['data']), subject_dataset.freqs)
    # visualization.plot_spectrum_channels(sample['data'], time_idx_from=0, time_idx_to=128 * 9)
    visualization.plot_spectrum_averaged(np.exp(sample['data'].cpu().numpy()), subject_dataset.freqs)
    visualization.plot_spectrum_channels(sample['data'].cpu().numpy(), time_idx_from=0, time_idx_to=128 * 9)

