import pprint
from functools import partial

import numpy as np
import pickle
import torch
import torchvision
import tqdm

import augmentations.flip
import augmentations.spec_augment
import datasets.datasets_static
import utils.neural.training


def predict(model, subject_key, subject_eeg_path, subject_seizures, config):
    model.eval()

    if 'stats_dir' in config and config['stats_dir'] is not None:
        stats_path = os.path.join(config['stats_dir'], subject_key + '.npy')
    else:
        stats_path = None

    subject_dataset = datasets.datasets_static.SubjectSequentialDataset(
        subject_eeg_path,
        subject_seizures,
        stats_path=stats_path,
        sample_duration=config['sample_duration'],
        shift=config['shift'],
        data_type='raw',
        baseline_correction=False,
    )
    collate_fn = partial(
        datasets.datasets_static.custom_collate_function,
        normalization=config['normalization'],
        transform=None,
        baseline_correction=False,
        data_type=config['data_type'],
    )
    # collate_fn = partial(
    #     datasets.datasets_static.tta_collate_function,
    #     tta_augs=config['tta_augs'],
    #     normalization=config['normalization'],
    #     data_type=config['data_type'],
    #     baseline_correction=config['baseline_correction'],
    # )
    loader = torch.utils.data.DataLoader(subject_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=False)

    device = config['device']

    data_keys = ['data'] + [f'data_aug{aug_idx:03}' for aug_idx in range(len(config['tta_augs']))]

    all_labels = np.array([])
    all_probs_wo_tta = np.array([])
    all_probs = np.array([])
    all_time_idxs_start = np.array([])
    all_time_idxs_end = np.array([])
    with tqdm.tqdm(loader) as tqdm_wrapper:
        for batch_idx, batch in enumerate(tqdm_wrapper):
            time_idxs_start = batch['time_idx_start']
            time_idxs_end = batch['time_idx_end']
            labels = batch['target'].to(device)
            probs_tta = torch.zeros_like(labels, dtype=torch.float32)

            for data_key in data_keys:
                inputs = batch[data_key].to(device)
                with torch.no_grad():
                    outputs = torch.squeeze(model(inputs))
                    probs = torch.sigmoid(outputs)
                    probs_tta += probs
                    if data_key == 'data':
                        probs_wo_tta = probs.clone()
            probs_tta = probs_tta / len(data_keys)

            all_labels = np.concatenate([all_labels, labels.cpu().detach().numpy()])
            all_probs = np.concatenate([all_probs, probs_tta.cpu().detach().numpy()])
            all_probs_wo_tta = np.concatenate([all_probs_wo_tta, probs_wo_tta.cpu().detach().numpy()])
            all_time_idxs_start = np.concatenate([all_time_idxs_start, time_idxs_start.cpu().detach().numpy()])
            all_time_idxs_end = np.concatenate([all_time_idxs_end, time_idxs_end.cpu().detach().numpy()])

            # if batch_idx > 2:
            #     break

            tqdm_wrapper.set_postfix()

    metrics = utils.neural.training.calc_metrics_diff_thresholds(all_probs, all_labels)
    metrics_wo_tta = utils.neural.training.calc_metrics_diff_thresholds(all_probs_wo_tta, all_labels)
    prediction_data = {
        'subject_eeg_path': subject_eeg_path,
        'subject_seizures': subject_seizures,
        'probs': all_probs,
        'probs_wo_tta': all_probs_wo_tta,
        'labels': all_labels,
        'time_idxs_start': all_time_idxs_start,
        'time_idxs_end': all_time_idxs_end,
        'min_time_in_seconds': subject_dataset.raw.times.min(),
        'max_time_in_seconds': subject_dataset.raw.times.max(),
        'metrics': metrics,
        'metrics_wo_tta': metrics_wo_tta,
    }

    print(f'metrics_wo_tta = {metrics_wo_tta}\n\n')

    return prediction_data


if __name__ == '__main__':
    import json
    import os

    dataset_info_path = './data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = './data'

    subject_keys_exclude = [
        'data1/dataset11',
        'data1/dataset13',
        'data1/dataset22',
        'data1/dataset27',
        'data2/004tl Anonim-20200926_213911-20211122_144051',
        'data2/004tl Anonim-20200929_081036-20211122_144552',
        'data2/006tl Anonim-20210208_063816-20211122_154113',
        'data2/009tl Anonim-20200213_130213-20211122_160907',
        'data2/009tl Anonim-20200215_021624-20211122_161231',
        'data2/015tl Anonim-20201116_134129-20211122_161958',
        'data2/017tl Anonim-20200708_143949-20211122_163253',
        'data2/018tl Anonim-20201212_101651-20211122_163821',
        'data2/020tl Anonim-20201218_194126-20211122_171755',
        'data2/026tl Anonim-20210302_093747-20211122_175031',
        'data2/037tl Anonim-20201102_102725-20211123_003801',
        'data2/039tl Anonim-20200607_035937-20211123_005921',
    ]

    # subject_key = 'data1/dataset28'
    # subject_key = 'data1/dataset2'
    # subject_keys = ['data2/038tl Anonim-20190821_113559-20211123_004935', 'data1/dataset27']
    # subject_keys = ['data2/038tl Anonim-20190821_113559-20211123_004935']
    subject_keys = [
        # stage_1
        # part1
        'data2/038tl Anonim-20190821_113559-20211123_004935',
        'data2/027tl Anonim-20200309_195746-20211122_175315',
        'data1/dataset27',
        'data1/dataset14',
        'data2/036tl Anonim-20201224_124349-20211122_181415',
        'data2/041tl Anonim-20201115_222025-20211123_011114',
        'data1/dataset24',
        'data2/026tl Anonim-20210301_013744-20211122_174658',

        'data2/020tl Anonim-20201218_071731-20211122_171454', 'data1/dataset13',
        'data2/018tl Anonim-20201211_130036-20211122_163611',
        'data2/038tl Anonim-20190822_155119-20211123_005457',
        'data2/025tl Anonim-20210128_233211-20211122_173425',
        'data2/015tl Anonim-20201116_134129-20211122_161958',

        # part2
        'data1/dataset3',
        'data2/027tl Anonim-20200310_035747-20211122_175503',
        'data2/002tl Anonim-20200826_124513-20211122_135804', 'data1/dataset23',
        'data2/022tl Anonim-20201210_132636-20211122_172649',
        'data1/dataset6', 'data1/dataset11',
        'data2/021tl Anonim-20201223_085255-20211122_172126', 'data1/dataset28',

        # part 3
        'data2/008tl Anonim-20210204_131328-20211122_160417',
        'data2/003tl Anonim-20200831_120629-20211122_140327',
        'data2/025tl Anonim-20210129_073208-20211122_173728',
        'data2/038tl Anonim-20190822_131550-20211123_005257', 'data1/dataset2',

        'data1/dataset22',
        'data2/040tl Anonim-20200421_100248-20211123_010147',
        'data2/020tl Anonim-20201216_073813-20211122_171341',
        'data2/019tl Anonim-20201213_072025-20211122_165918',

        'data2/003tl Anonim-20200831_040629-20211122_135924',
        'data2/006tl Anonim-20210208_063816-20211122_154113', 'data1/dataset4', 'data1/dataset20',
        'data2/035tl Anonim-20210324_231349-20211122_223059', 'data1/dataset16',
        'data2/035tl Anonim-20210324_151211-20211122_222545',
        'data2/038tl Anonim-20190822_203419-20211123_005705', 'data1/dataset25', 'data1/dataset5',
        'data2/018tl Anonim-20201215_022951-20211122_165644',
        'data1/dataset1',  # 3h
        'data1/dataset12',  # Failed because of memory

        # stage_2
        # 'data2/003tl Anonim-20200831_120629-20211122_140327',
        # 'data1/dataset12',
        # 'data2/025tl Anonim-20210129_073208-20211122_173728',
        # 'data2/038tl Anonim-20190822_131550-20211123_005257', 'data1/dataset2',
        # 'data1/dataset22',
        # 'data2/040tl Anonim-20200421_100248-20211123_010147',
        # 'data2/020tl Anonim-20201216_073813-20211122_171341',
        # 'data2/019tl Anonim-20201213_072025-20211122_165918',
        # 'data2/003tl Anonim-20200831_040629-20211122_135924',
        # 'data2/006tl Anonim-20210208_063816-20211122_154113', 'data1/dataset4', 'data1/dataset20',
        # 'data2/035tl Anonim-20210324_231349-20211122_223059', 'data1/dataset16',
        # 'data2/035tl Anonim-20210324_151211-20211122_222545',
        # 'data2/038tl Anonim-20190822_203419-20211123_005705', 'data1/dataset25', 'data1/dataset5',
        # 'data2/018tl Anonim-20201215_022951-20211122_165644',

        # # stage_1 train
        # 'data2/011tl Anonim-20200118_041022-20211122_161616',
        # 'data2/016tl Anonim-20210127_214910-20211122_162657',
        # 'data2/022tl Anonim-20201209_132645-20211122_172422',
        # 'data2/002tl Anonim-20200826_044516-20211122_135439',
        # 'data2/004tl Anonim-20200929_081036-20211122_144552', 'data1/dataset29',
        # 'data2/026tl Anonim-20210227_214223-20211122_174442', 'data1/dataset19', 'data1/dataset21',
        # 'data2/020tl Anonim-20201218_194126-20211122_171755',
        # 'data2/034tl Anonim-20210304_071124-20211122_222211', 'data1/dataset17',
        # 'data2/033tl Anonim-20200114_085935-20211122_180917', 'data1/dataset10',
        # 'data1/dataset9',
        # 'data2/028tl Anonim-20191014_212520-20211122_175854',
        # 'data2/030tl Anonim-20190910_110631-20211122_180335',
        # 'data2/016tl Anonim-20210128_054911-20211122_163013',
        # 'data2/023tl Anonim-20210110_080440-20211122_173058',
        # 'data1/dataset15',
        # 'data2/008tl Anonim-20210204_211327-20211122_160546',
        # 'data2/037tl Anonim-20191020_110036-20211122_223805', 'data1/dataset7',
        # 'data2/006tl Anonim-20210209_144403-20211122_155146',
        # 'data2/006tl Anonim-20210208_144401-20211122_154504',
        # # 'data2/009tl Anonim-20200215_021624-20211122_161231',  # 1000 sec seizure in the end
        # 'data2/035tl Anonim-20210326_231343-20211122_223404',
        # 'data2/009tl Anonim-20200213_130213-20211122_160907',
        # 'data2/041tl Anonim-20201112_194437-20211123_010804', 'data1/dataset26',
        # 'data2/018tl Anonim-20201212_101651-20211122_163821',
        # 'data2/037tl Anonim-20201102_102725-20211123_003801',
        # # 'data2/004tl Anonim-20200926_213911-20211122_144051',  # 1000 sec seizure in the end
        # 'data1/dataset8',
        # 'data1/dataset18',
        # 'data2/026tl Anonim-20210302_093747-20211122_175031',
    ]
    subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    for subject_idx, subject_key in enumerate(subject_keys):
        print(subject_key)
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        ttg_augs = list()
        # if subject_idx > 0:
        #     ttg_augs = list()
        # else:
        #     ttg_augs = [
        #         augmentations.flip.TimeFlip(p=1.0),
        #         # augmentations.flip.EEGChannelsFlip(p=1.0),
        #         augmentations.spec_augment.SpecAugment(
        #             p_aug=1,
        #             p_mask=0.5,
        #             max_freq_mask_width=40,
        #             max_time_mask_width=128,
        #             num_masks_per_channel=1,
        #             replace_with_zero=False,
        #         ),
        #         torchvision.transforms.Compose(
        #             [
        #                 augmentations.flip.TimeFlip(p=1),
        #                 augmentations.spec_augment.SpecAugment(
        #                     p_aug=1,
        #                     p_mask=0.5,
        #                     max_freq_mask_width=40,
        #                     max_time_mask_width=128,
        #                     num_masks_per_channel=1,
        #                     replace_with_zero=False,
        #                 ),
        #             ]
        #         )
        #     ]

        # stage_1 & stage_2
        # renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug & renset18_2nd_stage_MixUp_SpecTimeFlipEEGFlipAug
        # 30072023_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_continue
        prediction_config = {
            'sfreq': 128,
            'sample_duration': 10,
            'shift': 10,
            'data_type': 'power_spectrum',
            'normalization': 'meanstd',
            'batch_size': 16,
            'device': torch.device('cuda:0'),
            'tta_augs': ttg_augs,
            'baseline_correction': False,
        }

        # # 08082023_efficientnet_b0_all_subjects_MixUp_TimeSeriesAug_raw
        # prediction_config = {
        #     'sfreq': 128,
        #     'sample_duration': 10,
        #     'shift': 10,
        #     'data_type': 'raw',
        #     'normalization': 'meanstd',
        #     'batch_size': 16,
        #     'device': torch.device('cuda:0'),
        #     'tta_augs': ttg_augs,
        #     'baseline_correction': False,
        # }

        # # 14062023_resnet18_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm
        # prediction_config = {
        #     'sfreq': 128,
        #     'sample_duration': 10,
        #     'shift': 10,
        #     'data_type': 'power_spectrum',
        #     'normalization': 'minmax',
        #     'baseline_correction': True,
        #     'batch_size': 16,
        #     'device': torch.device('cuda:0'),
        #     'tta_augs': ttg_augs,
        # }

        # # 20230821_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_cwt_meanstd
        # prediction_config = {
        #     'stats_dir': 'D:\\Study\\asp\\thesis\\implementation\\data\\cwt_log_stats_v2',
        #     'sfreq': 128,
        #     'sample_duration': 10,
        #     'shift': 10,
        #     'data_type': 'power_spectrum',
        #     'normalization': 'cwt_meanstd',
        #     'batch_size': 16,
        #     'device': torch.device('cuda:0'),
        #     'tta_augs': ttg_augs,
        #     'baseline_correction': False,
        # }

        experiments_dir = r'D:\Study\asp\thesis\implementation\experiments'
        # experiment_name = 'renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug'  # stage_1
        # experiment_name = 'renset18_2nd_stage_MixUp_SpecTimeFlipEEGFlipAug'  # stage_2
        # experiment_name = '14062023_resnet18_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm'
        # experiment_name = '30072023_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_continue'
        # experiment_name = '08082023_efficientnet_b0_all_subjects_MixUp_TimeSeriesAug_raw'
        # experiment_name = '14082023_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_preprocessed_ocsvm'
        # experiment_name = '20230821_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_cwt_meanstd'
        # experiment_name = '20230912_efficientnet_b0_all_subjects_SpecTimeFlipEEGFlipAug_log_power_BCELossWithTimeToClosestSeizure'
        experiment_name = '20230925_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_16excluded'
        experiment_dir = os.path.join(experiments_dir, experiment_name)
        checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best.pth.tar')

        state_dict = torch.load(checkpoint_path)['model']['state_dict']
        # model = utils.neural.training.get_model('resnet18', model_kwargs=dict())
        model = utils.neural.training.get_model('efficientnet_b0', model_kwargs=dict(pretrained=True))
        # model = utils.neural.training.get_model('efficientnet_b0_1channel', model_kwargs=dict(pretrained=True))
        model.load_state_dict(state_dict)
        model = model.to(prediction_config['device'])

        predction_data = predict(model, subject_key, subject_eeg_path, subject_seizures, prediction_config)
        predictions_dir = os.path.join(experiment_dir, 'predictions', subject_key.split('/')[0])
        os.makedirs(predictions_dir, exist_ok=True)
        subject_name = subject_key.split('/')[-1]
        predction_data_path = os.path.join(predictions_dir, f'{subject_name}.pickle')
        with open(predction_data_path, 'wb') as fh:
            pickle.dump(predction_data, fh, pickle.HIGHEST_PROTOCOL)
