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

    subject_dataset = datasets.datasets_static.SubjectSequentialDataset(
        subject_eeg_path,
        subject_seizures,
        stats_path=None,
        sample_duration=config['sample_duration'],
        shift=config['shift'],
        data_type='raw',
        baseline_correction=False,
        force_calc_baseline_stats=config['force_calc_baseline_stats'],
    )
    collate_fn = partial(
        datasets.datasets_static.custom_collate_function,
        normalization=config['normalization'],
        transform=None,
        baseline_correction=config['baseline_correction'],
        data_type=config['data_type'],
    )
    loader = torch.utils.data.DataLoader(
        subject_dataset,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        shuffle=False,
    )

    device = config['device']

    h_prev = torch.zeros((model.rnn.num_layers, 1, model.rnn.hidden_size), dtype=torch.float32, device=device)
    c_prev = torch.zeros((model.rnn.num_layers, 1, model.rnn.hidden_size), dtype=torch.float32, device=device)

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
            inputs = batch['data'].to(device)

            # Turn batch dimension into sequence dimension, batch dim set equal 1
            inputs = torch.unsqueeze(inputs, dim=0)  # (1, B, C, H, W)

            with torch.no_grad():
                outputs, h_curr, c_curr = model(inputs, h_prev, c_prev)
                outputs = torch.squeeze(outputs)
                probs = torch.sigmoid(outputs)
            h_prev, c_prev = h_curr, c_curr

            all_labels = np.concatenate([all_labels, labels.cpu().detach().numpy()])
            all_probs = np.concatenate([all_probs, probs.cpu().detach().numpy()])
            all_time_idxs_start = np.concatenate([all_time_idxs_start, time_idxs_start.cpu().detach().numpy()])
            all_time_idxs_end = np.concatenate([all_time_idxs_end, time_idxs_end.cpu().detach().numpy()])

            # if batch_idx > 0:
            #     break

            tqdm_wrapper.set_postfix()

    metrics = utils.neural.training.calc_metrics_diff_thresholds(all_probs, all_labels)
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
    }

    print(f'metrics = {metrics}\n\n')

    return prediction_data


if __name__ == '__main__':
    import json
    import os

    dataset_info_path = './data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    data_dir = './data'
    experiments_dir = r'D:\Study\asp\thesis\implementation\experiments'
    experiment_name = '20231012_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded'
    # experiment_name = '20231008_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_v2_16excluded_wo_baseline_correction'
    experiment_dir = os.path.join(experiments_dir, experiment_name)

    # # 20231005_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded_wo_baseline_correction
    # # 20231008_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_v2_16excluded_wo_baseline_correction
    # prediction_config = {
    #     'sfreq': 128,
    #     'sample_duration': 10,
    #     'shift': 10,
    #     'data_type': 'raw',
    #     'normalization': None,
    #     'batch_size': 16,
    #     'device': torch.device('cuda:0'),
    #     'baseline_correction': False,
    #     'force_calc_baseline_stats': True,
    # }

    # 20231005_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded
    prediction_config = {
        'sfreq': 128,
        'sample_duration': 10,
        'shift': 10,
        'data_type': 'raw',
        'normalization': None,
        'batch_size': 16,
        'device': torch.device('cuda:0'),
        'baseline_correction': True,
        'force_calc_baseline_stats': True,
    }

    model_kwargs = {
        'cnn_backbone': 'EEGResNetCustomRaw',
        'cnn_backbone_kwargs': {
            'input_dim': 1,
        },
        'cnn_backbone_pretrained_path': None,
        'rnn_hidden_size': 128,
        'rnn_layers_num': 1,
    }
    model = utils.neural.training.get_model('CRNN', model_kwargs)
    model = model.to(prediction_config['device'])

    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best.pth.tar')
    state_dict = torch.load(checkpoint_path)['model']['state_dict']
    model.load_state_dict(state_dict)

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
    subject_keys = [
        # 'data2/038tl Anonim-20190821_113559-20211123_004935',  # metrics = {'auc_roc': {'auc_roc': 0.5396292164150738}, 'auc_pr': {'auc_pr': 0.0058062801597716885}, 'brier_score': {'brier_score': 0.9611422530369071}, 'calib_score': {'calib_score': 0.9831918345843341}, 'accuracy_score': {'01': 0.005738414898224339, '05': 0.005738414898224339, '10': 0.005738414898224339, '15': 0.005738414898224339, '20': 0.005738414898224339, '25': 0.005738414898224339, '30': 0.005738414898224339, '35': 0.005738414898224339, '40': 0.005738414898224339, '45': 0.005738414898224339, '50': 0.005738414898224339, '55': 0.005738414898224339, '60': 0.005738414898224339, '65': 0.005738414898224339, '70': 0.005738414898224339, '75': 0.005738414898224339, '80': 0.005738414898224339, '85': 0.005738414898224339, '90': 0.005738414898224339, '95': 0.005738414898224339, '99': 0.9942615851017756}, 'f1_score': {'01': 0.011411346754225428, '05': 0.011411346754225428, '10': 0.011411346754225428, '15': 0.011411346754225428, '20': 0.011411346754225428, '25': 0.011411346754225428, '30': 0.011411346754225428, '35': 0.011411346754225428, '40': 0.011411346754225428, '45': 0.011411346754225428, '50': 0.011411346754225428, '55': 0.011411346754225428, '60': 0.011411346754225428, '65': 0.011411346754225428, '70': 0.011411346754225428, '75': 0.011411346754225428, '80': 0.011411346754225428, '85': 0.011411346754225428, '90': 0.011411346754225428, '95': 0.011411346754225428, '99': 0.0}, 'precision_score': {'01': 0.005738414898224339, '05': 0.005738414898224339, '10': 0.005738414898224339, '15': 0.005738414898224339, '20': 0.005738414898224339, '25': 0.005738414898224339, '30': 0.005738414898224339, '35': 0.005738414898224339, '40': 0.005738414898224339, '45': 0.005738414898224339, '50': 0.005738414898224339, '55': 0.005738414898224339, '60': 0.005738414898224339, '65': 0.005738414898224339, '70': 0.005738414898224339, '75': 0.005738414898224339, '80': 0.005738414898224339, '85': 0.005738414898224339, '90': 0.005738414898224339, '95': 0.005738414898224339, '99': 0.0}, 'recall_score': {'01': 1.0, '05': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '25': 1.0, '30': 1.0, '35': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '55': 1.0, '60': 1.0, '65': 1.0, '70': 1.0, '75': 1.0, '80': 1.0, '85': 1.0, '90': 1.0, '95': 1.0, '99': 0.0}, 'cohen_kappa_score': {'01': 0.0, '05': 0.0, '10': 0.0, '15': 0.0, '20': 0.0, '25': 0.0, '30': 0.0, '35': 0.0, '40': 0.0, '45': 0.0, '50': 0.0, '55': 0.0, '60': 0.0, '65': 0.0, '70': 0.0, '75': 0.0, '80': 0.0, '85': 0.0, '90': 0.0, '95': 0.0, '99': 0.0}}
        # 'data2/011tl Anonim-20200118_041022-20211122_161616',  # metrics = {'auc_roc': {'auc_roc': 0.7144502923976608}, 'auc_pr': {'auc_pr': 0.022273617630819503}, 'brier_score': {'brier_score': 0.9563569767631006}, 'calib_score': {'calib_score': 0.9710736886611959}, 'accuracy_score': {'01': 0.010416666666666666, '05': 0.010416666666666666, '10': 0.010416666666666666, '15': 0.010416666666666666, '20': 0.010416666666666666, '25': 0.010416666666666666, '30': 0.010416666666666666, '35': 0.010416666666666666, '40': 0.010416666666666666, '45': 0.010416666666666666, '50': 0.010416666666666666, '55': 0.010416666666666666, '60': 0.010416666666666666, '65': 0.010416666666666666, '70': 0.010416666666666666, '75': 0.010416666666666666, '80': 0.010416666666666666, '85': 0.010416666666666666, '90': 0.010416666666666666, '95': 0.010416666666666666, '99': 0.9895833333333334}, 'f1_score': {'01': 0.020618556701030924, '05': 0.020618556701030924, '10': 0.020618556701030924, '15': 0.020618556701030924, '20': 0.020618556701030924, '25': 0.020618556701030924, '30': 0.020618556701030924, '35': 0.020618556701030924, '40': 0.020618556701030924, '45': 0.020618556701030924, '50': 0.020618556701030924, '55': 0.020618556701030924, '60': 0.020618556701030924, '65': 0.020618556701030924, '70': 0.020618556701030924, '75': 0.020618556701030924, '80': 0.020618556701030924, '85': 0.020618556701030924, '90': 0.020618556701030924, '95': 0.020618556701030924, '99': 0.0}, 'precision_score': {'01': 0.010416666666666666, '05': 0.010416666666666666, '10': 0.010416666666666666, '15': 0.010416666666666666, '20': 0.010416666666666666, '25': 0.010416666666666666, '30': 0.010416666666666666, '35': 0.010416666666666666, '40': 0.010416666666666666, '45': 0.010416666666666666, '50': 0.010416666666666666, '55': 0.010416666666666666, '60': 0.010416666666666666, '65': 0.010416666666666666, '70': 0.010416666666666666, '75': 0.010416666666666666, '80': 0.010416666666666666, '85': 0.010416666666666666, '90': 0.010416666666666666, '95': 0.010416666666666666, '99': 0.0}, 'recall_score': {'01': 1.0, '05': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '25': 1.0, '30': 1.0, '35': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '55': 1.0, '60': 1.0, '65': 1.0, '70': 1.0, '75': 1.0, '80': 1.0, '85': 1.0, '90': 1.0, '95': 1.0, '99': 0.0}, 'cohen_kappa_score': {'01': 0.0, '05': 0.0, '10': 0.0, '15': 0.0, '20': 0.0, '25': 0.0, '30': 0.0, '35': 0.0, '40': 0.0, '45': 0.0, '50': 0.0, '55': 0.0, '60': 0.0, '65': 0.0, '70': 0.0, '75': 0.0, '80': 0.0, '85': 0.0, '90': 0.0, '95': 0.0, '99': 0.0}}
        # 'data2/016tl Anonim-20210127_214910-20211122_162657',  # metrics = {'auc_roc': {'auc_roc': 0.5627206920903955}, 'auc_pr': {'auc_pr': 0.023868846672174895}, 'brier_score': {'brier_score': 0.9507181157524012}, 'calib_score': {'calib_score': 0.9645496677457697}, 'accuracy_score': {'01': 0.016666666666666666, '05': 0.016666666666666666, '10': 0.016666666666666666, '15': 0.016666666666666666, '20': 0.016666666666666666, '25': 0.016666666666666666, '30': 0.016666666666666666, '35': 0.016666666666666666, '40': 0.016666666666666666, '45': 0.016666666666666666, '50': 0.016666666666666666, '55': 0.016666666666666666, '60': 0.016666666666666666, '65': 0.016666666666666666, '70': 0.016666666666666666, '75': 0.016666666666666666, '80': 0.016666666666666666, '85': 0.016666666666666666, '90': 0.016666666666666666, '95': 0.016666666666666666, '99': 0.9833333333333333}, 'f1_score': {'01': 0.03278688524590164, '05': 0.03278688524590164, '10': 0.03278688524590164, '15': 0.03278688524590164, '20': 0.03278688524590164, '25': 0.03278688524590164, '30': 0.03278688524590164, '35': 0.03278688524590164, '40': 0.03278688524590164, '45': 0.03278688524590164, '50': 0.03278688524590164, '55': 0.03278688524590164, '60': 0.03278688524590164, '65': 0.03278688524590164, '70': 0.03278688524590164, '75': 0.03278688524590164, '80': 0.03278688524590164, '85': 0.03278688524590164, '90': 0.03278688524590164, '95': 0.03278688524590164, '99': 0.0}, 'precision_score': {'01': 0.016666666666666666, '05': 0.016666666666666666, '10': 0.016666666666666666, '15': 0.016666666666666666, '20': 0.016666666666666666, '25': 0.016666666666666666, '30': 0.016666666666666666, '35': 0.016666666666666666, '40': 0.016666666666666666, '45': 0.016666666666666666, '50': 0.016666666666666666, '55': 0.016666666666666666, '60': 0.016666666666666666, '65': 0.016666666666666666, '70': 0.016666666666666666, '75': 0.016666666666666666, '80': 0.016666666666666666, '85': 0.016666666666666666, '90': 0.016666666666666666, '95': 0.016666666666666666, '99': 0.0}, 'recall_score': {'01': 1.0, '05': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '25': 1.0, '30': 1.0, '35': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '55': 1.0, '60': 1.0, '65': 1.0, '70': 1.0, '75': 1.0, '80': 1.0, '85': 1.0, '90': 1.0, '95': 1.0, '99': 0.0}, 'cohen_kappa_score': {'01': 0.0, '05': 0.0, '10': 0.0, '15': 0.0, '20': 0.0, '25': 0.0, '30': 0.0, '35': 0.0, '40': 0.0, '45': 0.0, '50': 0.0, '55': 0.0, '60': 0.0, '65': 0.0, '70': 0.0, '75': 0.0, '80': 0.0, '85': 0.0, '90': 0.0, '95': 0.0, '99': 0.0}}

        # # stage_1
        # # part1
        # # 'data2/038tl Anonim-20190821_113559-20211123_004935',
        # 'data2/027tl Anonim-20200309_195746-20211122_175315',
        # 'data1/dataset27',
        # 'data1/dataset14',
        # 'data2/036tl Anonim-20201224_124349-20211122_181415',
        # 'data2/041tl Anonim-20201115_222025-20211123_011114',
        # 'data1/dataset24',
        # 'data2/026tl Anonim-20210301_013744-20211122_174658',
        #
        # 'data2/020tl Anonim-20201218_071731-20211122_171454', 'data1/dataset13',
        # 'data2/018tl Anonim-20201211_130036-20211122_163611',
        # 'data2/038tl Anonim-20190822_155119-20211123_005457',
        # 'data2/025tl Anonim-20210128_233211-20211122_173425',
        # 'data2/015tl Anonim-20201116_134129-20211122_161958',
        #
        # # part2
        # 'data1/dataset3',
        # 'data2/027tl Anonim-20200310_035747-20211122_175503',
        # 'data2/002tl Anonim-20200826_124513-20211122_135804', 'data1/dataset23',
        # 'data2/022tl Anonim-20201210_132636-20211122_172649',
        # 'data1/dataset6', 'data1/dataset11',
        # 'data2/021tl Anonim-20201223_085255-20211122_172126', 'data1/dataset28',
        #
        # # part 3
        # 'data2/008tl Anonim-20210204_131328-20211122_160417',
        # 'data2/003tl Anonim-20200831_120629-20211122_140327',
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
        # 'data1/dataset1',  # 3h
        # 'data1/dataset12',  # Failed because of memory
        #
        # # # stage_2
        # # 'data2/003tl Anonim-20200831_120629-20211122_140327',
        # # 'data1/dataset12',
        # # 'data2/025tl Anonim-20210129_073208-20211122_173728',
        # # 'data2/038tl Anonim-20190822_131550-20211123_005257', 'data1/dataset2',
        # # 'data1/dataset22',
        # # 'data2/040tl Anonim-20200421_100248-20211123_010147',
        # # 'data2/020tl Anonim-20201216_073813-20211122_171341',
        # # 'data2/019tl Anonim-20201213_072025-20211122_165918',
        # # 'data2/003tl Anonim-20200831_040629-20211122_135924',
        # # 'data2/006tl Anonim-20210208_063816-20211122_154113', 'data1/dataset4', 'data1/dataset20',
        # # 'data2/035tl Anonim-20210324_231349-20211122_223059', 'data1/dataset16',
        # # 'data2/035tl Anonim-20210324_151211-20211122_222545',
        # # 'data2/038tl Anonim-20190822_203419-20211123_005705', 'data1/dataset25', 'data1/dataset5',
        # # 'data2/018tl Anonim-20201215_022951-20211122_165644',
        #
        # # stage_1 train
        # 'data2/011tl Anonim-20200118_041022-20211122_161616',
        # 'data2/016tl Anonim-20210127_214910-20211122_162657',
        # 'data2/022tl Anonim-20201209_132645-20211122_172422',
        # 'data2/002tl Anonim-20200826_044516-20211122_135439',
        # 'data2/004tl Anonim-20200929_081036-20211122_144552', 'data1/dataset29',
        # 'data2/026tl Anonim-20210227_214223-20211122_174442', 'data1/dataset19',
        'data1/dataset21',
        'data2/020tl Anonim-20201218_194126-20211122_171755',
        'data2/034tl Anonim-20210304_071124-20211122_222211', 'data1/dataset17',
        'data2/033tl Anonim-20200114_085935-20211122_180917', 'data1/dataset10',
        'data1/dataset9',
        'data2/028tl Anonim-20191014_212520-20211122_175854',
        'data2/030tl Anonim-20190910_110631-20211122_180335',
        'data2/016tl Anonim-20210128_054911-20211122_163013',
        'data2/023tl Anonim-20210110_080440-20211122_173058',
        'data1/dataset15',
        'data2/008tl Anonim-20210204_211327-20211122_160546',
        'data2/037tl Anonim-20191020_110036-20211122_223805', 'data1/dataset7',
        'data2/006tl Anonim-20210209_144403-20211122_155146',
        'data2/006tl Anonim-20210208_144401-20211122_154504',
        # 'data2/009tl Anonim-20200215_021624-20211122_161231',  # 1000 sec seizure in the end
        'data2/035tl Anonim-20210326_231343-20211122_223404',
        'data2/009tl Anonim-20200213_130213-20211122_160907',
        'data2/041tl Anonim-20201112_194437-20211123_010804', 'data1/dataset26',
        'data2/018tl Anonim-20201212_101651-20211122_163821',
        'data2/037tl Anonim-20201102_102725-20211123_003801',
        # 'data2/004tl Anonim-20200926_213911-20211122_144051',  # 1000 sec seizure in the end
        'data1/dataset8',
        'data1/dataset18',
        'data2/026tl Anonim-20210302_093747-20211122_175031',
    ]
    # subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]
    for subject_idx, subject_key in enumerate(subject_keys):
        print(subject_key)
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        predction_data = predict(model, subject_key, subject_eeg_path, subject_seizures, prediction_config)

        predictions_dir = os.path.join(experiment_dir, 'predictions', subject_key.split('/')[0])
        os.makedirs(predictions_dir, exist_ok=True)

        subject_name = subject_key.split('/')[-1]
        predction_data_path = os.path.join(predictions_dir, f'{subject_name}.pickle')
        with open(predction_data_path, 'wb') as fh:
            pickle.dump(predction_data, fh, pickle.HIGHEST_PROTOCOL)
