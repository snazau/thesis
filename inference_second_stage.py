import copy
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
from utils.neural import training


def get_positive_start_times(prediction_data, threshold, sfreq=128):
    time_idxs_start = prediction_data['time_idxs_start']
    positive_start_times_mask = prediction_data['probs_wo_tta'] > threshold
    positive_start_times = time_idxs_start[positive_start_times_mask] / sfreq
    return positive_start_times, positive_start_times_mask


def predict(model, rough_model_prediction_data, subject_eeg_path, subject_seizures, config):
    # TODO: add ability to work with models that use baseline_correction

    model.eval()

    positive_start_times, positive_start_times_mask = get_positive_start_times(
        rough_model_prediction_data,
        threshold=config['rough_model_threshold'],
        sfreq=config['sfreq']
    )
    subject_dataset = datasets.datasets_static.SubjectInferenceDataset(
        subject_eeg_path,
        positive_start_times,
        sample_duration=config['sample_duration'],
        data_type='raw'
    )
    # collate_fn = partial(
    #     datasets.datasets_static.custom_collate_function,
    #     normalization=config['normalization'],
    #     transform=None,
    #     data_type=config['data_type'],
    # )
    collate_fn = partial(
        datasets.datasets_static.tta_collate_function,
        tta_augs=config['tta_augs'],
        normalization=config['normalization'],
        data_type=config['data_type'],
    )
    loader = torch.utils.data.DataLoader(subject_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=False)

    device = config['device']

    data_keys = ['data'] + [f'data_aug{aug_idx:03}' for aug_idx in range(len(config['tta_augs']))]

    all_probs_wo_tta = np.array([])
    all_probs = np.array([])
    all_time_idxs_start = np.array([])
    all_time_idxs_end = np.array([])
    with tqdm.tqdm(loader) as tqdm_wrapper:
        for batch_idx, batch in enumerate(tqdm_wrapper):
            time_idxs_start = batch['time_idx_start']
            time_idxs_end = batch['time_idx_end']
            probs_tta = torch.zeros_like(time_idxs_start, dtype=torch.float32).to(device)

            for data_key in data_keys:
                inputs = batch[data_key].to(device)
                with torch.no_grad():
                    outputs = torch.squeeze(model(inputs))
                    probs = torch.sigmoid(outputs)
                    probs_tta += probs
                    if data_key == 'data':
                        probs_wo_tta = probs.clone()
            probs_tta = probs_tta / len(data_keys)

            all_probs = np.concatenate([all_probs, probs_tta.cpu().detach().numpy()])
            all_probs_wo_tta = np.concatenate([all_probs_wo_tta, probs_wo_tta.cpu().detach().numpy()])
            all_time_idxs_start = np.concatenate([all_time_idxs_start, time_idxs_start.cpu().detach().numpy()])
            all_time_idxs_end = np.concatenate([all_time_idxs_end, time_idxs_end.cpu().detach().numpy()])

            # if batch_idx > 2:
            #     break

            tqdm_wrapper.set_postfix()

    # construct refined probs
    rough_probs = rough_model_prediction_data['probs']
    rough_positive_probs = rough_probs[positive_start_times_mask]
    refined_positive_probs = np.zeros_like(rough_positive_probs)
    refined_positive_probs[all_probs > 0.5] = np.maximum(refined_positive_probs[all_probs > 0.5], rough_positive_probs[all_probs > 0.5])
    refined_positive_probs[all_probs <= 0.5] = np.minimum(refined_positive_probs[all_probs <= 0.5], rough_positive_probs[all_probs <= 0.5])

    refined_probs = copy.deepcopy(rough_probs)
    refined_probs[positive_start_times_mask] = refined_positive_probs

    # construct refined probs_wo_tta
    rough_probs_wo_tta = rough_model_prediction_data['probs_wo_tta']
    rough_positive_probs_wo_tta = rough_probs_wo_tta[positive_start_times_mask]
    refined_positive_probs_wo_tta = np.zeros_like(rough_positive_probs_wo_tta)
    refined_positive_probs_wo_tta[all_probs_wo_tta > 0.5] = np.maximum(refined_positive_probs_wo_tta[all_probs_wo_tta > 0.5], rough_positive_probs_wo_tta[all_probs_wo_tta > 0.5])
    refined_positive_probs_wo_tta[all_probs_wo_tta <= 0.5] = np.minimum(refined_positive_probs_wo_tta[all_probs_wo_tta <= 0.5], rough_positive_probs_wo_tta[all_probs_wo_tta <= 0.5])

    refined_probs_wo_tta = copy.deepcopy(rough_probs_wo_tta)
    refined_probs_wo_tta[positive_start_times_mask] = refined_positive_probs_wo_tta

    metrics = utils.neural.training.calc_metrics_diff_thresholds(refined_probs, rough_model_prediction_data['labels'])
    metrics_wo_tta = utils.neural.training.calc_metrics_diff_thresholds(refined_probs_wo_tta, rough_model_prediction_data['labels'])
    prediction_data = {
        'subject_eeg_path': subject_eeg_path,
        'subject_seizures': subject_seizures,
        'probs': refined_probs,
        'probs_wo_tta': refined_probs_wo_tta,
        'labels': rough_model_prediction_data['labels'],
        'time_idxs_start': rough_model_prediction_data['time_idxs_start'],
        'time_idxs_end': rough_model_prediction_data['time_idxs_end'],
        'min_time_in_seconds': rough_model_prediction_data['min_time_in_seconds'],
        'max_time_in_seconds': rough_model_prediction_data['max_time_in_seconds'],
        'metrics': metrics,
        'metrics_wo_tta': metrics_wo_tta,
    }
    return prediction_data


if __name__ == '__main__':
    import json
    import os

    dataset_info_path = './data/dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    experiments_dir = r'D:\Study\asp\thesis\implementation\experiments'
    experiment_name = 'renset18_2nd_stage_MixUp_SpecTimeFlipEEGFlipAug'  # stage_2
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best.pth.tar')

    ttg_augs = list()
    prediction_config = {
        'sfreq': 128,
        'sample_duration': 10,
        'shift': 10,
        'data_type': 'power_spectrum',
        'normalization': 'meanstd',
        'batch_size': 16,
        'device': torch.device('cuda:0'),
        'tta_augs': ttg_augs,
        'rough_model_threshold': 0.95,
    }

    state_dict = torch.load(checkpoint_path)['model']['state_dict']
    model = utils.neural.training.get_model('resnet18', model_kwargs=dict())
    model.load_state_dict(state_dict)
    model = model.to(prediction_config['device'])

    data_dir = './data'
    rough_model_prediction_dir = r'D:\Study\asp\thesis\implementation\experiments\renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug\predictions'
    # subject_key = 'data1/dataset28'
    # subject_key = 'data1/dataset2'
    # subject_keys = ['data2/038tl Anonim-20190821_113559-20211123_004935', 'data1/dataset27']
    # subject_keys = ['data2/038tl Anonim-20190821_113559-20211123_004935']
    subject_keys = [
        # stage_2
        'data2/003tl Anonim-20200831_120629-20211122_140327',
        'data1/dataset12',
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
    ]
    for subject_idx, subject_key in enumerate(subject_keys):
        print(subject_key)
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        rough_model_prediction_path = os.path.join(rough_model_prediction_dir, subject_key + '.pickle')
        rough_model_prediction_data = pickle.load(open(rough_model_prediction_path, 'rb'))

        predction_data = predict(model, rough_model_prediction_data, subject_eeg_path, subject_seizures, prediction_config)
        predictions_dir = os.path.join(experiment_dir, 'predictions_positive_only', subject_key.split('/')[0])
        os.makedirs(predictions_dir, exist_ok=True)
        subject_name = subject_key.split('/')[-1]
        predction_data_path = os.path.join(predictions_dir, f'{subject_name}.pickle')
        with open(predction_data_path, 'wb') as fh:
            pickle.dump(predction_data, fh, pickle.HIGHEST_PROTOCOL)
