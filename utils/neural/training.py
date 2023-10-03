import json
import os
import random

import numpy as np
import sklearn.metrics
import torch
import torch.utils.data

import criterions
import datasets.datasets_static
import metrics.calib
import models.resnet
import models.efficientnet
import utils.neural.mixing


def get_criterion(criterion_name, criterion_kwargs):
    if criterion_name == 'BCE':
        criterion = criterions.BCELoss(**criterion_kwargs)
    elif criterion_name == 'BCELossWithTimeToClosestSeizure':
        criterion = criterions.BCELossWithTimeToClosestSeizure(**criterion_kwargs)
    elif criterion_name == 'FocalBCE':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return criterion


def get_model(model_name, model_kwargs):
    if model_name == 'resnet18':
        model = models.resnet.EEGResNet18Spectrum(**model_kwargs)
    elif model_name == 'resnet18_1channel':
        model = models.resnet.EEGResNet18Raw(**model_kwargs)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet.EEGEfficientNetB0Spectrum(**model_kwargs)
    elif model_name == 'efficientnet_b0_1channel':
        model = models.efficientnet.EEGEfficientNetB0Raw(**model_kwargs)
    else:
        raise NotImplementedError

    return model


def get_scheduler(scheduler_name, scheduler_kwargs, optimizer):
    scheduler = None
    if scheduler_name is not None:
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        elif scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
        else:
            raise NotImplementedError

    return scheduler


def get_datasets(data_dir, dataset_info_path, subject_keys, prediction_data_dir, stats_dir, dataset_class_name, dataset_kwargs):
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    datasets_list = list()
    for subject_key in subject_keys:
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        prediction_data_path = None if prediction_data_dir is None else os.path.join(prediction_data_dir, subject_key + '.pickle')
        stats_path = None if stats_dir is None else os.path.join(stats_dir, subject_key + '.npy')
        # if not os.path.exists(prediction_data_path):
        #     prediction_data_path = None

        if dataset_class_name == 'SubjectPreprocessedDataset':
            subject_dataset = datasets.datasets_static.SubjectPreprocessedDataset(
                preprocessed_dir=os.path.join(data_dir, f'{subject_key.split("/")[1]}'),
                seizures=subject_seizures,
                **dataset_kwargs,
            )
        else:
            dataset_kwargs['prediction_data_path'] = prediction_data_path
            dataset_kwargs['stats_path'] = stats_path
            subject_dataset = getattr(datasets, dataset_class_name)(subject_eeg_path, subject_seizures, **dataset_kwargs)

        datasets_list.append(subject_dataset)
    return datasets_list


def get_loader(datasets_list, loader_kwargs):
    dataset_merged = torch.utils.data.ConcatDataset(datasets_list)
    loader = torch.utils.data.DataLoader(dataset_merged, **loader_kwargs)
    return loader


def forward(strategy_name, strategy_kwargs, model, batch, criterion):
    if strategy_name.lower() == 'mixup':
        mixed_inputs, labels, labels_shuffled, lam = utils.neural.mixing.mixup(batch['data'], batch['target'], **strategy_kwargs)
        batch['outputs'] = model(mixed_inputs)

        loss = lam * criterion(batch)

        batch_copy = {
            key: value.clone() if hasattr(value, 'clone') else value
            for key, value in batch.items()
        }
        batch_copy['labels'] = labels_shuffled
        loss += (1 - lam) * criterion(batch_copy)
    elif strategy_name.lower() == 'fmix':
        raise NotImplementedError
    elif strategy_name.lower() == 'default':
        batch['outputs'] = model(batch['data'])
        loss = criterion(batch)
    else:
        raise NotImplementedError

    return batch['outputs'], loss


def calc_metric(function, labels, preds):
    try:
        return function(labels, preds)
    except Exception as e:
        return -1


def calc_metrics_diff_thresholds(probs, labels):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
    auc_roc = sklearn.metrics.auc(fpr, tpr)

    precision_pr, recall_pr, _ = sklearn.metrics.precision_recall_curve(labels, probs)
    auc_pr = sklearn.metrics.auc(recall_pr, precision_pr)

    brier_score = sklearn.metrics.brier_score_loss(labels, probs)
    calib_mean, calib_std = metrics.calib.fast_calibration_report(labels, probs, nbins=100, show_plots=False)

    metric_dict = {
        'auc_roc': {'auc_roc': auc_roc},
        'auc_pr': {'auc_pr': auc_pr},
        'brier_score': {'brier_score': brier_score},
        'calib_score': {'calib_score': calib_mean},
    }
    metric_names = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'cohen_kappa_score']
    thresholds = [0.01] + list(np.arange(0.05, 0.99, 0.05)) + [0.99]
    for threshold in thresholds:
        preds = probs > threshold
        for metric_name in metric_names:
            # metric_value = getattr(sklearn.metrics, metric_name)(labels, preds)
            metric_value = calc_metric(getattr(sklearn.metrics, metric_name), labels, preds)

            if metric_name not in metric_dict:
                metric_dict[metric_name] = dict()
            # metric_dict[metric_name][f'{metric_name}_{threshold * 100:03}'] = metric_value
            metric_dict[metric_name][f'{int(threshold * 100):02}'] = metric_value

    return metric_dict


def calc_metrics(probs, labels, threshold=0.5):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
    auc_roc = sklearn.metrics.auc(fpr, tpr)

    precision_pr, recall_pr, _ = sklearn.metrics.precision_recall_curve(labels, probs)
    auc_pr = sklearn.metrics.auc(recall_pr, precision_pr)

    brier_score = sklearn.metrics.brier_score_loss(labels, probs)
    calib_mean, calib_std = metrics.calib.fast_calibration_report(labels, probs, nbins=100, show_plots=False)

    preds = probs > threshold
    accuracy_combined = calc_metric(sklearn.metrics.accuracy_score, labels, preds)
    f1_score = calc_metric(sklearn.metrics.f1_score, labels, preds)
    precision = calc_metric(sklearn.metrics.precision_score, labels, preds)
    recall = calc_metric(sklearn.metrics.recall_score, labels, preds)
    cohen_kappa = calc_metric(sklearn.metrics.cohen_kappa_score, labels, preds)

    accuracy_class = np.sum(labels[labels == 1] * preds[labels == 1]) / np.sum(labels)

    metric_dict = {
        'brier_score': brier_score,
        'calib_score': calib_mean,
        'accuracy_combined': accuracy_combined,
        'accuracy_class': accuracy_class,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'cohen_kappa': cohen_kappa,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
    }

    return metric_dict


def save_checkpoint(save_path, epoch, config, model, optimizer, losses, metrics):
    checkpoint = {
        'epoch': epoch,
        'losses': losses,
        'metrics': metrics,
        'optimizer': {
            'name': optimizer.__class__.__name__,
            'state_dict': optimizer.state_dict(),
        },
        'model': {
            'name': model.__class__.__name__,
            'state_dict': model.state_dict()
        },
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def set_seed(seed=8, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
