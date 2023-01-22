import json
import os
import random

import numpy as np
import sklearn.metrics
import torch
import torch.utils.data
import torchvision.models

import datasets
import utils.neural.mixing


def get_criterion(criterion_name, criterion_kwargs):
    if criterion_name == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss(**criterion_kwargs)
    elif criterion_name == 'FocalBCE':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return criterion


def get_model(model_name, model_kwargs):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(**model_kwargs)

        conv1_pretrained_weight = model.conv1.weight
        model.conv1 = torch.nn.Conv2d(25, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1_weight = torch.cat([torch.mean(conv1_pretrained_weight, dim=1, keepdim=True) for _ in range(25)], dim=1)
        model.conv1.weight = torch.nn.parameter.Parameter(conv1_weight, requires_grad=True)

        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
    elif model_name == 'resnet18_1channel':
        model = torchvision.models.resnet18(**model_kwargs)

        conv1_pretrained_weight = model.conv1.weight
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1_weight = torch.mean(conv1_pretrained_weight, dim=1, keepdim=True)
        model.conv1.weight = torch.nn.parameter.Parameter(conv1_weight, requires_grad=True)

        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
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
        else:
            raise NotImplementedError

    return scheduler


def get_datasets(data_dir, dataset_info_path, subject_keys, mode, dataset_kwargs):
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    datasets_list = list()
    for subject_key in subject_keys:
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        if mode == 'train':
            subject_dataset = datasets.SubjectRandomDataset(subject_eeg_path, subject_seizures, **dataset_kwargs)
        elif mode == 'val':
            subject_dataset = datasets.SubjectSequentialDataset(subject_eeg_path, subject_seizures, **dataset_kwargs)
        else:
            raise NotImplementedError

        datasets_list.append(subject_dataset)
    return datasets_list


def get_loader(datasets_list, loader_kwargs):
    dataset_merged = torch.utils.data.ConcatDataset(datasets_list)
    loader = torch.utils.data.DataLoader(dataset_merged, **loader_kwargs)
    return loader


def forward(strategy_name, strategy_kwargs, model, inputs, labels, criterion):
    if strategy_name == 'mixup':
        mixed_inputs, labels, labels_shuffled, lam = utils.neural.mixing.mixup(inputs, labels, **strategy_kwargs)
        outputs = model(mixed_inputs)
        loss = lam * criterion(outputs, labels.unsqueeze(1)) + (1 - lam) * criterion(
            outputs,
            labels_shuffled.unsqueeze(1)
        )
    elif strategy_name == 'fmix':
        raise NotImplementedError
    elif strategy_name == 'default':
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
    else:
        raise NotImplementedError

    return outputs, loss


def calc_metric(function, labels, preds):
    try:
        return function(labels, preds)
    except Exception as e:
        return -1


def calc_metrics(probs, labels):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
    auc_roc = sklearn.metrics.auc(fpr, tpr)

    precision_pr, recall_pr, _ = sklearn.metrics.precision_recall_curve(labels, probs)
    auc_pr = sklearn.metrics.auc(recall_pr, precision_pr)

    # metric_dict = {'auc_roc': {'auc_roc': auc_roc}, 'auc_pr': {'auc_pr': auc_pr}}
    # metric_names = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'cohen_kappa_score']
    # thresholds = [0.01] + list(np.arange(0.05, 0.99, 0.05)) + [0.99]
    # for threshold in thresholds:
    #     preds = probs > threshold
    #     for metric_name in metric_names:
    #         # metric_value = getattr(sklearn.metrics, metric_name)(labels, preds)
    #         metric_value = calc_metric(getattr(sklearn.metrics, metric_name), labels, preds)
    #
    #         if metric_name not in metric_dict:
    #             metric_dict[metric_name] = dict()
    #         # metric_dict[metric_name][f'{metric_name}_{threshold * 100:03}'] = metric_value
    #         metric_dict[metric_name][f'{int(threshold * 100):02}'] = metric_value

    preds = probs > 0.5
    accuracy_combined = calc_metric(sklearn.metrics.accuracy_score, labels, preds)
    f1_score = calc_metric(sklearn.metrics.f1_score, labels, preds)
    precision = calc_metric(sklearn.metrics.precision_score, labels, preds)
    recall = calc_metric(sklearn.metrics.recall_score, labels, preds)
    cohen_kappa = calc_metric(sklearn.metrics.cohen_kappa_score, labels, preds)

    accuracy_class = np.sum(labels[labels == 1] * preds[labels == 1]) / np.sum(labels)

    metric_dict = {
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


def save_checkpoint(save_path, epoch, config, model, losses, metrics):
    checkpoint = {
        'epoch': epoch,
        'losses': losses,
        'metrics': metrics,
        'model': {
            'name': model.__class__.__name__,
            'state_dict': model.state_dict()
        },
    }
    torch.save(checkpoint, save_path)


def set_seed(seed=8, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
