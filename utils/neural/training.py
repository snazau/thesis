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


def get_loader(data_dir, dataset_info_path, subject_keys, mode, dataset_kwargs, loader_kwargs):
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    datasets_to_merge = list()
    for subject_key in subject_keys:
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        if mode == 'train':
            subject_dataset = datasets.SubjectRandomDataset(subject_eeg_path, subject_seizures, **dataset_kwargs)
        elif mode == 'val':
            subject_dataset = datasets.SubjectSequentialDataset(subject_eeg_path, subject_seizures, **dataset_kwargs)
        else:
            raise NotImplementedError

        datasets_to_merge.append(subject_dataset)

    dataset_merged = torch.utils.data.ConcatDataset(datasets_to_merge)
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


def calc_metrics(probs, labels):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
    auc_roc = sklearn.metrics.auc(fpr, tpr)

    precision_pr, recall_pr, _ = sklearn.metrics.precision_recall_curve(labels, probs)
    auc_pr = sklearn.metrics.auc(recall_pr, precision_pr)

    preds = probs > 0.5
    accuracy_combined = sklearn.metrics.accuracy_score(labels, preds)
    f1_score = sklearn.metrics.f1_score(labels, preds)
    precision = sklearn.metrics.precision_score(labels, preds)
    recall = sklearn.metrics.recall_score(labels, preds)
    cohen_kappa = sklearn.metrics.cohen_kappa_score(labels, preds)

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
    save_dir = os.path.basename(save_path)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'run_description': config.run_description,
        'epoch': epoch,
        'date': config.curr_date,
        'description': config.run_description,
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
