import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

import utils.avg_meters
import utils.neural.training


def train(strategy_name, strategy_params, loader, model, criterion, optimizer, epoch, writer, device):
    model.train()

    all_labels = np.array([])
    all_probs = np.array([])

    loss_avg_meter = utils.avg_meters.AverageMeter()
    for batch_idx, sample in enumerate(loader):
        inputs = sample["data"].to(device)
        labels = sample["target"].to(device)

        optimizer.zero_grad()
        outputs, loss = utils.neural.training.forward(strategy_name, strategy_params, model, inputs, labels, criterion)
        loss.backward()
        optimizer.step()

        loss_avg_meter.update(loss.item())

        probs = torch.sigmoid(outputs)
        all_labels = np.concatenate([all_labels, labels.cpu().detach().numpy()])
        all_probs = np.concatenate([all_probs, probs[:, 0].cpu().detach().numpy()])

        print(f'\rProgress {batch_idx + 1}/{len(loader)} loss = {loss_avg_meter.avg:.5f}', end='')
    print()

    writer.add_scalar('loss/lr', optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar('loss/train', loss_avg_meter.avg, epoch)
    metrics = utils.neural.training.calc_metrics(all_probs, all_labels)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'{metric_name}/train', metric_value, epoch)

    return loss_avg_meter.avg, metrics


def validate(loader, model, criterion, optimizer, epoch, writer, device):
    model.eval()

    all_labels = np.array([])
    all_probs = np.array([])

    loss_avg_meter = utils.avg_meters.AverageMeter()
    for batch_idx, sample in enumerate(loader):
        inputs = sample["data"].to(device)
        labels = sample["target"].to(device)

        with torch.no_grad():
            outputs, loss = utils.neural.training.forward('default', {}, model, inputs, labels, criterion)
            probs = torch.sigmoid(outputs)

        loss_avg_meter.update(loss.item())

        all_labels = np.concatenate([all_labels, labels.cpu().detach().numpy()])
        all_probs = np.concatenate([all_probs, probs[:, 0].cpu().detach().numpy()])

        print(f'\rProgress {batch_idx + 1}/{len(loader)} loss = {loss_avg_meter.avg:.5f}', end='')
    print()

    writer.add_scalar('loss/val', loss_avg_meter.avg, epoch)
    metrics = utils.neural.training.calc_metrics(all_probs, all_labels)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'{metric_name}/val', metric_value, epoch)

    return loss_avg_meter.avg, metrics


def run_training(config):
    run_dir = config['run_dir']
    device = torch.device(config['device'])
    print(f'run_dir = {run_dir} device = {device}')

    # Seeds
    if config['deterministic'] is True:
        utils.neural.training.set_seed(config['seed'], config['deterministic'])

    # Init worker
    writer = SummaryWriter(os.path.join(run_dir, 'tb_logs'))

    # Data
    loader_train = utils.neural.training.get_loader(
        config['data']['data_dir'],
        config['data']['dataset_info_path'],
        config['data']['train']['subject_keys'],
        mode='train',
        dataset_kwargs=config['data']['train']['dataset_params'],
        loader_kwargs=config['data']['train']['loader_params'],
    )

    loader_val = utils.neural.training.get_loader(
        config['data']['data_dir'],
        config['data']['dataset_info_path'],
        config['data']['val']['subject_keys'],
        mode='val',
        dataset_kwargs=config['data']['val']['dataset_params'],
        loader_kwargs=config['data']['val']['loader_params'],
    )

    print(f'len(loader_train) = {len(loader_train)}')
    print(f'len(loader_val) = {len(loader_val)}')
    print()

    # Model
    model = utils.neural.training.get_model(config['model']['name'], config['model']['params'])
    model = model.to(device)

    # Criterion
    criterion = utils.neural.training.get_criterion(config['criterion']['name'], config['criterion']['params'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    # Scheduler
    scheduler = utils.neural.training.get_scheduler(config['scheduler']['name'], config['scheduler']['params'], optimizer)

    # Train loop
    min_val_loss = 1e10
    for epoch in range(config['epochs']):
        print(f'training started e={epoch:03}')
        loss_avg_train, metrics_train = train(
            config['strategy']['name'],
            config['strategy']['params'],
            loader_train,
            model,
            criterion,
            optimizer,
            epoch,
            writer,
            device,
        )
        print(f'loss_avg_train = {loss_avg_train} metrics_train = {metrics_train}')

        print(f'Validation started e={epoch:03}')
        loss_avg_val, metrics_val = validate(loader_val, model, criterion, optimizer, epoch, writer, device)
        print(f'loss_avg_val = {loss_avg_val} metrics_val = {metrics_val}')

        # Save best checkpoints
        checkpoint_losses = {'train': loss_avg_train, 'val': loss_avg_val}
        checkpoint_metrics = {'train': metrics_train, 'val': metrics_val}

        checkpoints_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoints_dir, f'e={epoch:03}.pth.tar')
        utils.neural.training.save_checkpoint(
            checkpoint_path,
            epoch,
            config,
            model,
            checkpoint_losses,
            checkpoint_metrics,
        )

        checkpoint_path = os.path.join(checkpoints_dir, f'last.pth.tar')
        utils.neural.training.save_checkpoint(
            checkpoint_path,
            epoch,
            config,
            model,
            checkpoint_losses,
            checkpoint_metrics,
        )

        if loss_avg_val < min_val_loss:
            checkpoint_path = os.path.join(checkpoints_dir, 'best.pth.tar')
            utils.neural.training.save_checkpoint(
                checkpoint_path,
                epoch,
                config,
                model,
                checkpoint_losses,
                checkpoint_metrics,
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(min_val_loss)
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
        print()


if __name__ == "__main__":
    config = {
        'run_dir': 'D:\\Study\\asp\\thesis\\implementation\\experiments\\test_run\\',
        'device': 'cuda:0',
        'deterministic': True,
        'seed': 8,
        'epochs': 1,
        'lr': 1e-3,
        'strategy': {
            'name': 'default',
            'params': {},
        },
        'data': {
            'data_dir': 'D:\\Study\\asp\\thesis\\implementation\\data\\',
            'dataset_info_path': 'D:\\Study\\asp\\thesis\\implementation\\data\\dataset_info.json',
            'train': {
                'subject_keys': [
                    'data2/038tl Anonim-20190821_113559-20211123_004935',
                    # 'data1/dataset2',
                ],
                'dataset_params': {
                    'samples_num': 100,
                    'sample_duration': 10,
                },
                'loader_params': {
                    'batch_size': 4,
                    'shuffle': True,
                },
            },
            'val': {
                'subject_keys': [
                    'data2/038tl Anonim-20190821_113559-20211123_004935',
                    # 'data1/dataset28',
                ],
                'dataset_params': {
                    'sample_duration': 10,
                },
                'loader_params': {
                    'batch_size': 4,
                    'shuffle': False,
                },
            },
        },
        'model': {
            'name': 'resnet18',
            'params': {
                'pretrained': True,
            },
        },
        'criterion': {
            'name': 'BCE',
            'params': {},
        },
        'scheduler': {
            'name': None,
            'params': {},
        },
    }

    run_training(config)
