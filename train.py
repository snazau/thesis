import gc
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

import utils.avg_meters
import utils.neural.training


def train(strategy_name, strategy_params, loader, model, criterion, optimizer, epoch, writer, device):
    model.train()

    all_labels = np.array([])
    all_probs = np.array([])

    loss_avg_meter = utils.avg_meters.AverageMeter()
    with tqdm.tqdm(loader) as tqdm_wrapper:
        for batch_idx, batch in enumerate(tqdm_wrapper):
            tqdm_wrapper.set_description(f'Epoch {epoch:03}')

            batch["data"] = batch["data"].to(device)
            batch["target"] = batch["target"].to(device)

            optimizer.zero_grad()
            outputs, loss = utils.neural.training.forward(strategy_name, strategy_params, model, batch, criterion)
            loss.backward()
            optimizer.step()

            loss_avg_meter.update(loss.item())

            probs = torch.sigmoid(outputs)
            if len(probs.shape) > 2:  # select last sequence prediction for recurrent scheme
                probs = probs[:, -1]

            all_labels = np.concatenate([all_labels, batch["target"].cpu().detach().numpy()])
            all_probs = np.concatenate([all_probs, probs[:, 0].cpu().detach().numpy()])

            tqdm_wrapper.set_postfix(loss=loss_avg_meter.avg)

    writer.add_scalar('lr/train', optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar('loss/train', loss_avg_meter.avg, epoch)
    metrics = utils.neural.training.calc_metrics(all_probs, all_labels)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'{metric_name}/train', metric_value, epoch)
    # for metric_name, metric_dict in metrics.items():
    #     for threshold_str, metric_value in metric_dict.items():
    #         writer.add_scalar(f'{metric_name}/{threshold_str}/train', metric_value, epoch)

    return loss_avg_meter.avg, metrics


def validate(loader, model, criterion, optimizer, epoch, writer, device):
    model.eval()

    all_labels = np.array([])
    all_probs = np.array([])
    # all_start_times = np.array([])

    loss_avg_meter = utils.avg_meters.AverageMeter()
    with tqdm.tqdm(loader) as tqdm_wrapper:
        for batch_idx, batch in enumerate(tqdm_wrapper):
            batch["data"] = batch["data"].to(device)
            batch["target"] = batch["target"].to(device)
            batch["start_time"] = batch["start_time"].to(device)

            with torch.no_grad():
                outputs, loss = utils.neural.training.forward('default', {}, model, batch, criterion)
                probs = torch.sigmoid(outputs)
                if len(probs.shape) > 2:  # select last sequence prediction for recurrent scheme
                    probs = probs[:, -1]

            loss_avg_meter.update(loss.item())

            all_labels = np.concatenate([all_labels, batch["target"].cpu().detach().numpy()])
            all_probs = np.concatenate([all_probs, probs[:, 0].cpu().detach().numpy()])
            # all_start_times = np.concatenate([all_start_times, batch["start_time"].cpu().detach().numpy()])

            tqdm_wrapper.set_postfix(loss=loss_avg_meter.avg)

    writer.add_scalar('loss/val', loss_avg_meter.avg, epoch)
    metrics = utils.neural.training.calc_metrics(all_probs, all_labels)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'{metric_name}/val', metric_value, epoch)
    # for metric_name, metric_dict in metrics.items():
    #     for threshold_str, metric_value in metric_dict.items():
    #         writer.add_scalar(f'{metric_name}/{threshold_str}/val', metric_value, epoch)

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

    # Process subject keys to exclude
    if 'subject_keys_exclude' not in config['data']:
        config['data']['subject_keys_exclude'] = list()

    config['data']['train']['subject_keys'] = [
        subject_key
        for subject_key in config['data']['train']['subject_keys']
        if subject_key not in config['data']['subject_keys_exclude']
    ]

    config['data']['val']['subject_keys'] = [
        subject_key
        for subject_key in config['data']['val']['subject_keys']
        if subject_key not in config['data']['subject_keys_exclude']
    ]

    # Data val
    datasets_train = utils.neural.training.get_datasets(
        config['data']['data_dir'],
        config['data']['dataset_info_path'],
        config['data']['train']['subject_keys'],
        config['data']['train']['prediction_data_dir'],
        config['data']['train']['stats_dir'],
        dataset_class_name=config['data']['train']['dataset_class_name'],
        dataset_kwargs=config['data']['train']['dataset_params'],
    )

    datasets_val = utils.neural.training.get_datasets(
        config['data']['data_dir'],
        config['data']['dataset_info_path'],
        config['data']['val']['subject_keys'],
        config['data']['val']['prediction_data_dir'],
        config['data']['train']['stats_dir'],
        dataset_class_name=config['data']['val']['dataset_class_name'],
        dataset_kwargs=config['data']['val']['dataset_params'],
    )
    loader_val = utils.neural.training.get_loader(
        datasets_val,
        loader_kwargs=config['data']['val']['loader_params'],
    )

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

    # Pretrained weights
    pretrained_epochs_num = 0
    if 'pretrained_path' in config:
        checkpoint = utils.neural.training.load_checkpoint(config['pretrained_path'])
        state_dict = checkpoint['model']['state_dict']
        state_dict = {f'model.{key}': value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])

        if 'continue_epochs' in config and config['continue_epochs']:
            pretrained_epochs_num = checkpoint['epoch'] + 1
        print('Successfully loaded pretrained weights')
        print(f'Pretrained metrics:\n{checkpoint["metrics"]}')

    # Train loop
    min_val_loss = 1e10
    min_val_loss_epoch = -1
    epochs = config['epochs']
    for epoch in range(pretrained_epochs_num, pretrained_epochs_num + epochs):
        if len(config['data']['train']['subject_keys']) > 0:
            print('Renewing training raw data')
            for dataset_idx in range(len(datasets_train)):
                if hasattr(datasets_train, 'renew_data'):
                    datasets_train[dataset_idx].renew_data()
            loader_train = utils.neural.training.get_loader(
                datasets_train,
                loader_kwargs=config['data']['train']['loader_params'],
            )

            print(f'training started e={epoch:03}/{epochs:03}')
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
            gc.collect()
        else:
            loss_avg_train = -1
            metrics_train = dict()

        print(f'Validation started e={epoch:03}/{epochs:03}')
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
            optimizer,
            checkpoint_losses,
            checkpoint_metrics,
        )

        checkpoint_path = os.path.join(checkpoints_dir, 'last.pth.tar')
        utils.neural.training.save_checkpoint(
            checkpoint_path,
            epoch,
            config,
            model,
            optimizer,
            checkpoint_losses,
            checkpoint_metrics,
        )

        if loss_avg_val < min_val_loss:
            min_val_loss = loss_avg_val
            min_val_loss_epoch = epoch
            checkpoint_path = os.path.join(checkpoints_dir, 'best.pth.tar')
            utils.neural.training.save_checkpoint(
                checkpoint_path,
                epoch,
                config,
                model,
                optimizer,
                checkpoint_losses,
                checkpoint_metrics,
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(min_val_loss)
            else:
                scheduler.step()
        print(f'epoch = {epoch} min_val_loss = {min_val_loss} min_val_loss_epoch = {min_val_loss_epoch}')
        print()
