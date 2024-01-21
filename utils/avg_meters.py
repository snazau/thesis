from __future__ import division, absolute_import
from collections import defaultdict
import torch

__all__ = ['AverageMeter', 'MetricMeter']


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    """A collection of metrics.

    Source: https://github.com/KaiyangZhou/Dassl.pytorch

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            metric_str = f'{name} = {meter.avg:10.4f}'
            if '_num' in name or 'duration' in name:
                metric_str += f' ({meter.sum:7})'

            if '_num' in name:
                metric_str = metric_str.replace('_num', '')

            if '_score' in name:
                metric_str = metric_str.replace('_score', '')

            if 'precision' in name:
                metric_str = metric_str.replace('precision', 'p')

            if 'recall' in name:
                metric_str = metric_str.replace('recall', 'r')

            if 'positives' in name:
                metric_str = metric_str.replace('positives', 'pos')

            if 'negatives' in name:
                metric_str = metric_str.replace('negatives', 'neg')

            output_str.append(metric_str)
        return self.delimiter.join(output_str)
