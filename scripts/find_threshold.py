import os
import pickle

import numpy as np

import utils.avg_meters
from utils.common import filter_predictions, calc_metrics


def get_metrics(experiment_dir, subject_keys, threshold, filter_method='median', k_size=7, sfreq=128, verbose=0):
    metric_meter = utils.avg_meters.MetricMeter()
    for subject_key in subject_keys:
        if verbose:
            print(f'\rthreshold = {threshold:3.2f} ', end='')

        try:
            prediction_path = os.path.join(experiment_dir, 'predictions', rf'{subject_key}.pickle')
            prediction_data = pickle.load(open(prediction_path, 'rb'))
        except Exception as e:
            prediction_path = os.path.join(experiment_dir, 'predictions_positive_only', rf'{subject_key}.pickle')
            prediction_data = pickle.load(open(prediction_path, 'rb'))

        # get data from prediction_data
        time_idxs_start = prediction_data['time_idxs_start']
        time_idxs_end = prediction_data['time_idxs_end']
        probs = prediction_data['probs_wo_tta'] if 'probs_wo_tta' in prediction_data else prediction_data['probs']
        labels = prediction_data['labels']

        if len(probs) == 0:
            probs = prediction_data['probs']

        probs_filtered = filter_predictions(probs, filter_method, k_size)

        # metrics
        subject_metrics = calc_metrics(probs_filtered, labels, threshold)
        subject_metrics['duration'] = (time_idxs_end[-1] / sfreq - time_idxs_start[0] / sfreq) / 3600

        metric_meter.update(subject_metrics)

        if verbose > 1:
            print(
                f'{subject_key:60}'
                f'threshold = {threshold:3.2f} '
                f'p = {subject_metrics["precision_score"]:10.4f} ({metric_meter.meters["precision_score"].avg:10.4f}) '
                f'r = {subject_metrics["recall_score"]:10.4f} ({metric_meter.meters["recall_score"].avg:10.4f}) '
                f'f1 = {subject_metrics["f1_score"]:10.4f} ({metric_meter.meters["f1_score"].avg:10.4f}) '
                f'tp = {subject_metrics["tp_num"]:6} ({metric_meter.meters["tp_num"].avg:6.2f}) '
                f'tn = {subject_metrics["tn_num"]:6} ({metric_meter.meters["tn_num"].avg:6.2f}) '
                f'fp = {subject_metrics["fp_num"]:6} ({metric_meter.meters["fp_num"].avg:6.2f}) '
                # f'fn = {subject_metrics["fn_num"]:6} ({metric_meter.meters["fn_num"].avg:6.2f}) '
                f'duration = {subject_metrics["duration"]:6.2f}'
            )

    return metric_meter


def get_best_threshold(experiment_dir, subject_keys, filter_method='median', k_size=7, sfreq=128, verbose=0):
    assert filter_method in ['mean', 'median', None]

    best_avg_f1 = -1
    best_threshold = -1
    best_metric_meter = None
    threshold_range = list(np.round(np.arange(0.1, 1, 0.05), 2))
    for threshold_idx, threshold in enumerate(threshold_range):
        metric_meter = get_metrics(experiment_dir, subject_keys, threshold, filter_method, k_size, sfreq, verbose)

        if metric_meter.meters['f1_score'].avg > best_avg_f1:
            best_avg_f1 = metric_meter.meters['f1_score'].avg
            best_threshold = threshold
            best_metric_meter = metric_meter

        if verbose:
            print(f'threshold = {threshold:3.2f} f1_score = {metric_meter.meters["f1_score"].avg:.4f} precision_score = {metric_meter.meters["precision_score"].avg:.4f} recall_score = {metric_meter.meters["recall_score"].avg:.4f} best_threshold = {best_threshold:3.2f} best_f1_score = {best_metric_meter.meters["f1_score"].avg:.4f}')

    return best_threshold, best_metric_meter


if __name__ == '__main__':
    experiment_name = '20231012_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded'
    # experiment_name = '20231005_EEGResNetCustomRaw_MixUp_TimeSeriesAug_raw_16excluded'
    # experiment_name = '20231005_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded_wo_baseline_correction'
    experiment_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name)

    # filter_method = None
    # k_size = -1
    filter_method = 'median'
    k_size = 7

    print(experiment_name)
    print(f'filter={filter_method} k={k_size}')
    print()

    subject_keys = [
        # stage_1 train
        'data2/011tl Anonim-20200118_041022-20211122_161616',
        'data2/016tl Anonim-20210127_214910-20211122_162657',
        'data2/022tl Anonim-20201209_132645-20211122_172422',
        'data2/002tl Anonim-20200826_044516-20211122_135439',
        'data2/004tl Anonim-20200929_081036-20211122_144552', 'data1/dataset29',
        'data2/026tl Anonim-20210227_214223-20211122_174442', 'data1/dataset19', 'data1/dataset21',
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
    subject_keys = [subject_key for subject_key in subject_keys if subject_key not in subject_keys_exclude]

    best_threshold, best_metric_meter = get_best_threshold(
        experiment_dir,
        subject_keys,
        filter_method,
        k_size,
    )
    print(f'\nbest_threshold = {best_threshold} best_metric_meter:\n{best_metric_meter}')
