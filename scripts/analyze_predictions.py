import os
import pickle

import plotly.graph_objects as go
import scipy.ndimage
import sklearn.metrics


def filter_predictions(probs, filter, k_size):
    assert filter in ['mean', 'median', None]

    if filter == 'mean':
        probs_filtered = scipy.ndimage.uniform_filter1d(probs, size=k_size)
    elif filter == 'median':
        probs_filtered = scipy.ndimage.median_filter(probs, size=k_size)
    elif filter is None:
        probs_filtered = probs.copy()
    else:
        raise NotImplementedError

    return probs_filtered


def plot_hist(values, values_name):
    import matplotlib.pyplot as plt
    plt.hist(values, density=False, bins=100)
    plt.ylabel('counts')
    plt.xlabel(values_name)
    plt.title(values_name)
    plt.show()


def process_subject_predictions(
    prediction_data,
    threshold,
    filter=None,
    k_size=-1,
    sfreq=128,
    visualization_path=None,
):
    assert filter in ['mean', 'median', None]

    # get data from prediction_data
    time_idxs_start = prediction_data['time_idxs_start']
    time_idxs_end = prediction_data['time_idxs_end']
    seizures = prediction_data['subject_seizures']
    probs = prediction_data['probs_wo_tta']
    labels = prediction_data['labels']

    probs_filtered = filter_predictions(probs, filter, k_size)

    # metrics
    preds = probs_filtered > threshold
    f1_score = sklearn.metrics.f1_score(labels, preds)
    precision_score = sklearn.metrics.precision_score(labels, preds)
    recall_score = sklearn.metrics.recall_score(labels, preds)

    fp_mask = ((labels == 0) & (probs_filtered > threshold))
    fp_num = fp_mask.sum()

    # fp_series_sizes = list()
    # fp_idx = 0
    # while fp_idx < len(fp_mask):
    #     fp = fp_mask[fp_idx]
    #     if fp:
    #         fp_series_size = 1
    #         fp_idx += 1
    #         while fp_mask[fp_idx]:
    #             fp_idx += 1
    #             fp_series_size += 1
    #         fp_series_sizes.append(fp_series_size)
    #     else:
    #         fp_idx += 1
    # plot_hist(fp_series_sizes, 'fp_series_sizes')
    #
    # fp_min_dists_to_seizure = list()
    # for fp_idx, fp in enumerate(fp_mask):
    #     if not fp:
    #         continue
    #
    #     fp_time_start = time_idxs_start[fp_idx] / sfreq
    #     min_dist_to_seizure = 1e9
    #     for seizure in seizures:
    #         min_dist_to_seizure = min(min_dist_to_seizure, abs(fp_time_start - seizure['start']))
    #         min_dist_to_seizure = min(min_dist_to_seizure, abs(fp_time_start - seizure['end']))
    #     fp_min_dists_to_seizure.append(min_dist_to_seizure)
    #
    # plot_hist(fp_min_dists_to_seizure, 'fp_min_dists_to_seizure')

    fn_mask = ((labels == 1) & (probs_filtered <= threshold))
    fn_num = fn_mask.sum()

    # fn_series_sizes = list()
    # fn_idx = 0
    # while fn_idx < len(fn_mask):
    #     fn = fn_mask[fn_idx]
    #     if fn:
    #         fn_series_size = 1
    #         fn_idx += 1
    #         while fn_mask[fn_idx]:
    #             fn_idx += 1
    #             fn_series_size += 1
    #         fn_series_sizes.append(fn_series_size)
    #     else:
    #         fn_idx += 1
    # plot_hist(fn_series_sizes, 'fn_series_sizes')
    #
    # fn_min_dists_to_seizure = list()
    # for fn_idx, fn in enumerate(fn_mask):
    #     if not fn:
    #         continue
    #
    #     fn_time_start = time_idxs_start[fn_idx] / sfreq
    #     min_dist_to_seizure = 1e9
    #     for seizure in seizures:
    #         min_dist_to_seizure = min(min_dist_to_seizure, abs(fn_time_start - seizure['start']))
    #         min_dist_to_seizure = min(min_dist_to_seizure, abs(fn_time_start - seizure['end']))
    #     fn_min_dists_to_seizure.append(min_dist_to_seizure)
    #
    # plot_hist(fn_min_dists_to_seizure, 'fn_min_dists_to_seizure')

    metric_dict = {
        'f1_score': f1_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'fp_num': fp_num,
        'fn_num': fn_num,
        'duration': (time_idxs_end[-1] / sfreq - time_idxs_start[0] / sfreq) / 3600,
    }

    # visualization
    if visualization_path is not None:
        fig = go.Figure()
        fig.update_layout(title_text=f'{subject_key} by {experiment_name} filter = {filter} k = {k_size}')

        fig.add_trace(
            go.Scatter(x=time_idxs_start / sfreq, y=probs_filtered, name='probs')
        )

        # add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
            )
        )

        # seizures
        fig.add_trace(
            go.Scatter(
                x=[seizure['start'] + ((seizure['end'] - seizure['start']) / 2) for seizure in seizures],
                y=[1.1 for _ in range(len(seizures))],
                text=[f'seizure #{seizure_idx:02}' for seizure_idx, seizure in enumerate(seizures)],
                mode="text",
            )
        )

        seizure_shapes = [
            dict(
                fillcolor="rgba(255, 0, 0, 0.2)",
                line={"width": 0},
                type="rect",
                x0=seizure['start'],
                x1=seizure['end'],
                y0=0,
                y1=1.1,
            )
            for seizure in seizures
        ]

        # errors
        fp_shapes = [
            dict(
                fillcolor="rgba(255, 0, 255, 0.5)",
                line={"width": 0},
                type="rect",
                x0=time_idxs_start[fp_idx] / sfreq,
                x1=time_idxs_end[fp_idx] / sfreq,
                y0=0,
                y1=1,
            )
            for fp_idx, fp in enumerate(fp_mask) if fp
        ]

        fn_shapes = [
            dict(
                fillcolor="rgba(0, 0, 0, 0.2)",
                line={"width": 0},
                type="rect",
                x0=time_idxs_start[fn_idx] / sfreq,
                x1=time_idxs_end[fn_idx] / sfreq,
                y0=0,
                y1=1,
            )
            for fn_idx, fn in enumerate(fn_mask) if fn
        ]

        fig.update_layout(shapes=seizure_shapes + fp_shapes + fn_shapes)

        # add threshold line
        fig.add_shape(
            type='line',
            x0=0,
            y0=threshold,
            x1=time_idxs_end[-1] / sfreq,
            y1=threshold,
            line=dict(color='Red', ),
            xref='x',
            yref='y'
        )

        # fig.show()
        fig.write_html(visualization_path, auto_open=False)

    return metric_dict


if __name__ == '__main__':
    import utils.avg_meters
    metric_meter = utils.avg_meters.MetricMeter()

    # experiment_name = '14062023_resnet18_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm'
    # experiment_name = '14062023_resnet18_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm_validation_only'
    # experiment_name = '24072023_efficientnet_b0_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm'
    experiment_name = '30072023_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_continue'
    # experiment_name = 'renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug'
    # experiment_name = 'renset18_2nd_stage_MixUp_SpecTimeFlipEEGFlipAug'
    visualizations_dir = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\visualizations'
    os.makedirs(visualizations_dir, exist_ok=True)

    threshold = 0.95
    # filter_method = None
    # k_size = -1
    filter_method = 'median'
    k_size = 7
    print(f'threshold = {threshold:03.2f} filter={filter_method} k={k_size}')
    subject_keys = [
        # # 'data2/038tl Anonim-20190821_113559-20211123_004935'
        # # 'data2/008tl Anonim-20210204_131328-20211122_160417'
        #
        # 'data2/018tl Anonim-20201211_130036-20211122_163611',
        # 'data2/006tl Anonim-20210208_063816-20211122_154113',
        # 'data1/dataset20',

        # stage_1
        # part1
        'data2/038tl Anonim-20190821_113559-20211123_004935',  # val
        'data2/027tl Anonim-20200309_195746-20211122_175315',  # val
        'data1/dataset27',  # val
        'data1/dataset14',  # val
        'data2/036tl Anonim-20201224_124349-20211122_181415',  # val
        'data2/041tl Anonim-20201115_222025-20211123_011114',  # val
        'data1/dataset24',  # val
        'data2/026tl Anonim-20210301_013744-20211122_174658',  # val

        'data2/020tl Anonim-20201218_071731-20211122_171454', 'data1/dataset13',
        'data2/018tl Anonim-20201211_130036-20211122_163611',
        'data2/038tl Anonim-20190822_155119-20211123_005457',
        'data2/025tl Anonim-20210128_233211-20211122_173425',
        'data2/015tl Anonim-20201116_134129-20211122_161958',

        # part2
        'data1/dataset3',
        'data2/027tl Anonim-20200310_035747-20211122_175503',
        'data2/002tl Anonim-20200826_124513-20211122_135804', 'data1/dataset23',
        'data2/022tl Anonim-20201210_132636-20211122_172649', 'data1/dataset6', 'data1/dataset11',
        'data2/021tl Anonim-20201223_085255-20211122_172126', 'data1/dataset28',

        # part3
        'data1/dataset1',
        'data2/008tl Anonim-20210204_131328-20211122_160417',
        'data2/003tl Anonim-20200831_120629-20211122_140327', 'data1/dataset12',
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

        # # stage_2
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
    ]
    # subject_keys = ['data2/027tl Anonim-20200309_195746-20211122_175315']
    for subject_key in subject_keys:
        prediction_path = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\predictions\{subject_key}.pickle'
        # prediction_path = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\predictions_positive_only\{subject_key}.pickle'
        prediction_data = pickle.load(open(prediction_path, 'rb'))

        set_visualizations_dir = os.path.join(visualizations_dir, subject_key.split('/')[0])
        os.makedirs(set_visualizations_dir, exist_ok=True)

        visualization_path = os.path.join(visualizations_dir, f'{subject_key}_filter={filter_method}_k={k_size}.html')
        subject_metrics = process_subject_predictions(prediction_data, threshold, filter_method, k_size, visualization_path=visualization_path)
        metric_meter.update(subject_metrics)

        print(f'subject_key = {subject_key:60} subject_metrics {" ".join([f"{key} = {value:9.4f}" for key, value in subject_metrics.items()])}')
    print('metric_meter\n', metric_meter)
