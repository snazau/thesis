import os
import pickle

import plotly.graph_objects as go

from utils.common import filter_predictions, calc_metrics


def visualize_prediction(prediction_data, probs_filtered, visualization_path, sfreq=128):
    # get data from prediction_data
    time_idxs_start = prediction_data['time_idxs_start']
    time_idxs_end = prediction_data['time_idxs_end']
    seizures = prediction_data['subject_seizures']
    labels = prediction_data['labels']

    fp_mask = ((labels == 0) & (probs_filtered > threshold))
    fn_mask = ((labels == 1) & (probs_filtered <= threshold))

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


if __name__ == '__main__':
    import utils.avg_meters
    metric_meter = utils.avg_meters.MetricMeter()

    # experiment_name = '14062023_resnet18_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm'
    # experiment_name = '14062023_resnet18_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm_validation_only'
    # experiment_name = '24072023_efficientnet_b0_all_subjects_SpecTimeFlipEEGFlipAug_baseline_correction_minmax_norm'
    # experiment_name = '30072023_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_continue'
    # experiment_name = '08082023_efficientnet_b0_all_subjects_MixUp_TimeSeriesAug_raw'
    # experiment_name = '14082023_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_preprocessed_ocsvm'
    # experiment_name = '20230821_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_cwt_meanstd'
    # experiment_name = 'renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug'
    # experiment_name = 'renset18_2nd_stage_MixUp_SpecTimeFlipEEGFlipAug'
    # experiment_name = '20230912_efficientnet_b0_all_subjects_SpecTimeFlipEEGFlipAug_log_power_BCELossWithTimeToClosestSeizure'
    # experiment_name = '20230925_efficientnet_b0_all_subjects_MixUp_SpecTimeFlipEEGFlipAug_log_power_16excluded'
    # experiment_name = '20231005_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded_wo_baseline_correction'
    # experiment_name = '20231005_EEGResNetCustomRaw_MixUp_TimeSeriesAug_raw_16excluded'
    experiment_name = '20231012_CRNN_EEGResNetCustomRaw_BCERecurrentLoss_16excluded'
    visualizations_dir = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\visualizations_test'
    os.makedirs(visualizations_dir, exist_ok=True)

    sfreq = 128
    # threshold = 0.35
    threshold = 0.95
    # filter_method = None
    # k_size = -1
    filter_method = 'median'
    k_size = 7
    print(experiment_name)
    print(f'threshold = {threshold:03.2f} filter={filter_method} k={k_size}')
    subject_keys = [
        # 'data2/038tl Anonim-20190821_113559-20211123_004935'

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
        'data2/022tl Anonim-20201210_132636-20211122_172649',
        'data1/dataset6', 'data1/dataset11',
        'data2/021tl Anonim-20201223_085255-20211122_172126', 'data1/dataset28',

        # part3
        'data2/008tl Anonim-20210204_131328-20211122_160417',
        'data2/003tl Anonim-20200831_120629-20211122_140327',
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
        'data1/dataset1',
        'data1/dataset12',

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
    # subject_keys = ['data2/027tl Anonim-20200309_195746-20211122_175315']
    for subject_key in subject_keys:
        # read predictions
        prediction_path = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\predictions\{subject_key}.pickle'
        # prediction_path = rf'D:\Study\asp\thesis\implementation\experiments\{experiment_name}\predictions_positive_only\{subject_key}.pickle'
        prediction_data = pickle.load(open(prediction_path, 'rb'))

        # extract attributes
        time_idxs_start = prediction_data['time_idxs_start']
        time_idxs_end = prediction_data['time_idxs_end']
        labels = prediction_data['labels']
        probs = prediction_data['probs_wo_tta'] if 'probs_wo_tta' in prediction_data else prediction_data['probs']
        if len(probs) == 0:
            probs = prediction_data['probs']

        # smooth predictions
        probs_filtered = filter_predictions(probs, filter_method, k_size)

        # calc metrics over smoothed predictions
        subject_metrics = calc_metrics(probs_filtered, labels, threshold)
        subject_metrics['duration'] = (time_idxs_end[-1] / sfreq - time_idxs_start[0] / sfreq) / 3600

        # save visualizations
        visualization_path = os.path.join(visualizations_dir, f'{subject_key}_filter={filter_method}_k={k_size}.html')
        visualize_prediction(prediction_data, probs_filtered, visualization_path)
        metric_meter.update(subject_metrics)

        print(f'subject_key = {subject_key:60} subject_metrics {" ".join([f"{key} = {value:9.4f}" for key, value in subject_metrics.items()])}')
    print('metric_meter\n', metric_meter)
