import os

import matplotlib.pyplot as plt
import numpy as np

import scripts.process_segment_predictions


if __name__ == '__main__':
    experiment_name_to_public_name = {
        'renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug': 'Baseline CNN',
        '20231107_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NoFilt': 'Error-aware CNN',
        '20231213_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_OCSVM_positive_only_16excluded': 'OCSVM + CNN',
    }

    # global settings
    split_name = 'base'
    # split_name = '602020_v1'
    # split_name = '602020_v2'
    visualize_segments = False
    exclude_16 = True
    apply_filter = True
    verbose = 1
    min_recall_threshold = 0.9
    intersection_part_threshold = 0.51
    seizure_segments_true_dilation = 60 * 1
    # min_pred_duration = -1
    # max_pred_duration = float('inf')

    min_pred_durations = [-1, 10, 20, 30, 40]
    # min_pred_durations = [-1, 10]
    max_pred_durations = [float('inf'), 260, 240, 220, 200]
    # max_pred_durations = [float('inf'), 26]

    for experiment_name in experiment_name_to_public_name.keys():
        experiment_dir = os.path.join(rf'D:\Study\asp\thesis\implementation\experiments', experiment_name)

        metric_to_dynamic_table = dict()
        for min_pred_duration_idx, min_pred_duration in enumerate(min_pred_durations):
            for max_pred_duration_idx, max_pred_duration in enumerate(max_pred_durations):
                (
                    best_threshold_10sec_val,
                    metric_meter_10sec_train,
                    metric_meter_10sec_val,
                    metric_meter_10sec_test,
                    best_threshold_segment_merging_val,
                    metric_meter_segment_merging_train,
                    metric_meter_segment_merging_val,
                    metric_meter_segment_merging_test,
                ) = scripts.process_segment_predictions.process(
                    experiment_name,
                    experiment_dir,
                    split_name,
                    exclude_16,
                    apply_filter,
                    seizure_segments_true_dilation,
                    intersection_part_threshold,
                    min_pred_duration,
                    max_pred_duration,
                    min_recall_threshold,
                    visualize_segments,
                    verbose,
                )

                for metric_name in metric_meter_segment_merging_test.meters.keys():
                    if metric_name not in metric_to_dynamic_table:
                        metric_to_dynamic_table[metric_name] = np.zeros((len(min_pred_durations), len(max_pred_durations)), dtype=np.float32)

                    if metric_name.endswith('_num'):
                        metric_value = metric_meter_segment_merging_test.meters[metric_name].sum
                    else:
                        metric_value = metric_meter_segment_merging_test.meters[metric_name].avg
                    metric_to_dynamic_table[metric_name][min_pred_duration_idx, max_pred_duration_idx] = metric_value

        metric_names = [
            'tp_num', 'fp_num', 'fn_num',
            'f1_score_micro', 'precision_score_micro', 'recall_score_micro',
        ]
        fig_height, fig_width = 2 * 5, 3 * 5
        fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        for metric_name_idx, metric_name in enumerate(metric_names):
            axes_row_idx, axes_col_idx = metric_name_idx // 3, metric_name_idx % 3

            axis = axes[axes_row_idx, axes_col_idx]
            matrix = metric_to_dynamic_table[metric_name]
            # vmin = matrix.min() if metric_name.endswith('_num') else 0
            vmin = matrix.min()
            # vmax = matrix.max() if metric_name.endswith('_num') else 1
            vmax = matrix.max()
            im = axis.imshow(matrix, cmap='Reds', vmin=vmin, vmax=vmax)
            axis.set_xticks(np.arange(len(max_pred_durations)), labels=[str(v) for v in max_pred_durations])
            axis.set_yticks(np.arange(len(min_pred_durations)), labels=[str(v) for v in min_pred_durations])
            axis.set_title(metric_name)
            axis.set_xlabel('Max pred duration')
            axis.set_ylabel('Min pred duration')

            for i in range(len(min_pred_durations)):
                for j in range(len(max_pred_durations)):
                    value_text = f'{int(matrix[i, j]):d}' if metric_name.endswith('_num') else f'{matrix[i, j]:.2f}'
                    text = axis.text(j, i, f'{matrix[i, j]:.2f}', ha="center", va="center", color="black")

        fig.suptitle(experiment_name_to_public_name[experiment_name])

        save_path = os.path.join(experiment_dir, 'metrics_dynamics.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
