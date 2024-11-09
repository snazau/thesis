import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import visualization


if __name__ == '__main__':
    # data_dir = r'D:\Study\asp\thesis\implementation\experiments\renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug\vis_20241104\tp'
    # data_dir = r'D:\Study\asp\thesis\implementation\experiments\20231213_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_OCSVM_positive_only_16excluded\vis_20241104\tp'
    data_dir = r'D:\Study\asp\thesis\implementation\experiments\20231107_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_NoFilt\vis_20241104\tp'

    # get paths only for csv stats
    csv_paths = [
        os.path.join(data_dir, file_name)
        for file_name in sorted(os.listdir(data_dir))
        if file_name.endswith('_stats.csv')
    ]

    # aggregate multiple csvs for a single patient
    subject_key_to_csv_paths = dict()
    for csv_path in csv_paths:
        file_name = os.path.basename(csv_path)
        subject_key = '_'.join(file_name.split('_')[:-2])
        subject_key = subject_key.replace('_', '/', 1)
        print(f'subject_key = {subject_key} file_name = {file_name}')

        if subject_key not in subject_key_to_csv_paths:
            subject_key_to_csv_paths[subject_key] = list()

        subject_key_to_csv_paths[subject_key].append(csv_path)

    import pprint
    pprint.pprint(subject_key_to_csv_paths)

    # get rows and cols names
    channel_names = ['EEG Fp1', 'EEG Fp2', 'EEG F7', 'EEG F3', 'EEG Fz', 'EEG F4', 'EEG F8', 'EEG T3', 'EEG C3', 'EEG Cz', 'EEG C4', 'EEG T4', 'EEG T5', 'EEG P3', 'EEG Pz', 'EEG P4', 'EEG T6', 'EEG O1', 'EEG O2', 'EEG F9', 'EEG T9', 'EEG P9', 'EEG F10', 'EEG T10', 'EEG P10']
    channel_groups = {
        'frontal': {
            'channel_names': ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10']])
            ],
        },
        'central': {
            'channel_names': ['C3', 'Cz', 'C4'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['C3', 'Cz', 'C4']])
            ],
        },
        'perietal-occipital': {
            'channel_names': ['P3', 'Pz', 'P4', 'O1', 'O2'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['P3', 'Pz', 'P4', 'O1', 'O2']])
            ],
        },
        'temporal-left': {
            'channel_names': ['T9', 'T3', 'P9', 'T5'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['T9', 'T3', 'P9', 'T5']])
            ],
        },
        'temporal-right': {
            'channel_names': ['T10', 'T4', 'P10', 'T6'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['T10', 'T4', 'P10', 'T6']])
            ],
        },
    }
    freq_ranges = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 30),
        'gamma': (30, 40),
    }

    channel_group_names = [
        f'{"".join([name[0] for name in channel_group_name.split("-")])}'.upper()
        for channel_group_name in channel_groups.keys()
    ]
    freq_range_names = [fr'$\{freq_range_name}$' for freq_range_name in freq_ranges.keys()]
    freq_range_names.reverse()

    # build illustrations for patients
    segment_num_dataset = 0
    avg_matrix_importance_dataset = np.zeros((len(freq_ranges), len(channel_groups)), dtype=np.float32)
    for subject_key in subject_key_to_csv_paths.keys():
        seizure_df_list = list()
        for csv_path in subject_key_to_csv_paths[subject_key]:
            seizure_df = pd.read_csv(csv_path)
            seizure_df_list.append(seizure_df)

        seizure_num = len(seizure_df_list)
        max_segments_num = max([len(seizure_df) for seizure_df in seizure_df_list])
        max_segments_num = max(max_segments_num, 2)

        fig_width = 5 * max_segments_num
        fig_height = 5 * (seizure_num + 1)
        fig, axes = plt.subplots(seizure_num + 1, max_segments_num, figsize=(fig_width, fig_height))
        fig.suptitle(subject_key, fontsize=20)

        segment_num_subject = 0
        avg_matrix_importance_subject = np.zeros((len(freq_ranges), len(channel_groups)), dtype=np.float32)
        for seizure_idx, seizure_df in enumerate(seizure_df_list):
            data_np = seizure_df.to_numpy()
            matrix_importance_seizure = [
                ast.literal_eval(" ".join(matrix_str.split()).replace(" ", ",").replace("]\n[", "],[").replace(",,", ","))
                for matrix_str in seizure_df['importance_matrix']
            ]
            matrix_importance_seizure = np.array(matrix_importance_seizure)

            avg_matrix_importance_seizure = np.zeros((len(freq_ranges), len(channel_groups)), dtype=np.float32)
            for segment_idx in range(matrix_importance_seizure.shape[0]):
                segment_num_dataset += 1
                segment_num_subject += 1

                avg_matrix_importance_dataset += matrix_importance_seizure[segment_idx]
                avg_matrix_importance_subject += matrix_importance_seizure[segment_idx]
                avg_matrix_importance_seizure += matrix_importance_seizure[segment_idx]

                segment_rel_time_start = int(seizure_df['start_rel'][segment_idx])
                segment_rel_time_end = int(seizure_df['end_rel'][segment_idx])
                segment_idx_in_spectrum_visuals = int(seizure_df['start_rel'][segment_idx] / 10)
                visualization.visualize_importance_matrix(
                    importance_matrix=matrix_importance_seizure[segment_idx],
                    freq_range_names=freq_range_names,
                    channel_group_names=channel_group_names,
                    axis=axes[seizure_idx, segment_idx],
                    vmin=matrix_importance_seizure.min(),
                    vmax=matrix_importance_seizure.max(),
                )
                axes[seizure_idx, segment_idx].set_title(f'{segment_rel_time_start}-{segment_rel_time_end} secs\nsegment #{segment_idx_in_spectrum_visuals:02}', fontsize=12)
                if segment_idx == 0:
                    axes[seizure_idx, segment_idx].set_ylabel(f'seizure #{seizure_idx:02}')

            avg_matrix_importance_seizure = avg_matrix_importance_seizure / matrix_importance_seizure.shape[0]
            visualization.visualize_importance_matrix(
                importance_matrix=avg_matrix_importance_seizure,
                freq_range_names=freq_range_names,
                channel_group_names=channel_group_names,
                axis=axes[seizure_num, seizure_idx],
                vmin=avg_matrix_importance_seizure.min(),
                vmax=avg_matrix_importance_seizure.max(),
            )
            axes[seizure_num, seizure_idx].set_title(f'avg. seizure #{seizure_idx:02}', fontsize=18)

        avg_matrix_importance_subject = avg_matrix_importance_subject / segment_num_subject
        visualization.visualize_importance_matrix(
            importance_matrix=avg_matrix_importance_subject,
            freq_range_names=freq_range_names,
            channel_group_names=channel_group_names,
            axis=axes[seizure_num, seizure_num],
            vmin=avg_matrix_importance_subject.min(),
            vmax=avg_matrix_importance_subject.max(),
        )
        axes[seizure_num, seizure_num].set_title('avg. subject', fontsize=18)

        save_path = os.path.join(data_dir, f'{subject_key.replace("/", "_")}_MI.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
        # plt.show()

        # break

    avg_matrix_importance_dataset = avg_matrix_importance_dataset / segment_num_dataset
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    visualization.visualize_importance_matrix(
        importance_matrix=avg_matrix_importance_dataset,
        freq_range_names=freq_range_names,
        channel_group_names=channel_group_names,
        axis=ax,
        vmin=avg_matrix_importance_dataset.min(),
        vmax=avg_matrix_importance_dataset.max(),
    )
    plt.title('Dataset')
    save_path = os.path.join(data_dir, f'dataset_MI.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
    # plt.show()
