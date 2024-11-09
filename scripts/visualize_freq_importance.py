import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    data_dir = r'D:\Study\asp\thesis\implementation\experiments\renset18_all_subjects_MixUp_SpecTimeFlipEEGFlipAug\visual_GRADCAM_20241019_upd\tp'
    # data_dir = r'D:\Study\asp\thesis\implementation\experiments\20231213_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_OCSVM_positive_only_16excluded\vis_20241020\tp'

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

    # build illustrations for patients
    freqs = np.arange(1, 40.01, 0.1)
    xticks = np.arange(0, 40.01, 5)
    xticks[0] = 1
    print(freqs.shape)

    segment_num_dataset = 0
    # avg_freq_importance_dataset = np.empty((0, len(freqs)), dtype=np.float32)
    avg_freq_importance_dataset = np.zeros_like(freqs, dtype=np.float32)
    for subject_key in subject_key_to_csv_paths.keys():
        # if subject_key != 'data2/020tl Anonim-20201218_071731-20211122_171454':
        #     continue

        seizure_df_list = list()
        for csv_path in subject_key_to_csv_paths[subject_key]:
            seizure_df = pd.read_csv(csv_path)
            seizure_df_list.append(seizure_df)

        seizure_num = len(seizure_df_list)
        max_segments_num = max([len(seizure_df) for seizure_df in seizure_df_list])

        fig_width = 5 * max_segments_num
        fig_height = 5 * (seizure_num + 1)
        fig, axes = plt.subplots(seizure_num + 1, max_segments_num, figsize=(fig_width, fig_height))
        fig.suptitle(subject_key, fontsize=20)

        freq_1_idx = list(seizure_df_list[0].columns).index('freq_1.0_importance')
        freq_40_idx = list(seizure_df_list[0].columns).index('freq_40.0_importance')

        segment_num_subject = 0
        avg_freq_importance_subject = np.zeros_like(freqs, dtype=np.float32)
        for seizure_idx, seizure_df in enumerate(seizure_df_list):
            data_np = seizure_df.to_numpy()
            freq_importance_seizure = data_np[:, freq_1_idx:freq_40_idx + 1]  # (N_10, F)

            avg_freq_importance_seizure = np.zeros_like(freqs, dtype=np.float32)
            for segment_idx in range(freq_importance_seizure.shape[0]):
                segment_num_dataset += 1
                segment_num_subject += 1

                avg_freq_importance_dataset += freq_importance_seizure[segment_idx]
                avg_freq_importance_subject += freq_importance_seizure[segment_idx]
                avg_freq_importance_seizure += freq_importance_seizure[segment_idx]

                segment_rel_time_start = int(seizure_df['start_rel'][segment_idx])
                segment_rel_time_end = int(seizure_df['end_rel'][segment_idx])
                segment_idx_in_spectrum_visuals = int(seizure_df['start_rel'][segment_idx] / 10)
                axes[seizure_idx, segment_idx].plot(freqs, freq_importance_seizure[segment_idx])
                axes[seizure_idx, segment_idx].set_title(f'{segment_rel_time_start}-{segment_rel_time_end} secs\nsegment #{segment_idx_in_spectrum_visuals:02}', fontsize=12)
                axes[seizure_idx, segment_idx].set_xticks(xticks)
                if segment_idx == 0:
                    axes[seizure_idx, segment_idx].set_ylabel(f'seizure #{seizure_idx:02}')
                # axes[seizure_idx, segment_idx].set_ylabel('importance')
                # axes[seizure_idx, segment_idx].set_xlabel('freq')

            avg_freq_importance_seizure = avg_freq_importance_seizure / freq_importance_seizure.shape[0]
            axes[seizure_num, seizure_idx].plot(freqs, avg_freq_importance_seizure)
            axes[seizure_num, seizure_idx].set_title(f'avg. seizure #{seizure_idx:02}', fontsize=18)
            axes[seizure_num, seizure_idx].set_xticks(xticks)
            # axes[seizure_num, seizure_idx].set_ylabel('importance')
            # axes[seizure_num, seizure_idx].set_xlabel('freq')

        avg_freq_importance_subject = avg_freq_importance_subject / segment_num_subject
        axes[seizure_num, seizure_num].plot(freqs, avg_freq_importance_subject)
        axes[seizure_num, seizure_num].set_title('avg. subject', fontsize=18)
        axes[seizure_num, seizure_num].set_xticks(xticks)
        # axes[seizure_num, seizure_num].set_ylabel('importance')
        # axes[seizure_num, seizure_num].set_xlabel('freq')

        save_path = os.path.join(data_dir, f'{subject_key.replace("/", "_")}_FI.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
        # plt.show()

    avg_freq_importance_dataset = avg_freq_importance_dataset / segment_num_dataset
    fig = plt.figure(figsize=(10, 5))
    plt.plot(freqs, avg_freq_importance_dataset)
    plt.title('Dataset')
    plt.xticks(xticks)
    plt.ylabel('importance')
    plt.xlabel('freq')
    save_path = os.path.join(data_dir, f'dataset_FI.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
    # plt.show()
