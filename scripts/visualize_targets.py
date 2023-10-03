import json
import os

import datasets
import eeg_reader
import visualization


if __name__ == '__main__':
    dataset_info_path = r'D:\Study\asp\thesis\implementation\data\dataset_info.json'
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    # subject loop
    save_dir = os.path.join(r'D:\Study\asp\thesis\implementation\data', 'seizure_visual')
    os.makedirs(save_dir, exist_ok=True)

    data_dir = r'D:\Study\asp\thesis\implementation\data'
    for subject_idx, subject_key in enumerate(dataset_info['subjects_info'].keys()):
        print(f'\rProgress {subject_idx + 1}/{len(dataset_info["subjects_info"].keys())} {subject_key}', end='')

        seizure_0_path = os.path.join(save_dir, f'{subject_key.replace("/", "_")}_seizure0.png')
        if os.path.exists(seizure_0_path):
            continue

        recording_duration = dataset_info['subjects_info'][subject_key]['duration_in_seconds']
        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        subject_eeg_path = os.path.join(data_dir, subject_key + ('.dat' if 'data1' in subject_key else '.edf'))

        raw_data = eeg_reader.EEGReader.read_eeg(subject_eeg_path)
        channel_names = raw_data.info['ch_names']
        datasets.drop_unused_channels(subject_eeg_path, raw_data)

        for seizure_idx, seizure in enumerate(subject_seizures):
            try:
                start_time = max(0, seizure['start'] - 60 * 2)
                duration = seizure['end'] - seizure['start'] + 60 * 4
                if (start_time + duration) > recording_duration:
                    duration = duration - (start_time + duration - recording_duration) - 1

                seizure_sample, _, _ = datasets.generate_raw_samples(
                    raw_data,
                    sample_start_times=[start_time],
                    sample_duration=duration,
                )
                # print(f'{subject_key} seizure_idx = {seizure_idx} seizure_sample = {seizure_sample.shape}')

                seizure_local = {
                    'start': 60 * 2,
                    'end': seizure_sample.shape[-1] // 128 - 60 * 2,
                }

                save_path = os.path.join(save_dir, f'{subject_key.replace("/", "_")}_seizure{int(seizure_idx)}.png')
                visualization.visualize_raw(
                    seizure_sample[0],
                    channel_names,
                    seizure_idxs=seizure_local,
                    heatmap=None,
                    save_path=save_path,
                )
            except Exception as e:
                print(f'Something went wrong {subject_key} {seizure_idx}')
                print(e)
                print('\n\n\n')
                # raise e
        # break
