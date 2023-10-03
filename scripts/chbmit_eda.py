import os

import mne

import visualization


def parse_summary_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    duration_total, seizures_total = 0, 0
    file_name_to_seizures = dict()
    for line_idx, line in enumerate(lines):
        line = line.strip()

        if line.startswith('File Start Time'):
            file_start_time_str = line.split('File Start Time: ')[-1].strip()
            file_start_hour, file_start_min, file_start_sec = file_start_time_str.split(':')
            file_start_hour, file_start_min, file_start_sec = int(file_start_hour), int(file_start_min), int(file_start_sec)
            file_start_time = file_start_hour * 3600 + file_start_min * 60 + file_start_sec

            file_end_time_str = lines[line_idx + 1].split('File End Time: ')[-1].strip()
            file_end_hour, file_end_min, file_end_sec = file_end_time_str.split(':')
            file_end_hour, file_end_min, file_end_sec = int(file_end_hour), int(file_end_min), int(file_end_sec)
            file_end_time = file_end_hour * 3600 + file_end_min * 60 + file_end_sec

            duration_curr = file_end_time - file_start_time
            duration_total += duration_curr
            continue

        if not line.startswith('Number of Seizures'):
            continue

        if line == 'Number of Seizures in File: 0':
            continue

        seizures_num = int(line.split(' ')[-1])
        seizures_total += seizures_num

        # seizure times parsing
        for seizure_idx in range(seizures_num):
            if lines[line_idx - 3].startswith('File Name'):
                file_name = lines[line_idx - 3].split(' ')[-1].strip()
            else:
                file_name = lines[line_idx - 1].split(' ')[-1].strip()
            seizure_start_time = int(lines[line_idx + 1 + seizure_idx * 2].split(' ')[-2])
            seizure_end_time = int(lines[line_idx + 1 + seizure_idx * 2 + 1].split(' ')[-2])

            if file_name not in file_name_to_seizures:
                file_name_to_seizures[file_name] = list()
            file_name_to_seizures[file_name].append({
                'start': seizure_start_time,
                'end': seizure_end_time,
            })

    return file_name_to_seizures, duration_total, seizures_total


def visualize_seizures(raw_path, seizures):
    file_name = os.path.splitext(os.path.basename(raw_path))[0]
    raw = mne.io.read_raw_edf(raw_path, preload=True)
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    channel_names = raw.info['ch_names']

    for seizure_idx, seizure in enumerate(seizures):
        start_time = max(0, seizure['start'] - 60 * 2)
        duration = seizure['end'] - seizure['start'] + 60 * 4
        # if (start_time + duration) > recording_duration:
        #     duration = duration - (start_time + duration - recording_duration) - 1

        start_idx = int(start_time * sfreq)
        end_idx = int(min(data.shape[1], (start_time + duration) * sfreq))
        sample = data[:, start_idx:end_idx]

        seizure_local = {
            'start': 60 * 2,
            'end': sample.shape[-1] // 128 - 60 * 2,
        }

        save_dir = r'D:\Study\asp\thesis\implementation\data\seizure_visual_chbmit'
        save_path = os.path.join(save_dir, f'{file_name}_seizure{int(seizure_idx)}.png')
        visualization.visualize_raw(
            sample,
            channel_names,
            seizure_idxs=seizure_local,
            heatmap=None,
            save_path=save_path,
            trim_channels=False,
        )


def visualize_patient(patient_dir):
    patient_name = os.path.basename(patient_dir)
    summary_path = os.path.join(patient_dir, f'{patient_name}-summary.txt')
    file_name_to_seizures, duration, seizures_num = parse_summary_file(summary_path)
    print(f'{patient_name} {duration / 3600:.2f}h seizures_num = {seizures_num}')
    for file_idx, (file_name, seizures) in enumerate(file_name_to_seizures.items()):
        raw_path = os.path.join(patient_dir, file_name)
        visualize_seizures(raw_path, seizures)


if __name__ == '__main__':
    dataset_path = r'D:\Study\asp\thesis\implementation\data_orig\chb-mit-scalp-eeg-database-1.0.0'
    for patient_idx, file_name in enumerate(sorted(os.listdir(dataset_path))):
        patient_dir = os.path.join(dataset_path, file_name)
        if not os.path.isdir(patient_dir):
            continue

        # if patient_idx < 28:
        #     continue

        # if file_name != 'chb23':
        #     continue

        print(f'\rProgress {patient_idx + 1} {file_name}', end='')

        visualize_patient(patient_dir)
        # break
