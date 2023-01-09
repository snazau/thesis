import json


if __name__ == '__main__':
    metainfo_path = '../data/dataset_info.json'

    with open(metainfo_path) as f:
        dataset_info = json.load(f)

    total_duration = 0
    total_normal = 0
    total_seizure = 0
    total_percent = 0
    avg_percent = 0

    for subject_key in dataset_info['subjects_info'].keys():
        subject_seizure_duration = 0
        for seizure in dataset_info['subjects_info'][subject_key]['seizures']:
            subject_seizure_duration += seizure['end'] - seizure['start']

        subject_normal_duration = dataset_info['subjects_info'][subject_key]['duration_in_seconds'] - subject_seizure_duration
        subject_total_duration = dataset_info['subjects_info'][subject_key]['duration_in_seconds']

        total_duration += subject_total_duration
        total_normal += subject_normal_duration
        total_seizure += subject_seizure_duration

        avg_percent += subject_seizure_duration / subject_total_duration
    avg_percent /= len(dataset_info['subjects_info'].keys())
    total_percent = total_seizure / total_duration

    print(f'avg_percent = {avg_percent * 100:.4f}% total_percent = {total_percent * 100:.4f}%')
    print(f'total_normal = {total_normal / 3600:.2f}h total_seizure = {total_seizure / 3600:.2f}h total_duration = {total_duration / 3600:.2f}h')
