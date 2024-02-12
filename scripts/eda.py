import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


if __name__ == '__main__':
    metainfo_path = '../data/dataset_info.json'

    with open(metainfo_path) as f:
        dataset_info = json.load(f)

    total_duration = 0
    total_normal = 0
    total_seizure = 0
    total_percent = 0
    avg_percent = 0

    whole_durations = list()
    all_seizure_durations = list()
    data1_seizure_durations = list()
    data2_seizure_durations = list()
    for subject_key in dataset_info['subjects_info'].keys():
        subject_seizure_duration = 0
        for seizure in dataset_info['subjects_info'][subject_key]['seizures']:
            seizure_duration = seizure['end'] - seizure['start']

            if seizure_duration > 300:
                print('>300', subject_key, seizure_duration, seizure)
                continue

            subject_seizure_duration += seizure_duration

            all_seizure_durations.append(seizure_duration)
            if 'data1' in subject_key:
                data1_seizure_durations.append(seizure_duration)
            elif 'data2' in subject_key:
                data2_seizure_durations.append(seizure_duration)
            else:
                raise NotImplementedError

        subject_normal_duration = dataset_info['subjects_info'][subject_key]['duration_in_seconds'] - subject_seizure_duration
        subject_total_duration = dataset_info['subjects_info'][subject_key]['duration_in_seconds']

        whole_durations.append(subject_total_duration)
        total_duration += subject_total_duration
        total_normal += subject_normal_duration
        total_seizure += subject_seizure_duration

        avg_percent += subject_seizure_duration / subject_total_duration
    avg_percent /= len(dataset_info['subjects_info'].keys())
    total_percent = total_seizure / total_duration
    whole_durations = np.array(whole_durations)
    whole_duration_avg = whole_durations.mean()

    print(f'avg_percent = {avg_percent * 100:.4f}% total_percent = {total_percent * 100:.4f}%')
    print(f'total_normal = {total_normal / 3600:.2f}h ({total_normal / total_duration * 100:.2f}%) total_seizure = {total_seizure / 3600:.2f}h ({total_seizure / total_duration * 100:.2f}%) total_duration = {total_duration / 3600:.2f}h (100.00%)')
    print(f'whole_duration_avg = {whole_duration_avg / 3600}h')

    all_seizure_durations = np.array(all_seizure_durations)
    data1_seizure_durations = np.array(data1_seizure_durations)
    data2_seizure_durations = np.array(data2_seizure_durations)
    print(f'data1_seizure stats min = {data1_seizure_durations.min()} mean = {data1_seizure_durations.mean()} max = {data1_seizure_durations.max()} std = {data1_seizure_durations.std()}')
    print(f'data2_seizure stats min = {data2_seizure_durations.min()} mean = {data2_seizure_durations.mean()} max = {data2_seizure_durations.max()} std = {data2_seizure_durations.std()}')
    print(f'all_seizure stats min = {all_seizure_durations.min()} mean = {all_seizure_durations.mean()} max = {all_seizure_durations.max()} std = {all_seizure_durations.std()}')
    print(f'seizure_num = {len(all_seizure_durations)}')

    # log_bins = np.logspace(np.log10(50), np.log10(300), 25)
    # plt.hist([data1_seizure_durations, data2_seizure_durations], bins=log_bins, label=['data1', 'data2'])
    # plt.gca().set_xscale("log")
    bins = np.linspace(10, 300, 25)
    plt.hist([data1_seizure_durations, data2_seizure_durations], color=['red', 'blue'], alpha=0.5, bins=bins, label=['data1', 'data2'], density=True)
    plt.plot(bins, stats.norm.pdf(bins, data1_seizure_durations.mean(), data1_seizure_durations.std()), 'r--')
    plt.plot(bins, stats.norm.pdf(bins, data2_seizure_durations.mean(), data2_seizure_durations.std()), 'b--')
    plt.legend(loc='upper right')
    plt.title('Seizure durations')
    plt.show()

    bins = np.linspace(10, 300, 25)
    plt.hist(
        [all_seizure_durations],
        color=['gray'],
        alpha=0.5,
        bins=bins,
        label=['all'],
        density=True
    )
    # plt.plot(bins, stats.norm.pdf(bins, all_seizure_durations.mean(), all_seizure_durations.std()), 'r-', linewidth=2)
    # plt.axvline(x=all_seizure_durations.mean(), ymin=0.05, ymax=0.95, color='r', linewidth=2)
    plt.axvline(x=all_seizure_durations.mean(), color='red', linewidth=2)
    plt.title('Distribution of seizure duration', fontsize=18)
    plt.xlabel('Seizure duration, sec', fontsize=18)
    plt.ylabel('P', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
