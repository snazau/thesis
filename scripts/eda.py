import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import data_split


if __name__ == '__main__':
    exclude_16 = False
    metainfo_path = '../data/dataset_info.json'

    with open(metainfo_path) as f:
        dataset_info = json.load(f)

    subjects_to_exclude = data_split.get_subject_keys_exclude_16() if exclude_16 else list()

    total_duration, total_normal, total_seizure = 0, 0, 0
    total_percent, avg_percent = 0, 0

    whole_durations = list()
    all_record_durations = list()
    all_break_btw_seizure = list()
    all_seizure_nums = list()
    all_seizure_durations = list()
    data1_seizure_durations = list()
    data2_seizure_durations = list()
    for subject_key in dataset_info['subjects_info'].keys():
        if subject_key in subjects_to_exclude:
            continue

        subject_seizures = dataset_info['subjects_info'][subject_key]['seizures']
        all_seizure_nums.append(len(subject_seizures))
        all_record_durations.append(dataset_info['subjects_info'][subject_key]['duration_in_seconds'])

        seizure_prev = None
        subject_seizure_duration = 0
        for seizure in subject_seizures:
            seizure_duration = seizure['end'] - seizure['start']

            if seizure_duration > 300:
                all_seizure_nums[-1] = all_seizure_nums[-1] - 1
                if all_seizure_nums[-1] == 0:
                    del all_seizure_nums[-1]
                print('>300', subject_key, seizure_duration, seizure)
                continue

            if seizure_prev is not None:
                break_btw_seizure = seizure['start'] - seizure_prev['end']
                all_break_btw_seizure.append(break_btw_seizure)

            subject_seizure_duration += seizure_duration

            all_seizure_durations.append(seizure_duration)
            if 'data1' in subject_key:
                data1_seizure_durations.append(seizure_duration)
            elif 'data2' in subject_key:
                data2_seizure_durations.append(seizure_duration)
            else:
                raise NotImplementedError

            seizure_prev = seizure.copy()

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
    all_seizure_nums = np.array(all_seizure_nums)
    all_record_durations = np.array(all_record_durations)
    all_break_btw_seizure = np.array(all_break_btw_seizure)
    print(f'data1_seizure stats min = {data1_seizure_durations.min()} mean = {data1_seizure_durations.mean()} max = {data1_seizure_durations.max()} std = {data1_seizure_durations.std()}')
    print(f'data2_seizure stats min = {data2_seizure_durations.min()} mean = {data2_seizure_durations.mean()} max = {data2_seizure_durations.max()} std = {data2_seizure_durations.std()}')
    print(f'all_seizure_durations stats min = {all_seizure_durations.min()} mean = {all_seizure_durations.mean()} max = {all_seizure_durations.max()} std = {all_seizure_durations.std()}')
    print(f'all_seizure_nums stats min = {all_seizure_nums.min()} mean = {all_seizure_nums.mean()} max = {all_seizure_nums.max()} std = {all_seizure_nums.std()}')
    print(f'all_record_durations stats min = {all_record_durations.min() / 3600} mean = {all_record_durations.mean() / 3600} max = {all_record_durations.max() / 3600} std = {all_record_durations.std() / 3600}')
    print(f'all_break_btw_seizure stats min = {all_break_btw_seizure.min() / 3600} mean = {all_break_btw_seizure.mean() / 3600} max = {all_break_btw_seizure.max() / 3600} std = {all_break_btw_seizure.std() / 3600}')
    print(f'patients_num = {len(all_seizure_nums)} seizure_num = {len(all_seizure_durations)}')

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
