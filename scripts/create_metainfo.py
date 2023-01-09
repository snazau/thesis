import json
import os
import natsort

import eeg_reader
import utils.target_helper


if __name__ == '__main__':
    data_dir = os.path.join('..', 'data')

    # subject_ids & filenames to exclude. Filenames will extract in the following block of code based on ids
    subsets_info = {
        'data1': {
            'ids_to_exclude': list(),
            'filenames_to_exclude': list(),
        },
        'data2': {
            'ids_to_exclude': [18, 51],
            'filenames_to_exclude': list(),
        }
    }

    # extract targets & filenames_to_exclude
    subjects_info = dict()
    for subset_name in subsets_info.keys():
        print(f'Prosessing {subset_name}')
        subset_dir = os.path.join(data_dir, subset_name)
        for filename_idx, filename in enumerate(natsort.os_sorted(os.listdir(subset_dir))):
            subject_id = filename_idx + 1
            subject_name = os.path.splitext(filename)[0]
            if subject_id in subsets_info[subset_name]['ids_to_exclude']:
                subsets_info[subset_name]['filenames_to_exclude'].append(subject_name)
                print(f'Skipping {subset_name}/{subject_name}')
                continue

            subset_id = 1 if subset_name == 'data1' else 2
            seizures = utils.target_helper.get_seizures(dataset_id=subset_id, subject_id=subject_id)
            subjects_info[f'{subset_name}/{subject_name}'] = {
                'seizures': seizures,
            }
        print()
    print()

    # get eeg duration in seconds
    for subject_idx, subject_key in enumerate(subjects_info.keys()):
        print(f'\rReading progress {subject_idx + 1} {subject_key}', end='')

        extension = '.dat' if subject_key.startswith('data1/') else '.edf'
        raw_path = os.path.join(data_dir, f'{subject_key}{extension}')
        # if subject_key.startswith('data1/'):
        #     continue

        raw = eeg_reader.EEGReader.read_eeg(raw_path)
        duration_in_seconds = raw.times.max() - raw.times.min()
        subjects_info[subject_key]['duration_in_seconds'] = duration_in_seconds
    print()

    # data metainfo
    channels_info = utils.target_helper.get_channels_info()

    # build dataset info dict
    dataset_info = {
        'subjects_info': subjects_info,
        'subsets_info': subsets_info,
        'channels_info': channels_info,
    }
    dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
    with open(dataset_info_path, 'w') as fp:
        json.dump(dataset_info, fp)

    import pprint
    pprint.pprint(dataset_info)
