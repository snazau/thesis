import os

from autoreject import get_rejection_threshold
import mne
import matplotlib.pyplot as plt

import eeg_reader


def fit_ica(epochs, reject, time_step, ica_n_components=0.99, random_state=42, picks=None):
    # ica_n_components - Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(
        n_components=ica_n_components,
        random_state=random_state,
    )
    smth = ica.fit(
        epochs,
        reject=reject,
        tstep=time_step,
        picks=picks,
    )
    print(f'smth = {smth}')
    # ica.plot_components()
    # plt.show()

    # ica.plot_properties(epochs, picks=range(0, ica.n_components_), psd_args={'fmax': 30})
    # plt.show()

    return ica


def preprocess_raw(raw, save_ica_path=None, picks=None):
    # bandpass filter
    low_cut = 0.1
    hi_cut = 30
    raw_filtered = raw.copy().filter(low_cut, hi_cut, picks=picks)

    # filtering using ICA
    time_step = 1.0
    events_ica = mne.make_fixed_length_events(raw_filtered, duration=time_step)
    epochs_ica = mne.Epochs(
        raw_filtered,
        events_ica,
        tmin=0.0,
        tmax=time_step,
        baseline=None,
        preload=True
    )
    reject = get_rejection_threshold(epochs_ica)
    print(f'reject = {reject}')

    ica = fit_ica(epochs_ica, reject, time_step, picks=picks)

    ica_z_thresh = 1.96
    # correlation threshold - find_bads_eog computes correlations between each IC and channels that the researcher has designated as EOG (electro-oculogram) channels
    eog_indices, eog_scores = ica.find_bads_eog(
        raw_filtered,
        ch_name=['EEG Fp1', 'EEG F8'],
        threshold=ica_z_thresh
    )
    ica.exclude = eog_indices

    # ica.plot_scores(eog_scores)
    if save_ica_path is not None:
        save_ica_dir = os.path.dirname(save_ica_path)
        os.makedirs(save_ica_dir, exist_ok=True)
        ica.save(save_ica_path)
        # plt.savefig(save_ica_path.replace('.fif', '.png'))
    else:
        # plt.show()
        pass

    raw_filtered = ica.apply(raw_filtered)

    return raw_filtered


if __name__ == '__main__':
    # dataset_path = 'D:\\Study\\asp\\thesis\\implementation\\data\\data2'
    # dataset_sample_path = os.path.join(dataset_path, '003tl Anonim-20200831_040629-20211122_135924.edf')
    # raw = eeg_reader.EEGReader.read_eeg(dataset_sample_path, preload=True)
    # raw.drop_channels(['EEG ECG', 'Value MKR+'])
    #
    # print(raw.info)
    # print(raw.info['ch_names'])
    # # raw.plot_psd(fmax=64)
    # raw_filtered = preprocess_raw(raw)
    # # raw_filtered.plot_psd(fmax=64)
    # # raw_filtered.plot_psd(fmax=10)
    # # plt.show()
    #
    # raw.plot(duration=10)
    # plt.show()
    # raw_filtered.plot(duration=10)
    # plt.show()

    import json

    data_dir = './data'
    dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    for subject_key in dataset_info['subjects_info'].keys():
        # if subject_key not in ['data1/dataset16', 'data2/037tl Anonim-20201102_102725-20211123_003801']:
        #     continue

        # if subject_key != 'data1/dataset16':
        #     continue

        # if subject_key != 'data2/037tl Anonim-20201102_102725-20211123_003801':
        #     continue

        if 'data2' in subject_key:
            continue

        if 'data1' in subject_key:
            # picks = 'all'
            picks = None
            raw_path = os.path.join(data_dir, f'{subject_key}.dat')
            channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
        elif 'data2' in subject_key:
            picks = None
            raw_path = os.path.join(data_dir, f'{subject_key}.edf')
            channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
        else:
            raise NotImplementedError

        raw = eeg_reader.EEGReader.read_eeg(raw_path, preload=True)
        # channel_type = mne.io.pick.channel_type(raw.info, 12)
        # print('Channel #12 is of type:', channel_type)
        # raw.plot(duration=10)
        # plt.show()

        channels_num = len(raw.info['ch_names'])
        print(f'{subject_key} channels_num = {channels_num} channels = {raw.info["ch_names"]}')

        channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
        print(f"bads = {raw.info['bads']}")
        print(f"channels_to_drop = {channels_to_drop}")
        raw.drop_channels(channels_to_drop)

        ica_path = os.path.join(data_dir, 'ica_20230628', *subject_key.split('/'))
        ica_path += '.fif'
        print(f'Fitting {ica_path}')
        try:
            raw_filtered = preprocess_raw(raw, save_ica_path=ica_path, picks=picks)
            # raw_filtered = preprocess_raw(raw, save_ica_path=None, picks=picks)
            # raw_filtered.plot(duration=600)
            # plt.show()
        except Exception as e:
            import traceback
            print(f'Unable to process {subject_key}\nerror = {str(e)}\n{traceback.print_exc()}')
        print()
