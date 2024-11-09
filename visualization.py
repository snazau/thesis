import gc
import os
import traceback

import cv2
import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import torch


def create_black_bar_w_text(bar_shape, text, text_location):
    bar = np.zeros(bar_shape, np.float64)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (1, 1, 1)
    thickness = 2
    lineType = 2

    cv2.putText(
        bar,
        text,
        text_location,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType
    )

    return bar


def plot_hist(power_spectrum):
    n, bins, patches = plt.hist(power_spectrum.flatten(), bins=140)
    plt.show()


def plot_spectrum_channels(power_spectrum, time_idx_from, time_idx_to, channel_names=None):
    # power_spectrum.shape = (C, F, T)

    assert time_idx_from < time_idx_to

    fig, axes = plt.subplots(13, 2, figsize=(25, 5))
    for i in range(power_spectrum.shape[0]):
        temp = power_spectrum[i, :, time_idx_from:time_idx_to]
        axes[i % 13, i // 13].imshow((temp - temp.min()) / (temp.max() - temp.min()), cmap=plt.cm.Reds)
        axes[i % 13, i // 13].set_title(channel_names[i] if channel_names is not None else '')
    plt.show()


def plot_spectrum_channel(power_spectrum, channel_idx, time_idx_from, time_idx_to, channel_names=None):
    # power_spectrum.shape = (C, F, T)

    assert time_idx_from < time_idx_to

    temp = power_spectrum[channel_idx, :, time_idx_from:time_idx_to]
    plt.imshow((temp - temp.min()) / (temp.max() - temp.min()), cmap='viridis')
    plt.show()


def visualize_raw_with_spectrum_data(power_spectrum, raw_signal, channel_names, save_dir=None, baseline_correction=False):
    # power_spectrum.shape = (C, F, T)
    # raw_signal.shape = (C, T)

    # if baseline_correction:
    #     eps = 1e-9
    #     power_spectrum_mean = np.mean(power_spectrum, axis=2, keepdims=True)
    #     power_spectrum = (power_spectrum - power_spectrum_mean) / (power_spectrum_mean + eps)

    for i in range(power_spectrum.shape[0]):
        fig, axes = plt.subplots(2, figsize=(15, 10))
        fig.suptitle(f'{channel_names[i]}')

        time_values = np.linspace(0, 10, raw_signal.shape[1])
        axes[0].plot(time_values, raw_signal[i])
        axes[0].set_title('raw_signal')
        axes[0].set_xlim([0, 10])
        axes[0].set_xlabel('Time (s)')
        # axes[0].set_ylabel('Freq. (Hz)')
        # axes[1].imshow(power_spectrum[i], cmap=plt.cm.Reds, interpolation='none', extent=[0, 10, 40, 0])

        # im = axes[1].imshow(power_spectrum[i], cmap=plt.cm.Reds, aspect="auto", vmin=power_spectrum[i].min(), vmax=power_spectrum[i].max())
        im = axes[1].imshow(power_spectrum[i], cmap='viridis', aspect="auto", vmin=power_spectrum[i].min(), vmax=power_spectrum[i].max())
        axes[1].invert_yaxis()
        axes[1].set_title('power_spectrum')
        axes[1].set_xlim([0, 1280])
        axes[1].set_xticks([i * 128 for i in range(0, 11)])
        axes[1].set_xticklabels([f'{i}' for i in range(0, 11)])
        axes[1].set_yticks([i for i in np.linspace(0, len(np.arange(1, 40.01, 0.1)), 8)])
        axes[1].set_yticklabels([f'{i:.02f}' for i in np.linspace(1, 40, 8)])
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Freq. (Hz)')

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('bottom', size='5%', pad=0.5)
        plt.colorbar(im, cax=cax, ax=axes[1], orientation='horizontal')
        # plt.show()

        if save_dir is not None:
            save_path = os.path.join(save_dir, f'{channel_names[i]}.png')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)

        fig.clear()
        plt.close(fig)


def visualize_raw_with_spectrum_data_v2(
        power_spectrum,
        raw_signal,
        channel_names,
        heatmap=None,
        preds=None,
        save_path=None,
        sfreq=128,
        time_shift=0,
        channels_to_show=None,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
        max_spectrum_value=None,
):
    # power_spectrum.shape = (C, F, T)
    # raw_signal.shape = (C, T)
    # heatmap.shape = (1, F, T)
    # preds.shape = (N_10, )

    # if baseline_correction:
    #     eps = 1e-9
    #     power_spectrum_mean = np.mean(power_spectrum, axis=2, keepdims=True)
    #     power_spectrum = (power_spectrum - power_spectrum_mean) / (power_spectrum_mean + eps)

    # fig = plt.figure()
    # plt.imshow(power_spectrum[0], cmap='viridis', aspect='auto', vmin=power_spectrum[0].min(), vmax=power_spectrum[0].max())
    # plt.show()
    # exit()

    if channels_to_show is None:
        channels_to_show = channel_names

    channel_dim, freq_dim, time_dim = power_spectrum.shape[:3]

    segment_num = int(time_dim / sfreq / 10)

    fig_width = min(80 * 10, int(30 * power_spectrum.shape[-1] / sfreq / 120))
    fig_height = 5 * len(channels_to_show)
    print(f'figsize = ({fig_width}, {fig_height})')
    fig, axes = plt.subplots(
        len(channels_to_show) + (len(channels_to_show) % 2),
        2,
        figsize=(fig_width, fig_height),
        # num=1,
        # clear=True,
    )
    fig.suptitle(f'Duration = {time_dim / sfreq:.2f}', fontsize=16)

    time_step = 10
    time_ticks_10sec = [time_tick for time_tick in range(int(time_shift), int(time_shift) + int(time_dim / sfreq) + 1, time_step)]
    fig_row_idx = -1
    # for idx in range(power_spectrum.shape[0]):
    for channel_to_show in channels_to_show:
        if channel_to_show == '__avg__':
            raw_signal_channel = np.mean(raw_signal, axis=0)
            power_spectrum_channel = np.mean(power_spectrum, axis=0)
            channel_name = '__avg__'
        elif channel_to_show == '__heatmap__' and heatmap is not None:
            raw_signal_channel = np.zeros_like(raw_signal[0])
            if preds is not None:
                for segment_10sec_idx in range(len(preds)):
                    raw_signal_channel[segment_10sec_idx * sfreq * 10:(segment_10sec_idx + 1) * sfreq * 10] = preds[segment_10sec_idx]
            power_spectrum_channel = heatmap[0]
            channel_name = '__heatmap__'
        else:
            idx = [channel_idx for channel_idx, channel_name in enumerate(channel_names) if channel_to_show in channel_name]
            assert len(idx) == 1
            idx = idx[0]

            raw_signal_channel = raw_signal[idx]
            power_spectrum_channel = power_spectrum[idx]
            channel_name = channel_names[idx]

        fig_row_idx += 1

        ax_column_idx = fig_row_idx % 2
        ax_row_idx = fig_row_idx - fig_row_idx % 2

        # plot 10 sec lines for raw signal
        for time_tick_10sec in time_ticks_10sec[1:-1]:
            axes[ax_row_idx, ax_column_idx].axvline(x=time_tick_10sec, color='blue', ls='--')

        # plot raw signal
        time_values = np.linspace(time_shift, time_shift + time_dim / sfreq, raw_signal.shape[1])
        axes[ax_row_idx, ax_column_idx].plot(time_values, raw_signal_channel)
        axes[ax_row_idx, ax_column_idx].set_title(channel_name if channel_name != '__heatmap__' else '__prob__', fontsize=16)
        axes[ax_row_idx, ax_column_idx].set_ylabel('raw_signal')
        axes[ax_row_idx, ax_column_idx].set_xlim([time_shift, time_shift + time_dim / sfreq])
        # axes[ax_row_idx, ax_column_idx].set_xlabel('Time (s)')
        # axes[ax_row_idx, ax_column_idx].set_ylabel('Freq. (Hz)')
        # axes[ax_row_idx, ax_column_idx].imshow(power_spectrum_channel, cmap=plt.cm.Reds, interpolation='none', extent=[0, 10, 40, 0])

        # plot power spectrum
        vmin = power_spectrum_channel.min()
        vmax = power_spectrum_channel.max() if max_spectrum_value is None else max_spectrum_value
        # print(f'channel_name = {channel_name} vmin = {vmin} vmax = {vmax}')
        # im = axes[ax_row_idx + 1, ax_column_idx].imshow(power_spectrum_channel, cmap=plt.cm.Reds, aspect="auto", vmin=power_spectrum_channel.min(), vmax=power_spectrum_channel.max())
        # im = axes[ax_row_idx + 1, ax_column_idx].imshow(power_spectrum_channel, cmap='viridis', aspect='auto', vmin=power_spectrum_channel.min(), vmax=power_spectrum_channel.max())
        # im = axes[ax_row_idx + 1, ax_column_idx].imshow(power_spectrum_channel, cmap='viridis', aspect='auto', vmin=power_spectrum_channel.min(), vmax=50 if channel_name != '__heatmap__' else power_spectrum_channel.max())
        # im = axes[ax_row_idx + 1, ax_column_idx].imshow(power_spectrum_channel, cmap='viridis', aspect='auto', vmin=power_spectrum_channel.min(), vmax=100 if channel_name != '__heatmap__' else power_spectrum_channel.max())
        im = axes[ax_row_idx + 1, ax_column_idx].imshow(power_spectrum_channel, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax if channel_name != '__heatmap__' else power_spectrum_channel.max())
        axes[ax_row_idx + 1, ax_column_idx].invert_yaxis()
        # axes[ax_row_idx + 1, ax_column_idx].set_title(channel_name, fontsize=16)
        # axes[ax_row_idx + 1, ax_column_idx].set_xlim([0, 0 + time_dim / sfreq])
        axes[ax_row_idx + 1, ax_column_idx].set_xticks([0 + i * time_dim / 10 for i in range(0, 10)])
        axes[ax_row_idx + 1, ax_column_idx].set_xticklabels([f'{time_shift + i * time_dim / 10 / sfreq}' for i in range(0, 10)], fontsize=16)
        axes[ax_row_idx + 1, ax_column_idx].set_yticks([i for i in np.linspace(0, len(np.arange(1, 40.01, 0.1)), 8)])
        axes[ax_row_idx + 1, ax_column_idx].set_yticklabels([f'{i:.02f}' for i in np.linspace(1, 40, 8)], fontsize=16)
        axes[ax_row_idx + 1, ax_column_idx].set_xlabel('Time (s)')
        axes[ax_row_idx + 1, ax_column_idx].set_ylabel('Freq. (Hz)')

        # add relative time ticks for spectrum
        ax_spectrum_twin = axes[ax_row_idx + 1, ax_column_idx].twiny()
        ax_spectrum_twin.set_xlim(axes[ax_row_idx + 1, ax_column_idx].get_xlim())
        ax_spectrum_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
        ax_spectrum_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=16)

        # divider = make_axes_locatable(axes[ax_row_idx + 1, ax_column_idx])
        # cax = divider.append_axes('bottom', size='5%', pad=5.5)
        # plt.colorbar(im, cax=cax, ax=axes[ax_row_idx + 1, ax_column_idx], orientation='horizontal')

        # add vertical lines
        if seizure_times_list is not None:
            for seizure_idx, seizure_time in enumerate(seizure_times_list):
                # print('seizure_times_colors', seizure_times_colors)
                # print('seizure_times_ls', seizure_times_ls)
                # print('len(seizure_times_list)', len(seizure_times_list))
                # print('seizure_idx', seizure_idx)
                seizure_line_color = seizure_times_colors[seizure_idx % len(seizure_times_list)]
                seizure_line_style = seizure_times_ls[seizure_idx % len(seizure_times_list)]
                seizure_line_width = plt.rcParams['lines.linewidth'] * 2 if seizure_line_color == 'red' else plt.rcParams['lines.linewidth']
                if time_shift <= (seizure_time['start'] + time_shift) <= (time_shift + time_dim / sfreq):
                    x = time_shift + seizure_time['start']
                    axes[ax_row_idx, ax_column_idx].axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')
                    # plt.text(x, seizure_idx, f's{x:.1f}', color=seizure_line_color, rotation=90, verticalalignment='center')

                    x = seizure_time['start'] * sfreq
                    axes[ax_row_idx + 1, ax_column_idx].axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')
                    # plt.text(x, seizure_idx, f's{x:.1f}', color=seizure_line_color, rotation=90, verticalalignment='center')
                if time_shift <= (seizure_time['end'] + time_shift) <= (time_shift + time_dim / sfreq):
                    x = time_shift + seizure_time['end']
                    axes[ax_row_idx, ax_column_idx].axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')
                    # plt.text(x, seizure_idx, f'e{x:.1f}', color=seizure_line_color, rotation=90, verticalalignment='center')

                    x = seizure_time['end'] * sfreq
                    axes[ax_row_idx + 1, ax_column_idx].axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')
                    # plt.text(x, seizure_idx, f'e{x:.1f}', color=seizure_line_color, rotation=90, verticalalignment='center')

        # plot 10 sec lines for spectrum
        for time_tick_10sec in time_ticks_10sec[1:-1]:
            # x = time_tick
            # axes[0].axvline(x=x, color='blue', ls='--')

            if time_shift <= time_tick_10sec <= (time_shift + time_dim / sfreq):
                x = (time_tick_10sec - time_shift) * sfreq
                axes[ax_row_idx + 1, ax_column_idx].axvline(x=x, color='#FFFFFF', ls='--')

    if save_path is not None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
        except Exception as e:
            print(f'Unable to save {save_path}')
            print(f'{traceback.format_exc()}')
    fig.clear()
    plt.close(fig)

    del fig, axes, time_values, im, raw_signal_channel, power_spectrum_channel

    gc.collect()


def visualize_feature_maps(
        feature_maps,
        original_resolution,
        save_dir=None,
        sfreq=128,
        time_shift=0,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
):
    # feature_maps.shape = (512, 13, 40)
    # original_resolution = (F, T)

    freq_dim, time_dim = original_resolution
    fm_channels, fm_freq_dim, fm_time_dim = feature_maps.shape[:3]
    segment_num = int(time_dim / sfreq / 10)

    time_step = 10
    time_ticks_10sec = [time_tick for time_tick in range(int(time_shift), int(time_shift) + int(time_dim / sfreq) + 1, time_step)]

    x_ticks = [0 + i * time_dim / 10 for i in range(0, 10)]
    x_ticks_labels = [f'{time_shift + i * time_dim / 10 / sfreq}' for i in range(0, 10)]
    y_ticks = [i for i in np.linspace(0, len(np.arange(1, 40.01, 0.1)), 8)]
    y_ticks_labels = [f'{i:.02f}' for i in np.linspace(1, 40, 8)]

    fig_height = 5
    fig_width = min(80 * 10, int(30 * time_dim / sfreq / 120))
    for channel_idx in range(fm_channels):
        # if channel_idx < 260:
        #     continue

        data = torch.from_numpy(feature_maps[channel_idx])
        print(f'#{channel_idx:03} feature_maps = {feature_maps.shape} data = {data.shape}')
        # data = cv2.resize(data, (time_dim, freq_dim), interpolation=cv2.INTER_LINEAR)
        data = torch.nn.functional.interpolate(
            data.unsqueeze(0).unsqueeze(0),
            (freq_dim, time_dim),
            mode="bilinear",
            align_corners=False,
        )
        data = data[0, 0].detach().cpu().numpy()
        print(f'#{channel_idx:03} feature_maps = {feature_maps.shape} data = {data.shape}')

        fig = plt.figure(figsize=(fig_width // 2, fig_height))
        im = plt.imshow(data, cmap='viridis', aspect='auto', vmin=data.min(), vmax=data.max())
        plt.gca().invert_yaxis()
        plt.title(f'fm #{channel_idx:03}', fontsize=16)
        plt.xticks(x_ticks, x_ticks_labels, fontsize=16)
        plt.yticks(y_ticks, y_ticks_labels, fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Freq. (Hz)')

        ax_spectrum_twin = plt.twiny()
        ax_spectrum_twin.set_xlim(plt.gca().get_xlim())
        ax_spectrum_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
        ax_spectrum_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=16)

        plt.colorbar()

        # add vertical lines
        if seizure_times_list is not None:
            for seizure_idx, seizure_time in enumerate(seizure_times_list):
                seizure_line_color = seizure_times_colors[seizure_idx % len(seizure_times_list)]
                seizure_line_style = seizure_times_ls[seizure_idx % len(seizure_times_list)]
                seizure_line_width = plt.rcParams['lines.linewidth'] * 2 if seizure_line_color == 'red' else plt.rcParams['lines.linewidth']
                if time_shift <= (seizure_time['start'] + time_shift) <= (time_shift + time_dim / sfreq):
                    x = seizure_time['start'] * sfreq
                    plt.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')
                if time_shift <= (seizure_time['end'] + time_shift) <= (time_shift + time_dim / sfreq):
                    x = seizure_time['end'] * sfreq
                    plt.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

        # plot 10 sec lines for spectrum
        for time_tick_10sec in time_ticks_10sec[1:-1]:
            if time_shift <= time_tick_10sec <= (time_shift + time_dim / sfreq):
                x = (time_tick_10sec - time_shift) * sfreq
                plt.axvline(x=x, color='#FFFFFF', ls='--')

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{channel_idx:03}.png')
        print(save_path)
        if save_path is not None:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
            except Exception as e:
                print(f'Unable to save {save_path}')
                print(f'{traceback.format_exc()}')

        fig.clear()
        plt.close(fig)


def visualize_raw_with_spectrum_data_v3(
        power_spectrum,
        raw_signal,
        heatmap,
        channel_names,
        channels_to_show,
        segment_of_int_idx_start,
        segment_of_int_idx_end,
        save_path=None,
        sfreq=128,
        time_shift=0,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
        max_spectrum_value=None,
):
    # power_spectrum.shape = (C, F, T)
    # raw_signal.shape = (C, T)
    # heatmap.shape = (1, F, T)

    channel_groups = {
        'frontal': {
            'channel_names': ['F3', 'Fz', 'F4'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['F3', 'Fz', 'F4']])
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
        'temporal': {
            'channel_names': ['T3', 'T4', 'T5', 'T6'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['T3', 'T4', 'T5', 'T6']])
            ],
        }
    }

    # import pprint
    # print('channel_groups')
    # pprint.pprint(channel_groups)
    # print('channel_names')
    # pprint.pprint(channel_names)

    if channels_to_show is None:
        channels_to_show = channel_names

    channel_dim, freq_dim, time_dim = power_spectrum.shape[:3]
    segment_num = int(time_dim / sfreq / 10)

    time_step = 10
    time_ticks_10sec = [time_tick for time_tick in range(int(time_shift), int(time_shift) + int(time_dim / sfreq) + 1, time_step)]

    fontsize = 22
    x_ticks = [0 + i * time_dim / 10 for i in range(0, 10)]
    x_ticks_labels = [f'{time_shift + i * time_dim / 10 / sfreq}' for i in range(0, 10)]
    y_ticks = [i for i in np.linspace(0, len(np.arange(1, 40.01, 0.1)), 8)]
    y_ticks_labels = [f'{i:.02f}' for i in np.linspace(1, 40, 8)]

    fig_height = 7 * (len(channel_groups) + 3)
    fig_width = min(80 * 10, int(30 * time_dim / sfreq / 120))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)

    segment_of_int_num = segment_of_int_idx_end - segment_of_int_idx_start + 1
    gs = GridSpec(3 + len(channel_groups), segment_of_int_num, figure=fig)
    ax_raw = fig.add_subplot(gs[0, :])
    ax_spectrum = fig.add_subplot(gs[1, :])
    ax_heatmap = fig.add_subplot(gs[2, :])

    channel_idx = [channel_idx for channel_idx, channel_name in enumerate(channel_names) if channels_to_show[0] in channel_name]
    assert len(channel_idx) == 1
    channel_idx = channel_idx[0]

    raw_signal_channel = raw_signal[channel_idx]  # (T, )
    power_spectrum_channel = power_spectrum[channel_idx]  # (F, T)
    channel_name = channel_names[channel_idx]

    # plot 10 sec lines for raw signal
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        ax_raw.axvline(x=time_tick_10sec, color='#000000', ls='--')

    # plot raw signal
    time_values = np.linspace(time_shift, time_shift + time_dim / sfreq, raw_signal.shape[1])
    ax_raw.plot(time_values, raw_signal_channel)
    ax_raw.set_title(channel_name, fontsize=fontsize)
    ax_raw.set_ylabel('raw_signal')
    ax_raw.set_xlim([time_shift, time_shift + time_dim / sfreq])

    # plot power spectrum
    vmin = power_spectrum_channel.min()
    vmax = power_spectrum_channel.max() if max_spectrum_value is None else max_spectrum_value
    im = ax_spectrum.imshow(power_spectrum_channel, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax_spectrum.invert_yaxis()
    ax_spectrum.set_xticks(x_ticks)
    ax_spectrum.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_yticks(y_ticks)
    ax_spectrum.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_xlabel('Time (s)')
    ax_spectrum.set_ylabel('Freq. (Hz)')

    # add relative time ticks for spectrum
    ax_spectrum_twin = ax_spectrum.twiny()
    ax_spectrum_twin.set_xlim(ax_spectrum.get_xlim())
    ax_spectrum_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    ax_spectrum_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # plot power spectrum
    vmin = heatmap[0].min()
    vmax = heatmap[0].max()
    im = ax_heatmap.imshow(heatmap[0], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax_heatmap.invert_yaxis()
    ax_heatmap.set_xticks(x_ticks)
    ax_heatmap.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_xlabel('Time (s)')
    ax_heatmap.set_ylabel('Freq. (Hz)')

    # add relative time ticks for spectrum
    heatmap_twin = ax_heatmap.twiny()
    heatmap_twin.set_xlim(ax_heatmap.get_xlim())
    heatmap_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    heatmap_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # add vertical lines
    if seizure_times_list is not None:
        for seizure_idx, seizure_time in enumerate(seizure_times_list):
            seizure_line_color = seizure_times_colors[seizure_idx % len(seizure_times_list)]
            seizure_line_style = seizure_times_ls[seizure_idx % len(seizure_times_list)]
            seizure_line_width = plt.rcParams['lines.linewidth'] * 4 if seizure_line_color == 'red' else plt.rcParams['lines.linewidth'] * 2

            if time_shift <= (seizure_time['start'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['start']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

            if time_shift <= (seizure_time['end'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['end']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

    # plot 10 sec lines for spectrum
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        if time_shift <= time_tick_10sec <= (time_shift + time_dim / sfreq):
            x = (time_tick_10sec - time_shift) * sfreq
            ax_spectrum.axvline(x=x, color='#FFFFFF', ls='--')
            ax_heatmap.axvline(x=x, color='#FFFFFF', ls='--')

    # calc avg spectrum vals
    baseline_segments_num = 6
    segment_baseline_idx_start = segment_of_int_idx_start - baseline_segments_num
    segment_baseline_idx_end = segment_of_int_idx_start - 1

    baseline_start_time = segment_baseline_idx_start * 10 + 0.25
    baseline_end_time = (segment_baseline_idx_end + 1) * 10 - 0.25

    if time_shift <= (time_shift + baseline_start_time) <= (time_shift + time_dim / sfreq):
        x = time_shift + baseline_start_time
        ax_raw.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_start_time * sfreq
        ax_spectrum.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_start_time * sfreq
        ax_heatmap.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

    if time_shift <= (baseline_end_time + time_shift) <= (time_shift + time_dim / sfreq):
        x = time_shift + baseline_end_time
        ax_raw.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_end_time * sfreq
        ax_spectrum.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_end_time * sfreq
        ax_heatmap.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

    power_spectrum = torch.from_numpy(power_spectrum)
    power_spectrum_segments = torch.split(power_spectrum, time_step * sfreq, dim=2)  # list of (C, F, 1280)
    power_spectrum_segments = torch.stack(power_spectrum_segments, dim=0)  # (N_10, C, F, 1280)
    power_spectrum_segments_avg_freq = torch.mean(power_spectrum_segments, dim=3, keepdim=True)  # (N_10, C, F, 1)

    heatmap = torch.from_numpy(heatmap)
    heatmap_segments = torch.split(heatmap, time_step * sfreq, dim=2)  # list of (1, F, 1280)
    heatmap_segments = torch.stack(heatmap_segments, dim=0)  # (N_10, 1, F, 1280)
    heatmap_segments_avg_freq = torch.mean(heatmap_segments, dim=3, keepdim=True)  # (N_10, 1, F, 1)

    baseline_power_spectrum_avg_freq = torch.mean(
        power_spectrum_segments_avg_freq[segment_baseline_idx_start:segment_baseline_idx_end + 1],
        dim=0,
        keepdim=True,
    )  # (1, C, F, 1)

    freqs = np.arange(1, 40.01, 0.1)
    for group_idx, group_name in enumerate(channel_groups.keys()):
        group_channel_names = channel_groups[group_name]['channel_names']
        group_channel_idxs = channel_groups[group_name]['channel_idxs']

        baseline_power_spectrum_avg_freq_avg_group = torch.mean(
            baseline_power_spectrum_avg_freq[:, group_channel_idxs],
            dim=1,
            keepdim=True,
        )  # (1, 1, F, 1)
        baseline_power_spectrum_avg_freq_avg_group = baseline_power_spectrum_avg_freq_avg_group.squeeze()  # (F, )
        baseline_power_spectrum_avg_freq_avg_group = baseline_power_spectrum_avg_freq_avg_group.detach().cpu().numpy()

        power_spectrum_segments_avg_freq_avg_group = torch.mean(
            power_spectrum_segments_avg_freq[:, group_channel_idxs],
            dim=1,
            keepdim=True,
        )  # (N_10, 1, F, 1)
        power_spectrum_segments_avg_freq_avg_group = power_spectrum_segments_avg_freq_avg_group.squeeze()  # (N_10, F)
        power_spectrum_segments_avg_freq_avg_group = power_spectrum_segments_avg_freq_avg_group.detach().cpu().numpy()

        for segment_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1):
            ax_segment_freq = fig.add_subplot(gs[3 + group_idx, segment_idx - segment_of_int_idx_start])
            ax_segment_freq.tick_params(axis='both', which='major', labelsize=18)
            ax_segment_freq.set_title(f'{segment_idx * time_step}-{(segment_idx + 1) * time_step} secs\nsegment #{segment_idx:02}', fontsize=fontsize)
            if (segment_idx - segment_of_int_idx_start) == 0:
                ax_segment_freq.set_ylabel(f'{group_name}\n({",".join(group_channel_names)})', fontsize=18)

            power_spectrum_segment_avg = power_spectrum_segments_avg_freq_avg_group[segment_idx]  # (F, )
            ax_segment_freq.plot(freqs, power_spectrum_segment_avg, c='red', label='Segment')

            ax_segment_freq.plot(freqs, baseline_power_spectrum_avg_freq_avg_group, c='blue', label='Baseline')

            heatmap_avg = heatmap_segments_avg_freq[segment_idx].squeeze().detach().cpu().numpy()  # (F, )
            ax_segment_imp = ax_segment_freq.twinx()  # instantiate a second Axes that shares the same x-axis
            ax_segment_imp.tick_params(axis='y', which='major', labelsize=18)
            ax_segment_imp.set_ylabel('Importance', color='gray', fontsize=18)
            ax_segment_imp.tick_params(axis='y', labelcolor='gray')
            ax_segment_imp.plot(freqs, heatmap_avg, color='gray')
            fig.tight_layout()  # otherwise the right y-label is slightly clipped

            # ax_segment_freq.plot(freqs, heatmap_avg, c='gray', label='Importance')

    # save figure tp disk
    if save_path is not None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
        except Exception as e:
            print(f'Unable to save {save_path}')
            print(f'{traceback.format_exc()}')
    fig.clear()
    plt.close(fig)

    del fig, gs, ax_raw, ax_spectrum, time_values, im, raw_signal_channel, power_spectrum_channel

    gc.collect()


def visualize_channel_importance_at_time(channel_importance, start_time_sec, channel_names, axes, time_step_sec=10, vmin=None, vmax=None):
    # channel_importance.shape = (C, )
    channel_importance = channel_importance[np.newaxis]  # (1, C)

    # create df from channel_importance
    df_importance_columns = [channel_name.replace('EEG ', '') for channel_name in channel_names]
    df_importance = pd.DataFrame(channel_importance, columns=df_importance_columns)
    df_importance = df_importance.rename(columns={'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'})
    df_importance = df_importance * 1e-6

    # add missing channels to df
    available_channel_names_for_montage = [
        channel_name.replace('EEG ', '').replace('T3', 'T7').replace('T4', 'T8').replace('T5', 'P7').replace('T6', 'P8')
        for channel_name in df_importance.columns
    ]
    montage = mne.channels.make_standard_montage('standard_1020')

    missing_channels = list(set(montage.ch_names) - set(available_channel_names_for_montage))
    df_importance[missing_channels] = 0
    df_importance = df_importance.reindex(columns=montage.ch_names)

    # create info object
    fake_info = mne.create_info(ch_names=montage.ch_names, sfreq=1.0 / time_step_sec, ch_types='eeg')
    evoked = mne.EvokedArray(df_importance.to_numpy().T, fake_info)
    evoked.set_montage(montage)
    evoked = evoked.drop_channels(missing_channels)

    # create mask for top-k important channels
    mask_params = dict(markersize=20, markerfacecolor="y")
    mask = np.zeros_like(df_importance.to_numpy().T)  # (94, N_10)

    topk_channel_idxs = torch.topk(torch.from_numpy(channel_importance), k=5, dim=1).indices.numpy()  # (N_10, k)
    channel_names_old_idx_to_new_idx = {
        old_idx: list(df_importance.columns).index(channel_name)
        for old_idx, (channel_name) in enumerate(available_channel_names_for_montage)
    }
    for segment_idx in range(topk_channel_idxs.shape[0]):
        for channel_idx_old in topk_channel_idxs[segment_idx]:
            channel_idx_new = channel_names_old_idx_to_new_idx[channel_idx_old]
            mask[channel_idx_new, segment_idx] = 1

    channel_idxs_to_delete_from_mask = [
        list(df_importance.columns).index(channel_name)
        for channel_name in missing_channels
    ]
    channel_idxs_to_delete_from_mask = np.array(channel_idxs_to_delete_from_mask)
    mask = np.delete(mask, channel_idxs_to_delete_from_mask, axis=0)

    # plot
    evoked_fig = evoked.plot_topomap(
        evoked.times,
        mask=mask,
        mask_params=mask_params,
        units='Importance',
        nrows='auto',
        ncols='auto',
        ch_type='eeg',
        time_format=f'{int(start_time_sec):d}-{int(start_time_sec) + 10:d} sec',
        show_names=True,
        axes=axes,
        colorbar=False,
        vlim=(vmin, vmax),
    )


def visualize_raw_with_spectrum_data_v4(
        power_spectrum,
        raw_signal,
        heatmap,
        channel_importances,
        channel_names,
        channels_to_show,
        segment_of_int_idx_start,
        segment_of_int_idx_end,
        save_path=None,
        sfreq=128,
        time_shift=0,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
        max_spectrum_value=None,
        min_importance_value=None,
):
    # power_spectrum.shape = (C, F, T)
    # raw_signal.shape = (C, T)
    # heatmap.shape = (1, F, T)

    # save_path = os.path.join(os.path.dirname(save_path), 'temp.npy')
    # save_dict = {
    #     'raw_signal': raw_signal,
    #     'power_spectrum': power_spectrum,
    #     'heatmap': heatmap,
    #     'channel_importances': channel_importances,
    #     'channel_names': channel_names,
    #     'channels_to_show': channels_to_show,
    #     'segment_of_int_idx_start': segment_of_int_idx_start,
    #     'segment_of_int_idx_end': segment_of_int_idx_end,
    #     'sfreq': sfreq,
    #     'time_shift': time_shift,
    #     'seizure_times_list': seizure_times_list,
    #     'seizure_times_colors': seizure_times_colors,
    #     'seizure_times_ls': seizure_times_ls,
    #     'max_spectrum_value': max_spectrum_value,
    # }
    # np.save(save_path, save_dict)
    # exit()

    if min_importance_value is not None:
        channel_importances = np.clip(channel_importances, a_min=min_importance_value, a_max=None)
        channel_importances = (channel_importances - channel_importances.min()) / (channel_importances.max() - channel_importances.min())

    import matplotlib
    matplotlib.rcParams.update({'font.size': 22, 'legend.fontsize': 22, 'lines.markersize': 22})

    if channel_names[-1] == '__heatmap__':
        channel_names = channel_names[:-1]

    freq_ranges = {
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 25),
    }

    if channels_to_show is None:
        channels_to_show = channel_names

    channel_dim, freq_dim, time_dim = power_spectrum.shape[:3]
    segment_num = int(time_dim / sfreq / 10)

    time_step = 10
    time_ticks_10sec = [time_tick for time_tick in range(int(time_shift), int(time_shift) + int(time_dim / sfreq) + 1, time_step)]

    fontsize = 22
    x_ticks = [0 + i * time_dim / 10 for i in range(0, 10)]
    x_ticks_labels = [f'{time_shift + i * time_dim / 10 / sfreq}' for i in range(0, 10)]
    y_ticks = [i for i in np.linspace(0, len(np.arange(1, 40.01, 0.1)), 8)]
    y_ticks_labels = [f'{i:.02f}' for i in np.linspace(1, 40, 8)]

    fig_height = 7 * (3 + len(freq_ranges) + 1)
    fig_width = min(80 * 10, int(30 * time_dim / sfreq / 120))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)

    segment_of_int_num = segment_of_int_idx_end - segment_of_int_idx_start + 1
    gs = GridSpec(3 + len(freq_ranges) + 1, segment_of_int_num, figure=fig)
    ax_raw = fig.add_subplot(gs[0, :])
    ax_spectrum = fig.add_subplot(gs[1, :])
    ax_heatmap = fig.add_subplot(gs[2, :])

    channel_idx = [channel_idx for channel_idx, channel_name in enumerate(channel_names) if channels_to_show[0] in channel_name]
    assert len(channel_idx) == 1
    channel_idx = channel_idx[0]

    raw_signal_channel = raw_signal[channel_idx]  # (T, )
    power_spectrum_channel = power_spectrum[channel_idx]  # (F, T)
    channel_name = channel_names[channel_idx]

    # plot 10 sec lines for raw signal
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        ax_raw.axvline(x=time_tick_10sec, color='#000000', ls='--')

    # plot raw signal
    time_values = np.linspace(time_shift, time_shift + time_dim / sfreq, raw_signal.shape[1])
    ax_raw.plot(time_values, raw_signal_channel)
    ax_raw.set_title(channel_name, fontsize=fontsize)
    ax_raw.set_ylabel('raw_signal')
    ax_raw.set_xlim([time_shift, time_shift + time_dim / sfreq])

    # plot power spectrum
    vmin = power_spectrum_channel.min()
    vmax = power_spectrum_channel.max() if max_spectrum_value is None else max_spectrum_value
    im = ax_spectrum.imshow(power_spectrum_channel, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax_spectrum.invert_yaxis()
    ax_spectrum.set_xticks(x_ticks)
    ax_spectrum.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_yticks(y_ticks)
    ax_spectrum.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_xlabel('Time (s)')
    ax_spectrum.set_ylabel('Freq. (Hz)')

    # add relative time ticks for spectrum
    ax_spectrum_twin = ax_spectrum.twiny()
    ax_spectrum_twin.set_xlim(ax_spectrum.get_xlim())
    ax_spectrum_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    ax_spectrum_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # plot power spectrum
    vmin = heatmap[0].min()
    vmax = heatmap[0].max()
    im = ax_heatmap.imshow(heatmap[0], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax_heatmap.invert_yaxis()
    ax_heatmap.set_xticks(x_ticks)
    ax_heatmap.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_xlabel('Time (s)')
    ax_heatmap.set_ylabel('Freq. (Hz)')

    # add relative time ticks for spectrum
    heatmap_twin = ax_heatmap.twiny()
    heatmap_twin.set_xlim(ax_heatmap.get_xlim())
    heatmap_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    heatmap_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # add vertical lines
    if seizure_times_list is not None:
        for seizure_idx, seizure_time in enumerate(seizure_times_list):
            seizure_line_color = seizure_times_colors[seizure_idx % len(seizure_times_list)]
            seizure_line_style = seizure_times_ls[seizure_idx % len(seizure_times_list)]
            seizure_line_width = plt.rcParams['lines.linewidth'] * 4 if seizure_line_color == 'red' else plt.rcParams['lines.linewidth'] * 2

            if time_shift <= (seizure_time['start'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['start']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

            if time_shift <= (seizure_time['end'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['end']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

    # plot 10 sec lines for spectrum
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        if time_shift <= time_tick_10sec <= (time_shift + time_dim / sfreq):
            x = (time_tick_10sec - time_shift) * sfreq
            ax_spectrum.axvline(x=x, color='#FFFFFF', ls='--')
            ax_heatmap.axvline(x=x, color='#FFFFFF', ls='--')

    # plot vertical lines for baseline
    baseline_segments_num = 6
    segment_baseline_idx_start = segment_of_int_idx_start - baseline_segments_num
    segment_baseline_idx_end = segment_of_int_idx_start - 1

    baseline_start_time = segment_baseline_idx_start * 10 + 0.25
    baseline_end_time = (segment_baseline_idx_end + 1) * 10 - 0.25

    if time_shift <= (time_shift + baseline_start_time) <= (time_shift + time_dim / sfreq):
        x = time_shift + baseline_start_time
        ax_raw.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_start_time * sfreq
        ax_spectrum.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_start_time * sfreq
        ax_heatmap.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

    if time_shift <= (baseline_end_time + time_shift) <= (time_shift + time_dim / sfreq):
        x = time_shift + baseline_end_time
        ax_raw.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_end_time * sfreq
        ax_spectrum.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_end_time * sfreq
        ax_heatmap.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

    # plot topogram with importances (occluded)
    for segment_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1):
        ax_segment = fig.add_subplot(gs[3, segment_idx - segment_of_int_idx_start])
        visualize_channel_importance_at_time(
            channel_importances[segment_idx],
            start_time_sec=segment_idx * 10,
            channel_names=channel_names,
            axes=ax_segment,
            time_step_sec=10,
            vmin=0,
            vmax=1,
        )
        ax_segment.tick_params(axis='both', which='major', labelsize=22)
        if (segment_idx - segment_of_int_idx_start) == 0:
            ax_segment.set_ylabel('Channel importance\n(occlusion)', fontsize=22)

    # plot topogram with importances (spectrum)
    power_spectrum = torch.from_numpy(power_spectrum)
    power_spectrum_segments = torch.split(power_spectrum, time_step * sfreq, dim=2)  # list of (C, F, 1280)
    power_spectrum_segments = torch.stack(power_spectrum_segments, dim=0)  # (N_10, C, F, 1280)

    freqs = np.arange(1, 40.01, 0.1)
    freqs = np.round(freqs, decimals=1)

    for freq_range_idx, (freq_range_name, freq_range) in enumerate(freq_ranges.items()):
        freq_range_min, freq_range_max = freq_range
        freq_range_start_idx = np.where(freqs == freq_range_min)[0][0]
        freq_range_end_idx = np.where(freqs == freq_range_max)[0][0]

        w_freq_range = torch.mean(power_spectrum_segments[:, :, freq_range_start_idx:freq_range_end_idx], dim=(2, 3))  # (N_10, C)
        w_freq_range_baseline = torch.mean(w_freq_range[segment_baseline_idx_start:segment_baseline_idx_end], dim=0, keepdim=True)  # (1, C)
        w_freq_range_normed = w_freq_range - w_freq_range_baseline  # (N_10, C)
        w_freq_range_normed = w_freq_range_normed.detach().cpu().numpy()

        for segment_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1):
            ax_segment = fig.add_subplot(gs[3 + freq_range_idx + 1, segment_idx - segment_of_int_idx_start])
            visualize_channel_importance_at_time(
                w_freq_range_normed[segment_idx],
                start_time_sec=segment_idx * 10,
                channel_names=channel_names,
                axes=ax_segment,
                time_step_sec=10,
            )
            ax_segment.tick_params(axis='both', which='major', labelsize=22)
            if (segment_idx - segment_of_int_idx_start) == 0:
                ax_segment.set_ylabel(f'Channel importance\n(w_{freq_range_name} {freq_range_min:d}-{freq_range_max:d}Hz)', fontsize=22)

    # save figure tp disk
    if save_path is not None:
        try:
            if min_importance_value is not None:
                suffix = f'_clip{min_importance_value:.2f}'
                save_path = save_path.replace('.png', f'{suffix}.png')

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
        except Exception as e:
            print(f'Unable to save {save_path}')
            print(f'{traceback.format_exc()}')
    fig.clear()
    plt.close(fig)

    del fig, gs, ax_raw, ax_spectrum, time_values, im, raw_signal_channel, power_spectrum_channel

    gc.collect()


def get_importance_matrix(freq_importance, freq_ranges, channel_importance, channel_groups, freqs):
    # freq_importance.shape = (F, )
    # channel_importance.shape = (C, )
    # freqs.shape = (F, )

    # cartesian product of avg GCAM and Channel Importance (GCAM x CI)
    freq_range_importances = np.zeros((len(freq_ranges), ), dtype=np.float32)
    channel_group_importances = np.zeros((len(channel_groups), ), dtype=np.float32)
    importance_matrix = np.zeros((len(freq_ranges), len(channel_groups)), dtype=np.float32)
    for freq_range_idx, freq_range_name in enumerate(freq_ranges.keys()):
        freq_range_min, freq_range_max = freq_ranges[freq_range_name]
        freq_range_start_idx = np.where(freqs == freq_range_min)[0][0]
        freq_range_end_idx = np.where(freqs == freq_range_max)[0][0]
        freq_range_importance = np.mean(freq_importance[freq_range_start_idx:freq_range_end_idx])
        freq_range_importances[freq_range_idx] = freq_range_importance

        for channel_group_idx, channel_group_name in enumerate(channel_groups.keys()):
            channel_group_idxs = channel_groups[channel_group_name]['channel_idxs']
            channel_group_importance = np.mean(channel_importance[channel_group_idxs])
            channel_group_importances[channel_group_idx] = channel_group_importance

            importance_matrix[freq_range_idx, channel_group_idx] = freq_range_importance * channel_group_importance

    # # Long tick names
    # freq_range_names = list(freq_ranges.keys())
    # channel_group_names = [
    #     f'{channel_group_name}\n({",".join(channel_groups[channel_group_name]["channel_names"])})'
    #     for channel_group_name in channel_groups.keys()
    # ]

    # Short tick names
    freq_range_names = [fr'$\{freq_range_name }$' for freq_range_name in freq_ranges.keys()]
    channel_group_names = [
        f'{"".join([name[0] for name in channel_group_name.split("-")])}'.upper()
        for channel_group_name in channel_groups.keys()
    ]

    # flip freq axes (make lower freqs at the bottom)
    freq_range_importances = np.flip(freq_range_importances, axis=0)
    importance_matrix = np.flip(importance_matrix, axis=0)
    freq_range_names.reverse()

    # print(freq_range_names)
    # print(freq_range_importances)
    # print(channel_group_names)
    # print(channel_group_importances)
    # print()

    return importance_matrix, freq_range_names, channel_group_names


def get_importance_matrices(heatmap, freq_ranges, channel_importances, channel_groups, segment_len_in_sec=10, sfreq=128):
    # heatmap.shape = (1, F, T)
    # channel_importances.shape = (N_10, C)

    segment_num = channel_importances.shape[0]

    heatmap = torch.from_numpy(heatmap)
    heatmap_segments = torch.split(heatmap, segment_len_in_sec * sfreq, dim=2)  # list of (1, F, 1280)
    heatmap_segments = torch.stack(heatmap_segments, dim=0)  # (N_10, 1, F, 1280)
    heatmap_segments_avg_freq = torch.mean(heatmap_segments, dim=3, keepdim=True)  # (N_10, 1, F, 1)

    freqs = np.arange(1, 40.01, 0.1)
    freqs = np.round(freqs, decimals=1)

    importance_matrices = list()
    for segment_idx in range(segment_num):
        heatmap_avg = heatmap_segments_avg_freq[segment_idx].squeeze().detach().cpu().numpy()  # (F, )
        importance_matrix, freq_range_names, channel_group_names = get_importance_matrix(
            heatmap_avg,
            freq_ranges,
            channel_importances[segment_idx],
            channel_groups,
            freqs,
        )
        importance_matrices.append(importance_matrix)

    importance_matrices = np.array(importance_matrices)  # (N_10, 5, 5)

    return importance_matrices, freq_range_names, channel_group_names


# def visualize_importance_matrix(freq_importance, freq_ranges, channel_importance, channel_groups, freqs, axis):
def visualize_importance_matrix(importance_matrix, freq_range_names, channel_group_names, axis, vmin, vmax):
    # importance_matrix, freq_range_names, channel_group_names = get_importance_matrix(
    #     freq_importance,
    #     freq_ranges,
    #     channel_importance,
    #     channel_groups,
    #     freqs,
    # )

    im = axis.imshow(importance_matrix, cmap='Reds', vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(channel_group_names)), labels=channel_group_names)
    axis.set_yticks(np.arange(len(freq_range_names)), labels=freq_range_names)

    for i in range(len(freq_range_names)):
        for j in range(len(channel_group_names)):
            text = axis.text(j, i, f'{importance_matrix[i, j]:.2f}', ha="center", va="center", color="black")


def visualize_raw_with_spectrum_data_v5(
        freq_ranges,
        channel_groups,
        power_spectrum,
        raw_signal,
        heatmap,
        channel_importances,
        channel_names,
        channels_to_show,
        segment_of_int_idx_start,
        segment_of_int_idx_end,
        save_path=None,
        sfreq=128,
        time_shift=0,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
        max_spectrum_value=None,
        min_importance_value=None,
        min_importance_matrix_value=None,
        max_importance_matrix_value=None,
):
    # power_spectrum.shape = (C, F, T)
    # raw_signal.shape = (C, T)
    # heatmap.shape = (1, F, T)
    # channel_importances.shape = (N_10, C)

    import matplotlib
    matplotlib.rcParams.update({'font.size': 22, 'legend.fontsize': 22, 'lines.markersize': 22})

    if channel_names[-1] == '__heatmap__':
        channel_names = channel_names[:-1]

    if channels_to_show is None:
        channels_to_show = channel_names

    channel_dim, freq_dim, time_dim = power_spectrum.shape[:3]
    segment_num = int(time_dim / sfreq / 10)

    time_step = 10
    time_ticks_10sec = [time_tick for time_tick in range(int(time_shift), int(time_shift) + int(time_dim / sfreq) + 1, time_step)]

    fontsize = 22
    x_ticks = [0 + i * time_dim / 10 for i in range(0, 10)]
    x_ticks_labels = [f'{time_shift + i * time_dim / 10 / sfreq}' for i in range(0, 10)]
    # y_ticks = [i for i in np.linspace(0, len(np.arange(1, 40.01, 0.1)), 8)]
    # y_ticks_labels = [f'{i:.02f}' for i in np.linspace(1, 40, 8)]
    y_ticks = [10 * freq - 10 for freq in [1, 4, 8, 14, 30, 40]]
    y_ticks_labels = [f'{freq:d}Hz' for freq in [1, 4, 8, 14, 30, 40]]

    fig_height = 7 * 5
    fig_width = min(80 * 10, int(30 * time_dim / sfreq / 120))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)

    segment_of_int_num = segment_of_int_idx_end - segment_of_int_idx_start + 1
    gs = GridSpec(5, segment_of_int_num, figure=fig)
    ax_raw = fig.add_subplot(gs[0, :])
    ax_spectrum = fig.add_subplot(gs[1, :])
    ax_heatmap = fig.add_subplot(gs[4, :])

    channel_idx = [channel_idx for channel_idx, channel_name in enumerate(channel_names) if channels_to_show[0] in channel_name]
    assert len(channel_idx) == 1
    channel_idx = channel_idx[0]

    raw_signal_channel = raw_signal[channel_idx]  # (T, )
    power_spectrum_channel = power_spectrum[channel_idx]  # (F, T)
    channel_name = channel_names[channel_idx]

    # plot 10 sec lines for raw signal
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        ax_raw.axvline(x=time_tick_10sec, color='#000000', ls='--')

    # plot raw signal
    time_values = np.linspace(time_shift, time_shift + time_dim / sfreq, raw_signal.shape[1])
    ax_raw.plot(time_values, raw_signal_channel)
    ax_raw.set_title(channel_name, fontsize=fontsize)
    ax_raw.set_ylabel('raw_signal')
    ax_raw.set_xlim([time_shift, time_shift + time_dim / sfreq])

    # plot power spectrum
    vmin = power_spectrum_channel.min()
    vmax = power_spectrum_channel.max() if max_spectrum_value is None else max_spectrum_value
    im = ax_spectrum.imshow(power_spectrum_channel, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    # im = ax_spectrum.imshow(power_spectrum_channel, cmap='Reds', aspect='auto', vmin=vmin, vmax=vmax)
    ax_spectrum.invert_yaxis()
    ax_spectrum.set_xticks(x_ticks)
    ax_spectrum.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_yticks(y_ticks)
    ax_spectrum.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_xlabel('Time (s)')
    ax_spectrum.set_ylabel('Freq. (Hz)')

    # add relative time ticks for spectrum
    ax_spectrum_twin = ax_spectrum.twiny()
    ax_spectrum_twin.set_xlim(ax_spectrum.get_xlim())
    ax_spectrum_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    ax_spectrum_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # plot power spectrum
    vmin = heatmap[0].min()
    vmax = heatmap[0].max()
    im = ax_heatmap.imshow(heatmap[0], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    # im = ax_heatmap.imshow(heatmap[0], cmap='Reds', aspect='auto', vmin=vmin, vmax=vmax)
    ax_heatmap.invert_yaxis()
    ax_heatmap.set_xticks(x_ticks)
    ax_heatmap.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_xlabel('Time (s)')
    ax_heatmap.set_ylabel('Freq. (Hz)')

    # add relative time ticks for spectrum
    heatmap_twin = ax_heatmap.twiny()
    heatmap_twin.set_xlim(ax_heatmap.get_xlim())
    heatmap_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    heatmap_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # add vertical lines
    if seizure_times_list is not None:
        for seizure_idx, seizure_time in enumerate(seizure_times_list):
            seizure_line_color = seizure_times_colors[seizure_idx % len(seizure_times_list)]
            seizure_line_style = seizure_times_ls[seizure_idx % len(seizure_times_list)]
            seizure_line_width = plt.rcParams['lines.linewidth'] * 4 if seizure_line_color == 'red' else plt.rcParams['lines.linewidth'] * 2

            if time_shift <= (seizure_time['start'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['start']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

            if time_shift <= (seizure_time['end'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['end']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

    # plot 10 sec lines for spectrum
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        if time_shift <= time_tick_10sec <= (time_shift + time_dim / sfreq):
            x = (time_tick_10sec - time_shift) * sfreq
            ax_spectrum.axvline(x=x, color='#FFFFFF', ls='--')
            ax_heatmap.axvline(x=x, color='#FFFFFF', ls='--')

    # plot band lines for spectrum
    for tick in y_ticks:
        ax_spectrum.axhline(y=tick, color='#FFFFFF', ls='--')
        ax_heatmap.axhline(y=tick, color='#FFFFFF', ls='--')

    # plot vertical lines for baseline
    baseline_segments_num = 6
    segment_baseline_idx_start = segment_of_int_idx_start - baseline_segments_num
    segment_baseline_idx_end = segment_of_int_idx_start - 1

    baseline_start_time = segment_baseline_idx_start * 10 + 0.25
    baseline_end_time = (segment_baseline_idx_end + 1) * 10 - 0.25

    if time_shift <= (time_shift + baseline_start_time) <= (time_shift + time_dim / sfreq):
        x = time_shift + baseline_start_time
        ax_raw.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_start_time * sfreq
        ax_spectrum.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_start_time * sfreq
        ax_heatmap.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

    if time_shift <= (baseline_end_time + time_shift) <= (time_shift + time_dim / sfreq):
        x = time_shift + baseline_end_time
        ax_raw.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_end_time * sfreq
        ax_spectrum.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

        x = baseline_end_time * sfreq
        ax_heatmap.axvline(x=x, color='blue', ls='solid', lw=plt.rcParams['lines.linewidth'] * 2)

    # plot topogram with importances (occluded)
    if min_importance_value is not None:
        channel_importances_clipped = np.clip(channel_importances, a_min=min_importance_value, a_max=None)
        channel_importances_clipped = (channel_importances_clipped - channel_importances_clipped.min()) / (channel_importances_clipped.max() - channel_importances_clipped.min())
    else:
        channel_importances_clipped = channel_importances.copy()

    for segment_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1):
        ax_segment = fig.add_subplot(gs[3, segment_idx - segment_of_int_idx_start])
        visualize_channel_importance_at_time(
            channel_importances_clipped[segment_idx],
            start_time_sec=segment_idx * 10,
            channel_names=channel_names,
            axes=ax_segment,
            time_step_sec=10,
            vmin=0,
            vmax=1,
        )
        ax_segment.tick_params(axis='both', which='major', labelsize=22)
        if (segment_idx - segment_of_int_idx_start) == 0:
            ax_segment.set_ylabel('Channel importance\n(occlusion)', fontsize=22)

    # plot importance matrices
    # heatmap = torch.from_numpy(heatmap)
    # heatmap_segments = torch.split(heatmap, time_step * sfreq, dim=2)  # list of (1, F, 1280)
    # heatmap_segments = torch.stack(heatmap_segments, dim=0)  # (N_10, 1, F, 1280)
    # heatmap_segments_avg_freq = torch.mean(heatmap_segments, dim=3, keepdim=True)  # (N_10, 1, F, 1)

    importance_matrices, freq_range_names, channel_group_names = get_importance_matrices(
        heatmap, freq_ranges, channel_importances, channel_groups, segment_len_in_sec=10, sfreq=sfreq,
    )  # (N_10, 5, 5), list with row names, list with col names

    if min_importance_matrix_value == 'min':
        min_importance_matrix_value = importance_matrices.min()

    if max_importance_matrix_value == 'max':
        max_importance_matrix_value = importance_matrices.max()

    # freqs = np.arange(1, 40.01, 0.1)
    # freqs = np.round(freqs, decimals=1)
    for segment_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1):
        ax_segment = fig.add_subplot(gs[2, segment_idx - segment_of_int_idx_start])
        ax_segment.set_title(f'{segment_idx * time_step}-{(segment_idx + 1) * time_step} secs', fontsize=22)
        if (segment_idx - segment_of_int_idx_start) == 0:
            ax_segment.set_ylabel('Importance matrices\nCartesian prod (GCAM x CI)', fontsize=22)

        # start zoom line
        if segment_idx == segment_of_int_idx_start:
            con = ConnectionPatch(
                xyA=((time_ticks_10sec[segment_idx] - time_shift) * sfreq, 0),
                xyB=(0, 1),
                coordsA="data",
                coordsB="axes fraction",
                axesA=ax_spectrum,
                axesB=ax_segment,
                arrowstyle="-",
                # shrinkB=5,
            )
            con.set_color([1.0, 0.0, 0.0])
            con.set_linewidth(4)
            con.set_in_layout(False)

            ax_segment.add_artist(con)

        # end zoom line
        if segment_idx == segment_of_int_idx_end:
            con = ConnectionPatch(
                xyA=((time_ticks_10sec[segment_idx + 1] - time_shift) * sfreq, 0),
                xyB=(1, 1),
                coordsA="data",
                coordsB="axes fraction",
                axesA=ax_spectrum,
                axesB=ax_segment,
                arrowstyle="-",
                # shrinkB=5,
            )
            con.set_color([1.0, 0.0, 0.0])
            con.set_linewidth(4)
            con.set_in_layout(False)

            ax_segment.add_artist(con)

        visualize_importance_matrix(
            importance_matrices[segment_idx],
            freq_range_names,
            channel_group_names,
            ax_segment,
            vmin=min_importance_matrix_value,
            vmax=max_importance_matrix_value,
        )

        # heatmap_avg = heatmap_segments_avg_freq[segment_idx].squeeze().detach().cpu().numpy()  # (F, )
        # visualize_importance_matrix(
        #     heatmap_avg,
        #     freq_ranges,
        #     channel_importances[segment_idx],
        #     channel_groups,
        #     freqs,
        #     ax_segment,
        # )

    # save figure tp disk
    if save_path is not None:
        try:
            if min_importance_value is not None:
                suffix = f'_clip{min_importance_value:.2f}'
                save_path = save_path.replace('.png', f'{suffix}.png')

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
        except Exception as e:
            print(f'Unable to save {save_path}')
            print(f'{traceback.format_exc()}')
    fig.clear()
    plt.close(fig)

    del fig, gs, ax_raw, ax_spectrum, time_values, im, raw_signal_channel, power_spectrum_channel

    gc.collect()


def visualize_spectrum_channels(power_spectrum, channel_names):
    # power_spectrum.shape = (C, F, T)

    # vmin = power_spectrum.min()
    # vmax = power_spectrum.max()
    # power_spectrum = (power_spectrum - vmin) / (vmax - vmin)
    # print(f'vmin = {vmin} vmax = {vmax}')

    row_images = list()
    column_images = list()
    for i in range(power_spectrum.shape[0]):
        power_spectrum_channel = power_spectrum[i]
        power_spectrum_channel = (power_spectrum_channel - power_spectrum_channel.min()) / (power_spectrum_channel.max() - power_spectrum_channel.min())
        black_bar = create_black_bar_w_text(bar_shape=(40, power_spectrum_channel.shape[-1]), text=channel_names[i], text_location=(10, 30))
        power_spectrum_channel_w_name = cv2.vconcat([black_bar, power_spectrum_channel])
        # plt.imshow(power_spectrum_channel_w_name)
        # plt.show()

        column_images.append(power_spectrum_channel_w_name)
        # print(i, len(column_images), power_spectrum_channel.shape)
        if (i + 1) % 3 == 0:
            column_image = cv2.hconcat(column_images)
            row_images.append(column_image)
            column_images = list()
        elif i == (power_spectrum.shape[0] - 1):
            for _ in range(3 - len(column_images)):
                column_images.append(np.zeros_like(column_images[0]))
            column_image = cv2.hconcat(column_images)
            row_images.append(column_image)
            column_images = list()
    merged_image = cv2.vconcat(row_images)
    # plt.imshow(merged_image, cmap='gray')
    # plt.show()
    return merged_image


def plot_spectrum_averaged(power_spectrum, freqs):
    # power_spectrum.shape = (C, F, T)

    power_avg = np.mean(power_spectrum, axis=(0, 2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].loglog(freqs, power_avg, 'bo')
    # axes[0].set_title('loglog')
    axes[0].plot(freqs, power_avg, 'bo')
    axes[0].set_title('linear')
    axes[1].semilogy(freqs, power_avg, 'bo')
    axes[1].set_title('semilogy')
    plt.show()


def visualize_raw(
        raw_signal,
        channel_names,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
        heatmap=None,
        time_start=0,
        save_path=None,
        trim_channels=True,
):
    # raw_signal.shape = (C, T)
    # heatmap.shape = (C, T)

    # fig, axes = plt.subplots(raw_signal.shape[0], figsize=(15, 10))
    # fig.suptitle(f'raw_signal')
    # for channel_idx in range(raw_signal.shape[0]):
    #
    #     time_values = np.linspace(0, raw_signal.shape[1] / 128, raw_signal.shape[1])
    #     axes[channel_idx].plot(time_values, raw_signal[channel_idx])
    #     axes[channel_idx].set_xlim([0, raw_signal.shape[1] / 128])
    #     axes[channel_idx].set_xlabel('Time (s)')
    #     # axes[0].set_ylabel('Freq. (Hz)')
    #     # axes[1].imshow(power_spectrum[channel_idx], cmap=plt.cm.Reds, interpolation='none', extent=[0, 10, 40, 0])
    #     # plt.show()
    #
    #     # if save_dir is not None:
    #     #     save_path = os.path.join(save_dir, f'{channel_names[channel_idx]}.png')
    #     #     os.makedirs(save_dir, exist_ok=True)
    #     #     plt.savefig(save_path, dpi=300)
    #
    #     # fig.clear()
    #     # plt.close(fig)
    # plt.show()

    # normalize heatmap
    if heatmap is not None:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        import matplotlib.cm as cm
        colors = cm.plasma(heatmap)

    from matplotlib import gridspec
    nrow, ncol = raw_signal.shape[0], 1
    fig = plt.figure(figsize=(24, 30))
    gs = gridspec.GridSpec(
        nrow,
        ncol,
        width_ratios=[1],
        wspace=0.0,
        hspace=0.0,
        top=0.95,
        bottom=0.05,
        left=0.17,
        right=0.845,
    )

    for channel_idx in range(raw_signal.shape[0]):
        time_end = time_start + raw_signal.shape[1] / 128
        time_values = np.linspace(time_start, time_end, raw_signal.shape[1])
        ax = plt.subplot(gs[channel_idx, 0])
        ax.set_xlim([time_start, time_end])
        # ax.set_xticklabels([])

        channel_name = channel_names[channel_idx][4:] if trim_channels else channel_names[channel_idx]
        ax.set_ylabel(channel_name)

        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,  # labels along the bottom edge are off
        )
        if channel_idx != (raw_signal.shape[0] - 1):
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,  # labels along the bottom edge are off
            )
        else:
            ax.set_xlabel('Time (s)')

        if seizure_times_list is not None:
            for seizure_idx, seizure_time in enumerate(seizure_times_list):
                seizure_line_color = seizure_times_colors[seizure_idx % len(seizure_times_list)]
                seizure_line_style = seizure_times_ls[seizure_idx % len(seizure_times_list)]
                if time_start <= seizure_time['start'] <= time_end:
                    ax.axvline(x=seizure_time['start'], color=seizure_line_color, ls=seizure_line_style, label=f'Seizure {seizure_idx:02} start')
                if time_start <= seizure_time['end'] <= time_end:
                    ax.axvline(x=seizure_time['end'], color=seizure_line_color, ls=seizure_line_style, label=f'Seizure {seizure_idx:02} end')

        if heatmap is not None:
            for time_idx in np.arange(len(time_values) - 1):
                im = ax.plot(
                    [time_values[time_idx], time_values[time_idx + 1]],
                    [raw_signal[channel_idx][time_idx], raw_signal[channel_idx][time_idx + 1]],
                    c=colors[channel_idx][time_idx],
                )
        else:
            ax.plot(time_values, raw_signal[channel_idx])

    if heatmap is not None:
        fig.colorbar(cm.ScalarMappable(norm=None, cmap=cm.get_cmap("plasma")), ax=fig.axes)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.clf()


def visualize_channel_importance(channel_importance, start_time_sec, channel_names, time_step_sec=10, save_path=None):
    # channel_importance.shape = (N_10, 25)
    # start_time_sec - start time in seconds
    # channel_names - ['EEG Fp1', 'EEG Fp2', ...]

    # create df from channel_importance
    df_importance_columns = [channel_name.replace('EEG ', '') for channel_name in channel_names]
    df_importance = pd.DataFrame(channel_importance, columns=df_importance_columns)
    df_importance = df_importance.rename(columns={'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'})
    df_importance = df_importance * 1e-6

    # add missing channels to df
    available_channel_names_for_montage = [
        channel_name.replace('EEG ', '').replace('T3', 'T7').replace('T4', 'T8').replace('T5', 'P7').replace('T6', 'P8')
        for channel_name in df_importance.columns
    ]
    montage = mne.channels.make_standard_montage('standard_1020')

    missing_channels = list(set(montage.ch_names) - set(available_channel_names_for_montage))
    df_importance[missing_channels] = 0
    df_importance = df_importance.reindex(columns=montage.ch_names)

    # create info object
    fake_info = mne.create_info(ch_names=montage.ch_names, sfreq=1.0 / time_step_sec, ch_types='eeg')
    evoked = mne.EvokedArray(df_importance.to_numpy().T, fake_info)
    evoked.set_montage(montage)
    evoked = evoked.drop_channels(missing_channels)

    # create mask for top-k important channels
    mask_params = dict(markersize=10, markerfacecolor="y")
    mask = np.zeros_like(df_importance.to_numpy().T)  # (94, N_10)

    topk_channel_idxs = torch.topk(torch.from_numpy(channel_importance), k=5, dim=1).indices.numpy()  # (N_10, k)
    channel_names_old_idx_to_new_idx = {
        old_idx: list(df_importance.columns).index(channel_name)
        for old_idx, (channel_name) in enumerate(available_channel_names_for_montage)
    }
    for segment_idx in range(topk_channel_idxs.shape[0]):
        for channel_idx_old in topk_channel_idxs[segment_idx]:
            channel_idx_new = channel_names_old_idx_to_new_idx[channel_idx_old]
            mask[channel_idx_new, segment_idx] = 1

    channel_idxs_to_delete_from_mask = [
        list(df_importance.columns).index(channel_name)
        for channel_name in missing_channels
    ]
    channel_idxs_to_delete_from_mask = np.array(channel_idxs_to_delete_from_mask)
    mask = np.delete(mask, channel_idxs_to_delete_from_mask, axis=0)

    # plot
    evoked_fig = evoked.plot_topomap(
        evoked.times,
        mask=mask,
        mask_params=mask_params,
        units='Importance',
        nrows='auto',
        ncols=7,
        ch_type='eeg',
        time_format=f'{int(start_time_sec):d}+%d sec',
        show_names=True,
    )
    if save_path is not None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=1)
        except Exception as e:
            print(f'Unable to save {save_path}')
            print(f'{traceback.format_exc()}')
    evoked_fig.clear()
    plt.close(evoked_fig)
    gc.collect()

    # plt.clf()


if __name__ == '__main__':
    # import matplotlib.cm as cm
    # import matplotlib.pyplot as plt
    #
    #
    # def plot_colourline(x, y, c):
    #     col = cm.jet((c - np.min(c)) / (np.max(c) - np.min(c)))
    #     ax = plt.gca()
    #     for i in np.arange(len(x) - 1):
    #         ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=col[i])
    #     im = ax.scatter(x, y, c=c, s=0, cmap=cm.jet)
    #     return im
    #
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # n = 100
    # x = 1. * np.arange(n)
    # y = np.random.rand(n)
    # prop = x ** 2
    #
    # fig = plt.figure(1, figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # im = plot_colourline(x, y, prop)
    # fig.colorbar(im)
    # plt.show()

    # import eeg_reader
    # import datasets.datasets_static
    #
    # seed = 8
    # np.random.seed(seed)
    #
    # data_dir = r'D:\Study\asp\thesis\implementation\data'
    # subject_name = 'data2/021tl Anonim-20201223_085255-20211122_172126.edf'
    # subject_eeg_path = os.path.join(data_dir, subject_name)
    # raw = eeg_reader.EEGReader.read_eeg(subject_eeg_path, preload=True)
    # datasets.datasets_static.drop_unused_channels(subject_eeg_path, raw)
    # channel_names = raw.info['ch_names']
    #
    # channel_name_to_idx = {
    #     channel_name.replace('EEG ', ''): channel_idx
    #     for channel_idx, channel_name in enumerate(channel_names)
    # }
    #
    # channel_importance = np.random.rand(31, 25)
    # visualize_channel_importance(
    #     channel_importance,
    #     13500,
    #     channel_names,
    #     time_step_sec=10,
    #     save_path=r'D:\Study\asp\thesis\implementation\scripts\interpretation\visualize_channel_importance_test.png',
    # )

    data_path = r'D:\Study\asp\thesis\implementation\experiments\20231213_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_OCSVM_positive_only_16excluded\vis_20241031\temp.npy'
    data = np.load(data_path, allow_pickle=True)
    data_dict = data.item()

    save_path = r'D:\Study\asp\thesis\implementation\experiments\20231213_EEGResNet18Spectrum_Default_SpecTimeFlipEEGFlipAug_meanstd_norm_Stage2_OCSVM_positive_only_16excluded\vis_20241031\temp_v5.png'
    # visualize_raw_with_spectrum_data_v4(**data_dict, save_path=save_path, min_importance_value=0.75)

    channel_names = data_dict['channel_names']
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

    visualize_raw_with_spectrum_data_v5(
        freq_ranges=freq_ranges,
        channel_groups=channel_groups,
        **data_dict,
        save_path=save_path,
        min_importance_value=0.75,
        min_importance_matrix_value='min',
        max_importance_matrix_value='max',
    )
