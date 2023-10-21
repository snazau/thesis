import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os


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


if __name__ == '__main__':
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt


    def plot_colourline(x, y, c):
        col = cm.jet((c - np.min(c)) / (np.max(c) - np.min(c)))
        ax = plt.gca()
        for i in np.arange(len(x) - 1):
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=col[i])
        im = ax.scatter(x, y, c=c, s=0, cmap=cm.jet)
        return im


    import numpy as np
    import matplotlib.pyplot as plt

    n = 100
    x = 1. * np.arange(n)
    y = np.random.rand(n)
    prop = x ** 2

    fig = plt.figure(1, figsize=(5, 5))
    ax = fig.add_subplot(111)
    im = plot_colourline(x, y, prop)
    fig.colorbar(im)
    plt.show()
