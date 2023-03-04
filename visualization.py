import cv2
import matplotlib.pyplot as plt
import numpy as np


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
        axes[i % 13, i // 13].imshow((temp - temp.min()) / (temp.max() - temp.min()))
        axes[i % 13, i // 13].set_title(channel_names[i] if channel_names is not None else '')
    plt.show()


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
