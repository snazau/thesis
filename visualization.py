import matplotlib.pyplot as plt
import numpy as np


def plot_hist(power_spectrum):
    n, bins, patches = plt.hist(power_spectrum.flatten(), bins=140)
    plt.show()


def plot_spectrum_channels(power_spectrum, time_idx_from, time_idx_to, channel_names=None):
    # power_spectrum.shape = (C, F, T)

    assert time_idx_from < time_idx_to

    fig, axes = plt.subplots(13, 2, figsize=(25, 5))
    for i in range(power_spectrum.shape[1]):
        temp = power_spectrum[0, i, :, time_idx_from:time_idx_to]
        axes[i % 13, i // 13].imshow((temp - temp.min()) / (temp.max() - temp.min()))
        axes[i % 13, i // 13].set_title(channel_names[i] if channel_names is not None else '')
    plt.show()


def plot_spectrum_averaged(power_spectrum, freqs):
    # power_spectrum.shape = (C, F, T)

    power_avg = np.mean(power_spectrum, axis=(0, 2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(freqs, power_avg, 'bo')
    axes[1].semilogy(freqs, power_avg, 'bo')
    plt.show()
