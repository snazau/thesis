from numba import njit
import numpy as np
from math import floor
from matplotlib import pyplot as plt


@njit()
def fast_calibration_binning(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    """Computes bins of predicted vs actual events frequencies. Corresponds to sklearn's UNIFORM strategy."""

    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)

    min_val, max_val = 1.0, 0.0
    for predicted_prob in y_pred:
        if predicted_prob > max_val:
            max_val = predicted_prob
        elif predicted_prob < min_val:
            min_val = predicted_prob
    span = max_val - min_val
    multiplier = nbins / span
    for true_class, predicted_prob in zip(y_true, y_pred):
        ind = floor((predicted_prob - min_val) * multiplier)
        pockets_predicted[ind] += 1
        pockets_true[ind] += true_class

    idx = np.nonzero(pockets_predicted > 0)[0]

    hits = pockets_true[idx]
    freqs_predicted, freqs_true = min_val + (np.arange(nbins)[idx] + 0.5) * span / nbins, hits / pockets_predicted[idx]

    return freqs_predicted, freqs_true, hits


def show_calibration_plot(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    show_plots: bool = True,
    plot_file: str = "",
    plot_title: str = "",
    figsize: tuple = (12, 6),
):
    """Plots reliability digaram from the binned predictions."""
    fig = plt.figure(figsize=figsize)
    plt.scatter(freqs_predicted, freqs_true, marker="o", s=5000 * hits / hits.sum(), c=hits, label="Real")
    x_min, x_max = np.min(freqs_predicted), np.max(freqs_predicted)
    plt.plot([x_min, x_max], [x_min, x_max], "g--", label="Perfect")
    if plot_title:
        plt.title(plot_title)
    if plot_file:
        fig.savefig(plot_file)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def fast_calibration_report(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100, show_plots: bool = True, plot_file: str = "", figsize: tuple = (12, 6)):
    """Bins predictions, then computes regresison-like error metrics between desired and real binned probs."""

    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    diffs = np.abs((freqs_predicted - freqs_true))
    calibration_mae, calibration_std = np.mean(diffs), np.std(diffs)

    if plot_file or show_plots:
        show_calibration_plot(
            freqs_predicted=freqs_predicted,
            freqs_true=freqs_true,
            hits=hits,
            plot_title=f"Calibration MAE={calibration_mae:.4f} ± {calibration_std:.4f}",
            show_plots=show_plots,
            plot_file=plot_file,
            figsize=figsize,
        )

    return calibration_mae, calibration_std
