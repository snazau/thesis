import scipy.ndimage
import sklearn.metrics


def get_min_deviation_from_seizure(seizures, time_in_seconds):
    min_deviation = 1e9
    for seizure in seizures:
        if seizure['start'] <= time_in_seconds <= seizure['end']:
            return -1

        min_deviation = min(min_deviation, abs(time_in_seconds - seizure['start']), abs(time_in_seconds - seizure['end']))
    return min_deviation


def filter_predictions(probs, filter, k_size):
    assert filter in ['mean', 'median', None]

    if filter == 'mean':
        probs_filtered = scipy.ndimage.uniform_filter1d(probs, size=k_size)
    elif filter == 'median':
        probs_filtered = scipy.ndimage.median_filter(probs, size=k_size)
    elif filter is None:
        probs_filtered = probs.copy()
    else:
        raise NotImplementedError

    return probs_filtered


def calc_metrics(probs, labels, threshold):
    # metrics
    preds = probs > threshold

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_score = sklearn.metrics.f1_score(labels, preds)
        precision_score = sklearn.metrics.precision_score(labels, preds)
        recall_score = sklearn.metrics.recall_score(labels, preds)

    fp_mask = ((labels == 0) & (probs > threshold))
    fp_num = fp_mask.sum()

    fn_mask = ((labels == 1) & (probs <= threshold))
    fn_num = fn_mask.sum()

    tp_mask = ((labels == 1) & (probs > threshold))
    tp_num = tp_mask.sum()

    tn_mask = ((labels == 0) & (probs <= threshold))
    tn_num = tn_mask.sum()

    metric_dict = {
        'f1_score': f1_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'fp_num': fp_num,
        'tp_num': tp_num,
        'fn_num': fn_num,
        'tn_num': tn_num,
    }

    return metric_dict


def calc_segments_metrics(
        seizure_segments_true,
        normal_segments_true,
        seizure_segments_pred,
        normal_segments_pred,
        intersection_part_threshold,
):
    assert 0 < intersection_part_threshold <= 1

    seizures_true_used_mask = [0 for _ in range(len(seizure_segments_true))]

    tp_num, fp_num = 0, 0
    for seizure_segment_pred in seizure_segments_pred:
        find_tp = False
        for seizure_true_idx, seizure_segment_true in enumerate(seizure_segments_true):
            overlapping_distance = overlapping_segment(seizure_segment_true, seizure_segment_pred)
            true_distance = seizure_segment_true['end'] - seizure_segment_true['start']
            # if overlapping_distance / true_distance > intersection_part_threshold:
            if overlapping_distance / true_distance > intersection_part_threshold and seizures_true_used_mask[seizure_true_idx] == 0:
                tp_num += 1
                find_tp = True
                seizures_true_used_mask[seizure_true_idx] += 1
                break

        if not find_tp:
            fp_num += 1

    fn_num = seizures_true_used_mask.count(0)

    # tn_num = 0
    # for normal_segment_pred in normal_segments_pred:
    #     find_tn = False
    #     for normal_segment_true in normal_segments_true:
    #         overlapping_distance = overlapping_segment(normal_segment_true, normal_segment_pred)
    #         some_distance = min(normal_segment_true['end'] - normal_segment_true['start'], normal_segment_pred['end'] - normal_segment_pred['start'])
    #         if overlapping_distance / some_distance > intersection_part_threshold:
    #             tn_num += 1
    #             find_tn = True
    #             break

    precision_score = tp_num / (tp_num + fp_num) if (tp_num + fp_num) > 0 else 0
    recall_score = tp_num / (tp_num + fn_num) if (tp_num + fn_num) > 0 else 0
    f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0

    metric_dict = {
        'f1_score': f1_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'fp_num': fp_num,
        'tp_num': tp_num,
        'fn_num': fn_num,
        'tn_num': -1,
    }

    return metric_dict


def overlapping_segment(segment1, segment2):
    overlap_start = max(segment1['start'], segment2['start'])
    overlap_end = min(segment1['end'], segment2['end'])

    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    return 0
