import numpy as np


def make_labels(length, start, end):
    #  return an array of `length` points,
    # 1 from `start` to `end` (end included), 0 everywhere else
    labels = np.zeros(length, dtype=int)
    labels[start:end+1] = 1
    return labels

def confusion_counts(labels, alerts):
    #  return (tp, fp, fn)
    tp = np.sum((labels == 1) & (alerts == 1))
    fp = np.sum((labels == 0) & (alerts == 1))
    fn = np.sum((labels == 1) & (alerts == 0))
    return int(tp), int(fp), int(fn)

def precision_recall_f1(tp, fp, fn):
    #  return (precision, recall, f1)
    precision= tp / (tp + fp) if (tp + fp) > 0 else 0
    recall= tp / (tp + fn) if (tp + fn) > 0 else 0
    f1= 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1    

def point_adjust(labels, alerts):
    # start from the original alerts, so false alarms outside the anomaly stay as they are
    # astype(bool) also makes a NEW array, so the input is never changed
    adjusted = np.asarray(alerts).astype(bool)      

    in_anomaly = False
    for i in range(len(labels)):
        if labels[i] == 1:
            in_anomaly = True
        elif in_anomaly and labels[i] == 0:
            in_anomaly = False
        if in_anomaly and alerts[i]:
            # mark all points in the current anomaly as alerted
            j = i
            while j >= 0 and labels[j] == 1:
                adjusted[j] = True                  
                j -= 1
            j = i + 1
            while j < len(labels) and labels[j] == 1:
                adjusted[j] = True                  
                j += 1
    return adjusted                                 

def count_segments(mask):
    #  return how many separate groups of consecutive True values
    segments = 0
    in_segment = False
    for val in mask:
        if val and not in_segment:
            segments += 1
            in_segment = True
        elif not val:
            in_segment = False
    return segments

def evaluate_alerts(labels, alerts):
    # evaluate ONE series: the alerts are already decided (yes/no per point),
    # so this function never sees the scores and never picks a threshold

    # make both inputs True/False, so "~" and "&" work the way I expect
    labels = np.asarray(labels).astype(bool)          
    alerts = np.asarray(alerts).astype(bool)          

    # both arrays must describe the same points
    if labels.shape != alerts.shape:                  
        raise ValueError("labels and alerts must have the same length")

    # point-level metrics (no adjustment)
    tp, fp, fn = confusion_counts(labels, alerts)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    # same metrics after point adjustment (PA)
    adjusted_alerts = point_adjust(labels, alerts)
    pa_tp, pa_fp, pa_fn = confusion_counts(labels, adjusted_alerts)
    pa_precision, pa_recall, pa_f1 = precision_recall_f1(pa_tp, pa_fp, pa_fn)

    # false alarms = alerts where there is no real anomaly
    # count how many SEPARATE groups they form (events, not points)
    false_alarm_events = count_segments(~labels & alerts)

    # did the detector put at least one alert inside the real anomaly?
    anomaly_detected = bool(np.any(labels & alerts))  

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pa_precision": pa_precision,
        "pa_recall": pa_recall,
        "pa_f1": pa_f1,
        "false_alarm_events": false_alarm_events,
        "anomaly_detected": anomaly_detected,
    }