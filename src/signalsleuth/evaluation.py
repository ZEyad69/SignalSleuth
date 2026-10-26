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
    adjusted = np.asarray(alerts).astype(bool)      # <-- changed

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
                adjusted[j] = True                  # <-- changed name
                j -= 1
            j = i + 1
            while j < len(labels) and labels[j] == 1:
                adjusted[j] = True                  # <-- changed name
                j += 1
    return adjusted                                 # <-- changed name

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