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