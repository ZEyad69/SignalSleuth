import numpy as np


def make_labels(length, start, end):
    #  return an array of `length` points,
    # 1 from `start` to `end` (end included), 0 everywhere else
    labels = np.zeros(length, dtype=int)
    labels[start:end+1] = 1
    return labels

def confusion_counts(labels, alerts):
    # TODO: return (tp, fp, fn)
    tp = np.sum((labels == 1) & (alerts == 1))
    fp = np.sum((labels == 0) & (alerts == 1))
    fn = np.sum((labels == 1) & (alerts == 0))
    return int(tp), int(fp), int(fn)
