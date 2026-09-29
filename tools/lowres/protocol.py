"""Batch boundaries and epoch learning rates, independent of TensorFlow."""
import math


def batch_ranges(samples, batch=32, merge_tail=False):
    if samples <= 0 or batch < 2:
        raise ValueError('Positive sample count and batch >= 2 required')
    ends = list(range(batch, samples, batch)) + [samples]
    remainder = samples % batch
    if merge_tail and 0 < remainder < batch // 2 and len(ends) > 1:
        ends.pop(-2)
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def learning_rate(epoch, epochs, initial, schedule='constant', minimum=1e-6):
    if not 1 <= epoch <= epochs or not 0 < minimum <= initial:
        raise ValueError('Invalid epoch or learning-rate bounds')
    if schedule == 'constant':
        return initial
    if schedule != 'cosine' or epochs < 2:
        raise ValueError('Cosine requires at least two epochs')
    return minimum + (initial - minimum) * (1 + math.cos(math.pi * (epoch - 1) / (epochs - 1))) / 2
