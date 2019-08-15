import numpy as np
from .histogram import normalize


LI_LANDMARKS = "4.4408920985e-16 8.06305571158 15.5085721044 18.7007018006 21.5032879029 26.1413278906 29.9862059045 33.8384058795 38.1891334787 40.7217966068 44.0109152758 58.3906435207 100.0"
LI_LANDMARKS = np.array([float(n) for n in LI_LANDMARKS.split()])


def preprocess(data, padding):
    # data = pad(data, padding)
    data = standardize(data, masking_function=None)
    data = whiten(data)
    data = data.astype(np.float32)
    data = pad(data, padding)  # should I pad at the beginning instead?
    return data


def pad(data, padding):
    # Should I use this value for padding?
    value = data[0, 0, 0]
    return np.pad(data, padding, mode='constant', constant_values=value)


def crop(data, padding):
    p = padding
    return data[p:-p, p:-p, p:-p]


def standardize(data, landmarks=LI_LANDMARKS, masking_function='mean_plus'):
    masking_function = mean_plus if masking_function == 'mean_plus' else None
    return normalize(data, landmarks, masking_function=masking_function)


def whiten(data):
    mask_data = mean_plus(data)
    values = data[mask_data]
    mean, std = values.mean(), values.std()
    data -= mean
    data /= std
    return data


def mean_plus(data):
    return data > data.mean()
