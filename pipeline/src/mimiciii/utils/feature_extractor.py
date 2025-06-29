import numpy as np
from scipy.stats import skew
import concurrent.futures

all_functions = [min, max, np.mean, np.std, skew, len]

functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
               (3, 10), (3, 25), (3, 50)]


def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)

def calculate(channel_data, period, sub_period, functions):
    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period, functions):
    global sub_periods
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)

def extract_features_wrapper(episode, period, functions):
    return extract_features_single_episode(episode, period, functions)

import warnings
warnings.filterwarnings(
    "ignore", 
    message="Precision loss occurred in moment calculation due to catastrophic cancellation"
)

def extract_features(data_raw, period, features):
    period = periods_map[period]
    functions = functions_map[features]
    # Create lists to pass for each episode.
    period_list = [period] * len(data_raw)
    functions_list = [functions] * len(data_raw)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        episodes = list(executor.map(extract_features_wrapper, data_raw, period_list, functions_list))
    return np.array(episodes)