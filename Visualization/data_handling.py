# This module comtains a data handling functions for reading CSV files saved from the main notebook.
# It is called in the visualization and slider modules to load data for plotting.
# It also contains a function to extract features from multiple data files for analysis.

# Imports
import csv
import glob
import os
import re
import numpy as np

def _parse_optional_float(value):
    if value == "":
        return np.nan
    return float(value)

# Function to load data from a CSV file
def load_data(filepath):
    file_path = filepath
    print(f"Reading data from {file_path}")
    with open(file_path, 'r', newline='') as csvfile:
        # Create a DictReader object
        csv_dict_reader = csv.DictReader(csvfile)
        
        t, tau, x, y, z, E, T, IL, s_1_array, s_2_array, Fitness = [], [], [], [], [], [], [], [], [], [], []
        
        for row in csv_dict_reader:
            if row['t'] == '':
                continue  # Skip rows with empty t

            t.append(float(row['t']))
            tau.append(_parse_optional_float(row['tau']))
            x.append(_parse_optional_float(row['x']))
            y.append(_parse_optional_float(row['y']))
            z.append(_parse_optional_float(row['z']))
            E.append(_parse_optional_float(row['E']))
            T.append(_parse_optional_float(row['T']))
            IL.append(_parse_optional_float(row['IL']))
            s_1_array.append(_parse_optional_float(row['s_1']))
            s_2_array.append(_parse_optional_float(row['s_2']))
            Fitness.append(_parse_optional_float(row['Fitness']))
    
    return t, tau, x, y, z, E, T, IL, s_1_array, s_2_array, Fitness


def parse_antigenicity_from_filename(filepath, offset=1.0):
    filename = os.path.basename(filepath)
    match = re.search(r'_(-?\d+(?:\.\d+)?)_\.csv$', filename)
    if match is None:
        raise ValueError(f"Could not extract antigenicity from filename: {filepath}")
    return float(match.group(1)) - offset


def sorted_sweep_files(pattern, offset=1.0):
    filepaths = glob.glob(pattern)
    filepaths.sort(key=lambda path: parse_antigenicity_from_filename(path, offset=offset))
    antigenicity_values = np.array(
        [parse_antigenicity_from_filename(path, offset=offset) for path in filepaths]
    )
    return filepaths, antigenicity_values


def _resolve_sweep_inputs(filepaths, antigenicity_values, offset=1.0):
    resolved_filepaths = list(filepaths)
    if not resolved_filepaths:
        return [], np.array([], dtype=float)

    try:
        parsed_c_vals = np.array(
            [
                parse_antigenicity_from_filename(path, offset=offset)
                for path in resolved_filepaths
            ],
            dtype=float,
        )
    except ValueError:
        parsed_c_vals = None

    if parsed_c_vals is not None:
        order = np.argsort(parsed_c_vals)
        resolved_filepaths = [resolved_filepaths[idx] for idx in order]
        return resolved_filepaths, parsed_c_vals[order]

    c_vals = np.asarray(antigenicity_values, dtype=float)
    if c_vals.shape[0] != len(resolved_filepaths):
        raise ValueError(
            "filepath and antigenicity_values must have the same length when "
            "antigenicity cannot be parsed from the filenames."
        )

    return resolved_filepaths, c_vals


def _load_numeric_series(filepath, value_column, min_length, time_columns=("tau", "t")):
    import pandas as pd

    df = pd.read_csv(filepath)
    if value_column not in df:
        raise KeyError(f"Column '{value_column}' not found in {filepath}")

    series = pd.to_numeric(df[value_column], errors="coerce").to_numpy()
    time_values = None

    for column in time_columns:
        if column not in df:
            continue

        candidate_time = pd.to_numeric(df[column], errors="coerce").to_numpy()
        mask = np.isfinite(series) & np.isfinite(candidate_time)
        if mask.sum() < min_length:
            continue

        return series[mask], candidate_time[mask]

    series = series[np.isfinite(series)]
    if series.size < min_length:
        raise ValueError(
            f"{filepath} does not contain enough numeric '{value_column}' values."
        )

    time_values = np.arange(series.size, dtype=float)
    return series, time_values


def _segment_frequency_axis(time_segment):
    if len(time_segment) < 2:
        raise ValueError("Need at least 2 time points to compute frequencies.")

    dt = float(np.median(np.diff(time_segment)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time axis must be strictly increasing.")

    return np.fft.fftfreq(len(time_segment), d=dt)


# function to extract features from data files
def extract_features(filepath, antigenicity_values):
    import pandas as pd

    filepath, c_vals = _resolve_sweep_inputs(filepath, antigenicity_values)

    amplitudes = []
    max_vals1 = []
    max_vals2 = []
    ave_vals1 = []
    ave_vals2 = []

    # find amplitudes for each file
    for idx, f in enumerate(filepath):
        df = pd.read_csv(f)

        y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
        y = y[np.isfinite(y)]
        if y.size < 1500:
            raise ValueError(
                f"{f} does not contain enough numeric y-values for feature extraction."
            )

        tail = y[-500:]
        center1 = y[500:1000]
        center2 = y[1000:1500]

        amp = tail.max() - tail.min()

        max_val1 = center1.max()
        max_val2 = center2.max()
        ave_val1 = center1.mean()
        ave_val2 = center2.mean()

        # store per run data
        amplitudes.append(amp)
        max_vals1.append(max_val1)
        max_vals2.append(max_val2)
        ave_vals1.append(ave_val1)
        ave_vals2.append(ave_val2)

    c_vals = np.asarray(c_vals, dtype=float)
    amplitudes_0 = np.array(amplitudes)
    max1_non = np.array(max_vals1)
    max2_non =  np.array(max_vals2)
    ave1_non = np.array(ave_vals1)
    ave2_non = np.array(ave_vals2)


    features = np.vstack((c_vals, amplitudes_0, max1_non, max2_non, ave1_non, ave2_non)).T
    return features, c_vals, amplitudes_0

def osc_features(filepath, antigenicity_values):
    """
    Extract FFT coefficients, amplitudes, and frequency axes for fixed y-segments.

    When the filepath names encode antigenicity, the files are sorted and labeled
    from the filenames so the returned c-values stay aligned even if the caller
    passes an unsorted glob list.
    """
    from scipy.fft import fft

    filepath, c_vals = _resolve_sweep_inputs(filepath, antigenicity_values)

    fft_initials = []
    fft_center1s = []
    fft_center2s = []
    fft_tails = []

    amp_initials = []
    amp_center1s = []
    amp_center2s = []
    amp_tails = []

    freq_initials = []
    freq_center1s = []
    freq_center2s = []
    freq_tails = []

    segment_length = 500
    min_length = 4 * segment_length

    for f in filepath:
        y, time_values = _load_numeric_series(
            f, value_column="y", min_length=min_length
        )

        initial = y[:segment_length]
        center1 = y[500:1000]
        center2 = y[1000:1500]
        tail = y[-segment_length:]

        time_initial = time_values[:segment_length]
        time_center1 = time_values[500:1000]
        time_center2 = time_values[1000:1500]
        time_tail = time_values[-segment_length:]

        # Fourier transforms
        fft_initial = fft(initial)
        fft_center1 = fft(center1)
        fft_center2 = fft(center2)
        fft_tail = fft(tail)

        # amplitudes of Fourier coefficients
        amp_initial = np.abs(fft_initial)
        amp_center1 = np.abs(fft_center1)
        amp_center2 = np.abs(fft_center2)
        amp_tail = np.abs(fft_tail)

        # frequency components
        freq_initial = _segment_frequency_axis(time_initial)
        freq_center1 = _segment_frequency_axis(time_center1)
        freq_center2 = _segment_frequency_axis(time_center2)
        freq_tail = _segment_frequency_axis(time_tail)

        # store FFT data
        fft_initials.append(fft_initial)
        fft_center1s.append(fft_center1)
        fft_center2s.append(fft_center2)
        fft_tails.append(fft_tail)

        # store amplitude data
        amp_initials.append(amp_initial)
        amp_center1s.append(amp_center1)
        amp_center2s.append(amp_center2)
        amp_tails.append(amp_tail)

        # store frequency data
        freq_initials.append(freq_initial)
        freq_center1s.append(freq_center1)
        freq_center2s.append(freq_center2)
        freq_tails.append(freq_tail)

    # convert lists to numpy arrays
    fft_initials = np.array(fft_initials)
    fft_center1s = np.array(fft_center1s)
    fft_center2s = np.array(fft_center2s)
    fft_tails = np.array(fft_tails)

    amp_initials = np.array(amp_initials)
    amp_center1s = np.array(amp_center1s)
    amp_center2s = np.array(amp_center2s)
    amp_tails = np.array(amp_tails)

    freq_initials = np.array(freq_initials)
    freq_center1s = np.array(freq_center1s)
    freq_center2s = np.array(freq_center2s)
    freq_tails = np.array(freq_tails)

    y_values = (fft_initials, fft_center1s, fft_center2s, fft_tails)
    amp_values = (amp_initials, amp_center1s, amp_center2s, amp_tails)
    freq_values = (freq_initials, freq_center1s, freq_center2s, freq_tails)


    return y_values, amp_values, freq_values, c_vals
