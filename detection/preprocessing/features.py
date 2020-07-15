from os import path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def get_statistics(input, prefix, ignore_nan=False):
    """
    Get the statistical features of a list variable.

    :param input: list to calculate the statistics from
    :param prefix: prefix of the statistical feature names
    :param ignore_nan: remove all NaN values from `input` before calculating the statistics
    :return: `pd.Series` including all the statistical features
    """
    output = dict()
    if type(input) is not np.ndarray:
        input = np.array(input)
    if len(input) == 0:
        return pd.Series(output)
    if ignore_nan:
        input = input[~np.isnan(input)]
    output[f"{prefix}_min"] = np.min(input)
    output[f"{prefix}_max"] = np.max(input)
    output[f"{prefix}_mean"] = np.mean(input)
    output[f"{prefix}_std"] = np.std(input)
    output[f"{prefix}_perc25"] = np.percentile(input, 25)
    output[f"{prefix}_median"] = np.median(input)
    output[f"{prefix}_perc75"] = np.percentile(input, 75)
    output[f"{prefix}_perc99"] = np.percentile(input, 99)
    output[f"{prefix}_range"] = output[f"{prefix}_max"] - output[f"{prefix}_min"]
    output[f"{prefix}_kurtosis"] = kurtosis(input)
    output[f"{prefix}_skew"] = skew(input)

    return pd.Series(output)


def extract_prt_features(row, signal, rpeaks):
    """
    Extract the PRT features - the segment lengths and intervals between them.

    :param row: a `BaseDataset` row to calculate the features from
    :param signal: the raw ECG signal
    :param rpeaks: the R peaks detected by `nk.ecg_peaks`
    :return: `row` with the added features
    """
    _, segments = nk.ecg_delineate(signal, rpeaks, sampling_rate=row.Fs, method="dwt")

    for key in segments:
        segments[key] = np.array(segments[key]) / row.Fs

    P_length = segments['ECG_P_Offsets'] - segments['ECG_P_Onsets']
    R_length = segments['ECG_R_Offsets'] - segments['ECG_R_Onsets']
    T_length = segments['ECG_T_Offsets'] - segments['ECG_T_Onsets']

    PR_interval = segments['ECG_R_Onsets'] - segments['ECG_P_Onsets']
    RT_interval = segments['ECG_T_Onsets'] - segments['ECG_R_Onsets']

    features = [
        (P_length, 'P_LENGTH'),
        (R_length, 'R_LENGTH'),
        (T_length, 'T_LENGTH'),
        (PR_interval, 'PR_INTERVAL'),
        (RT_interval, 'RT_INTERVAL'),
    ]

    for feature, name in features:
        row = row.append(get_statistics(feature, name, ignore_nan=True))

    return row


def extract_rpeak_features(row, signal):
    """
    Extract the R peak features.

    :param row: a `BaseDataset` row to calculate the features from
    :param signal: the raw ECG signal
    :return: `row` with the added features
    """
    ecg_cleaned = nk.ecg_clean(signal, sampling_rate=row.Fs)

    peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=row.Fs)
    r_peaks_sec = np.where(peaks['ECG_R_Peaks'].to_numpy() == 1)[0].astype(np.float32)
    r_peaks_sec /= row.Fs  # get R-peak times in seconds

    num_peaks = len(r_peaks_sec)
    if num_peaks > 2:
        hrv = nk.hrv(peaks, sampling_rate=row.Fs, show=False).iloc[0]
        row = row.append(hrv)
    row['N_QRS'] = num_peaks

    rr = np.diff(r_peaks_sec)
    row = row.append(get_statistics(rr, 'RR'))
    row = row.append(get_statistics(signal, 'signal'))

    return row, info


def extract_qrs_correlation(row, signal, rpeaks):
    """
    Extract the QRS correlation features.

    :param row: row: a `BaseDataset` row to calculate the features from
    :param signal: the raw ECG signal
    :param rpeaks: the R peaks detected by `nk.ecg_peaks`
    :return: `row` with the added features
    """
    beat_lengths = np.diff(rpeaks['ECG_R_Peaks'])
    correlation_window = int(np.percentile(beat_lengths, 25) // 2)

    beat_windows = []
    for rpeak in rpeaks['ECG_R_Peaks']:
        rpeak = int(rpeak)
        if correlation_window <= rpeak < len(signal) - correlation_window:
            onset, offset = rpeak - correlation_window, rpeak + correlation_window
            beat_windows.append(signal[onset:offset])

    correlations = []
    for i in range(1, len(beat_windows)):
        curr_beat, prev_beat = beat_windows[i], beat_windows[i - 1]
        correlations.append(np.corrcoef(curr_beat, prev_beat)[1, 0])

    row = row.append(get_statistics(correlations, 'QRS_CORRELATION'))
    return row


def extract_row_features(row, dataset):
    """
    Extract all the features from a row.

    :param row: a `BaseDataset` row to calculate the features from
    :param dataset: a `BaseDataset`
    :return: `row` with the added features
    """
    signal = dataset.read_record(row.Record)

    try:
        row, rpeaks = extract_rpeak_features(row, signal)
        row = extract_prt_features(row, signal, rpeaks)
        row = extract_qrs_correlation(row, signal, rpeaks)
    except Exception as e:
        print(f'[ERROR] {e}')
    return row


def extract_features(dataset):
    """
    Extract all the features from a dataset.

    :param dataset: a `BaseDataset`
    :return: `pd.DataFrame` containing the extracted features
    """
    return dataset.data.progress_apply(extract_row_features, args=[dataset], axis=1)


def extract_features_datasets(datasets, output_dir):
    """
    Extract the features from all datasets and save the feature datasets to `output_dir`.

    :param datasets: the `BaseDataset` collection to extract the features from
    :param output_dir: output directory
    """
    for dataset in datasets:
        name = dataset.name()
        if dataset.features_exist():
            print(f'Dataset {name} has already extracted features. Skipping...')
            continue
        print(f'Extracting features for dataset {name}...')

        df_features = extract_features(dataset)
        dataset.data = df_features
        output_filename = path.join(output_dir, name + 'Features.pkl')
        df_features.to_pickle(output_filename, protocol=4)

        print(f'Extracted features saved to {output_filename}.')
