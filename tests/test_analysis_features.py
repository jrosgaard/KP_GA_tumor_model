"""Column, filename offset, and physical-frequency handling for KP and Dixon."""
import numpy as np
import pandas as pd
import pytest

from Visualization.data_handling import extract_features, osc_features


@pytest.fixture
def signal_frame():
    days = np.arange(2000)*2.0
    return pd.DataFrame(dict(t=days, tau=0.18*days,
        T=0.4 + 0.2*np.sin(2*np.pi*0.01*days), y=np.full(2000, 0.8)))


def test_dixon_features_use_tumor_column_and_direct_alpha(tmp_path, signal_frame):
    files = [tmp_path / f'simu_data_dixon_alpha_{alpha}.csv' for alpha in [10, 2]]
    for path in files:
        signal_frame.to_csv(path, index=False)
    features, alpha, amplitude = extract_features(files, [10, 2], value_column='T', offset=0)
    np.testing.assert_array_equal(alpha, [2, 10])
    np.testing.assert_array_equal(features[:, 0], alpha)
    np.testing.assert_allclose(features[:, -2:], 0.4)
    np.testing.assert_allclose(amplitude, np.ptp(signal_frame['T'].values[-500:]))


def test_dixon_fft_frequency_is_cycles_per_day_and_mean_removed(tmp_path, signal_frame):
    path = tmp_path / 'simu_data_dixon_alpha_715.72.csv'
    signal_frame.to_csv(path, index=False)
    _, amplitudes, frequencies, alpha = osc_features(
        [path], [715.72], value_column='T', offset=0,
        time_columns=('t',), detrend=True,
    )
    assert alpha[0] == 715.72
    for amp, freq in zip(amplitudes, frequencies):
        assert amp[0, 0] < 1e-10
        positive = freq[0] > 0
        maximum = np.argmax(amp[0, positive])
        assert freq[0, positive][maximum] == pytest.approx(0.01)
        assert 2*amp[0, positive][maximum]/500 == pytest.approx(0.2)


def test_kp_feature_defaults_keep_y_and_offset(tmp_path, signal_frame):
    path = tmp_path / 'simu_data_0001_c_1.0297_.csv'
    signal_frame.to_csv(path, index=False)
    features, c, amplitude = extract_features([path], [0.0297])
    assert c[0] == pytest.approx(0.0297)
    assert amplitude[0] == 0
    np.testing.assert_allclose(features[:, 1:], [[0, .8, .8, .8, .8]])
    _, amplitudes, frequencies, _ = osc_features([path], [0.0297])
    assert amplitudes[-1][0, 0] == pytest.approx(500*.8)
    assert frequencies[-1][0, 1] == pytest.approx(1/(500*2*.18))
