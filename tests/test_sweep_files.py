"""Dixon alpha filenames and legacy KP offset filenames sort numerically."""
import numpy as np
import pytest

from Visualization.data_handling import (
    parse_antigenicity_from_filename,
    parse_sweep_value_from_filename,
    sorted_sweep_files,
)


@pytest.mark.parametrize("name,value", [
    ("simu_data_dixon_alpha_977.40857575757593.csv", 977.40857575757593),
    ("simu_data_dixon_alpha_1667.csv", 1667),
    ("simu_data_dixon_alpha_1e-05.csv", 1e-5),
    ("simu_data_dixon_alpha_-2.5E+02.csv", -250),
    ("simu_data_0001_c_1.0297_.csv", 1.0297),
])
def test_numeric_suffix(name, value):
    assert parse_sweep_value_from_filename(name) == value


def test_kp_offset_preserved():
    assert parse_antigenicity_from_filename("simu_data_0001_c_1.0297_.csv") == pytest.approx(0.0297)


@pytest.mark.parametrize("prefix,suffix,offset,expected", [
    ("simu_data_dixon_alpha_", ".csv", 0.0, [2, 10, 100]),
    ("simu_data_0001_c_", "_.csv", 1.0, [1, 9, 99]),
])
def test_numeric_sort_and_alignment(tmp_path, prefix, suffix, offset, expected):
    for value in (100, 2, 10):
        (tmp_path / f"{prefix}{value}{suffix}").touch()
    files, values = sorted_sweep_files(str(tmp_path / '*.csv'), offset=offset)
    np.testing.assert_array_equal(values, expected)
    assert [parse_sweep_value_from_filename(p, offset=offset) for p in files] == expected


def test_kp_sort_default_and_empty_glob(tmp_path):
    (tmp_path / 'simu_data_0001_c_1.0297_.csv').touch()
    _, values = sorted_sweep_files(str(tmp_path / '*.csv'))
    np.testing.assert_allclose(values, [0.0297])
    paths, values = sorted_sweep_files(str(tmp_path / 'missing_*.csv'))
    assert paths == []
    assert values.size == 0


def test_invalid_filename():
    with pytest.raises(ValueError, match='sweep value'):
        parse_sweep_value_from_filename('simu_data_dixon_alpha_unknown.csv')
