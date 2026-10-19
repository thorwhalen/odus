"""Tests for odus.raw_formats: reading other formats and layouts into trajectories."""

import os
import shutil

import pandas as pd
import pytest

from odus import data_dir
from odus.dacc import DfStore
from odus.raw_formats import (
    YEAR_OF_AGE,
    SurveyLayout,
    read_raw_table,
    trajectory_from_long,
    trajectory_from_table,
)

SURVEYS = os.path.join(data_dir, "surveys")


def _store(folder, ext):
    return DfStore(str(folder) + os.sep + "{}" + ext)


def test_a_csv_or_tsv_export_gives_the_same_trajectory(tmp_path):
    raw = read_raw_table(os.path.join(SURVEYS, "A01.xlsx"))
    raw.to_csv(tmp_path / "A01.csv", header=False, index=False)
    raw.to_csv(tmp_path / "A01.tsv", header=False, index=False, sep="\t")
    shutil.copy(os.path.join(SURVEYS, "A01.xlsx"), tmp_path / "A01.xlsx")

    from_xlsx = _store(tmp_path, ".xlsx")["A01.xlsx"]
    for ext in (".csv", ".tsv"):
        other = _store(tmp_path, ext)["A01" + ext]
        pd.testing.assert_frame_equal(other, from_xlsx)
        assert other.attrs[YEAR_OF_AGE] == from_xlsx.attrs[YEAR_OF_AGE]


def test_another_layout():
    raw = pd.DataFrame(
        [
            ["AGE", 30, 31, 32],
            ["notes", "x", None, None],
            ["HEROIN", 1, None, 1],
            ["WORK", "0", "1", 1.0],
        ]
    )
    layout = SurveyLayout(year_row=None, age_row=0, data_start_row=2, empty_value=0)
    t = trajectory_from_table(raw, layout=layout)
    assert t.index.tolist() == [30, 31, 32]
    assert t.columns.tolist() == ["HEROIN", "WORK"]
    assert t.values.tolist() == [[1, 0], [0, 1], [1, 1]]
    assert YEAR_OF_AGE not in t.attrs


def test_a_store_with_another_layout(tmp_path):
    pd.DataFrame([["YEAR", 1990, 1991], ["AGE", 20, 21], ["HEROIN", 0, 1]]).to_csv(
        tmp_path / "R1.csv", header=False, index=False
    )
    store = _store(tmp_path, ".csv")
    store.layout = SurveyLayout(year_row=0, age_row=1, data_start_row=2)
    t = store["R1.csv"]
    assert t["HEROIN"].tolist() == [0, 1]
    assert t.attrs[YEAR_OF_AGE] == {20: 1990, 21: 1991}


def test_informative_errors(tmp_path):
    raw = pd.DataFrame([["ID", "R1"], ["Y", 1990], ["A", 20], ["HEROIN", None]])
    with pytest.raises(ValueError, match="'HEROIN' at age 20 is empty"):
        trajectory_from_table(raw)
    raw.iat[3, 1] = "yes"
    with pytest.raises(ValueError, match="'HEROIN' at row 4 should be a number"):
        trajectory_from_table(raw)
    with pytest.raises(ValueError, match=r"No reader for '.json'.*readers="):
        read_raw_table(b"{}", ext=".json")
    with pytest.raises(ValueError, match="ext="):
        read_raw_table(b"a,b")


def test_more_readers(tmp_path):
    raw = pd.DataFrame([["ID", "R1"], ["Y", 1990], ["A", 20], ["HEROIN", 1]])
    raw.to_json(tmp_path / "R1.json")
    store = _store(tmp_path, ".json")
    store.readers = {".json": pd.read_json}
    assert store["R1.json"]["HEROIN"].tolist() == [1]


def test_long_tables():
    long = pd.DataFrame(
        {"age": [21, 20, 21], "var": ["HEROIN", "WORK", "WORK"], "val": [1, 1, 0]}
    )
    t = trajectory_from_long(long, variable="var", value="val")
    assert t.index.tolist() == [20, 21]
    assert t.to_dict() == {"HEROIN": {20: 0, 21: 1}, "WORK": {20: 1, 21: 0}}
    with pytest.raises(KeyError, match="no columns"):
        trajectory_from_long(long)
    with pytest.raises(ValueError, match="more than once"):
        trajectory_from_long(pd.concat([long, long]), variable="var", value="val")


def test_diagnosis_follows_the_layout(tmp_path):
    from openpyxl import Workbook

    from odus.raw_data_diagnosis import diagnose_file

    wb = Workbook()
    for row in (["AGE", 20, 21], ["HEROIN", 0, 7]):
        wb.active.append(row)
    fp = tmp_path / "x.xlsx"
    wb.save(fp)
    layout = SurveyLayout(year_row=None, age_row=0, data_start_row=1)
    problems = {(p.rule, p.coordinate) for p in diagnose_file(fp, layout=layout)}
    assert problems == {("not_binary", "C2")}
