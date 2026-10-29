import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import CATEGORICAL, NUMERIC, clean, engineer, icd9_group, load_raw  # noqa: E402


def test_icd9_grouping():
    assert icd9_group("428") == "Circulatory"
    assert icd9_group("786") == "Respiratory"
    assert icd9_group("250.83") == "Diabetes"
    assert icd9_group("820") == "Injury"
    assert icd9_group("V57") == "Other"
    assert icd9_group(float("nan")) == "Missing"


def test_cleaning_removes_patients_who_cannot_be_readmitted():
    raw = load_raw()
    c = clean(raw)
    assert not c["discharge_disposition_id"].isin({11, 13, 14, 19, 20, 21}).any()
    assert "weight" not in c.columns
    assert set(c["readmit_30"].unique()) == {0, 1}
    assert len(raw) == 101_766 and len(c) == 99_340


def test_engineered_frame_is_complete_and_model_ready():
    d = engineer(clean(load_raw().head(5000)))
    assert d[CATEGORICAL + NUMERIC].isna().sum().sum() == 0
    assert d["age_years"].between(5, 95).all()
    assert (d["prior_visits_total"] == d[["number_inpatient", "number_emergency", "number_outpatient"]].sum(axis=1)).all()


def test_no_target_leakage_columns():
    banned = {"readmitted", "readmit_30", "encounter_id", "patient_nbr"}
    assert not banned & set(CATEGORICAL + NUMERIC)
