"""
Load and prepare the Diabetes 130-US Hospitals data (1999-2008, 101,766 encounters).

Target: readmitted within 30 days ("<30"), the window hospitals are penalised on under
Medicare's Hospital Readmissions Reduction Program.

Cleaning decisions (each one documented because each one changes the answer):
- Encounters ending in death or hospice are removed: those patients cannot be readmitted,
  so keeping them would teach the model that "sickest = least likely to return".
- '?' is missing. `weight` (97% missing) is dropped. `payer_code` and `medical_specialty`
  keep missing as an explicit "Unknown" category, because missingness itself can be informative.
- ICD-9 diagnosis codes are grouped into 9 clinical categories (Strack et al., 2014).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "diabetic_data.csv.gz"

DEATH_OR_HOSPICE = {11, 13, 14, 19, 20, 21}
MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
        "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

ADMISSION_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Trauma"}
DISCHARGE = {1: "Home", 6: "Home with health service", 3: "Skilled nursing facility",
             22: "Rehab facility", 2: "Other short-term hospital", 5: "Other inpatient care"}
ADMISSION_SOURCE = {7: "Emergency room", 1: "Physician referral", 2: "Clinic referral",
                    4: "Transfer from hospital", 6: "Transfer from other facility"}


def icd9_group(code) -> str:
    """Map an ICD-9 code to a clinical category (Strack et al., 2014)."""
    if pd.isna(code):
        return "Missing"
    c = str(code)
    if c.startswith(("V", "E")):
        return "Other"
    try:
        v = float(c)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def load_raw(path: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(path, na_values="?", low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["discharge_disposition_id"].isin(DEATH_OR_HOSPICE)].copy()
    df = df[df["gender"] != "Unknown/Invalid"]
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns=["weight"])


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["patient_nbr"] = df["patient_nbr"]
    out["encounter_id"] = df["encounter_id"]

    # demographics
    out["age_years"] = df["age"].str.extract(r"\[(\d+)-").astype(float)[0] + 5   # bracket midpoint
    out["gender"] = df["gender"]
    out["race"] = df["race"].fillna("Unknown")

    # admission context
    out["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE).fillna("Other/unknown")
    out["admission_source"] = df["admission_source_id"].map(ADMISSION_SOURCE).fillna("Other/unknown")
    out["discharge_to"] = df["discharge_disposition_id"].map(DISCHARGE).fillna("Other/unknown")
    spec = df["medical_specialty"].fillna("Unknown")
    top = spec.value_counts().index[:10]
    out["medical_specialty"] = spec.where(spec.isin(top), "Other")
    out["payer_known"] = df["payer_code"].notna().astype(int)

    # utilisation history: the strongest readmission signal in most studies
    for c in ["number_inpatient", "number_emergency", "number_outpatient"]:
        out[c] = df[c]
    out["prior_visits_total"] = df[["number_inpatient", "number_emergency", "number_outpatient"]].sum(axis=1)

    # stay intensity
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]:
        out[c] = df[c]

    # diagnoses
    out["primary_dx"] = df["diag_1"].map(icd9_group)
    out["secondary_dx"] = df["diag_2"].map(icd9_group)
    out["has_diabetes_dx_any"] = df[["diag_1", "diag_2", "diag_3"]].astype(str).apply(
        lambda s: s.str.startswith("250")).any(axis=1).astype(int)

    # labs and medication management
    out["a1c_result"] = df["A1Cresult"].fillna("Not measured")
    out["glucose_serum"] = df["max_glu_serum"].fillna("Not measured")
    out["insulin"] = df["insulin"]
    out["n_meds_changed"] = df[MEDS].isin(["Up", "Down"]).sum(axis=1)
    out["n_diabetes_meds"] = (df[MEDS] != "No").sum(axis=1)
    out["med_change"] = (df["change"] == "Ch").astype(int)
    out["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)

    out["readmit_30"] = df["readmit_30"]
    return out


def load_dataset() -> pd.DataFrame:
    return engineer(clean(load_raw()))


CATEGORICAL = ["gender", "race", "admission_type", "admission_source", "discharge_to", "medical_specialty",
               "primary_dx", "secondary_dx", "a1c_result", "glucose_serum", "insulin"]
NUMERIC = ["age_years", "payer_known", "number_inpatient", "number_emergency", "number_outpatient",
           "prior_visits_total", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_diagnoses", "has_diabetes_dx_any", "n_meds_changed", "n_diabetes_meds", "med_change",
           "on_diabetes_med"]
