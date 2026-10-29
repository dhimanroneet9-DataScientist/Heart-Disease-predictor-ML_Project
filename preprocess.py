"""Reproduces the exact preprocessing used in the training notebook."""
import json
import numpy as np

with open("preprocess.json") as f:
    CFG = json.load(f)

COLUMNS = CFG["columns"]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def build_features(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                   resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    """Raw patient values -> the 15-feature vector the model was trained on."""
    # 1) Zeros in these two columns are invalid readings; the notebook filled them with the mean
    if resting_bp == 0:
        resting_bp = CFG["bp_fill"]
    if cholesterol == 0:
        cholesterol = CFG["chol_fill"]

    row = {c: 0.0 for c in COLUMNS}
    # 2) The notebook ran .astype(int), which truncates decimals toward zero
    row["Age"] = int(age)
    row["RestingBP"] = int(resting_bp)
    row["Cholesterol"] = int(cholesterol)
    row["MaxHR"] = int(max_hr)
    row["Oldpeak"] = int(oldpeak)
    row["FastingBS"] = int(fasting_bs == "Yes")
    row["Sex_M"] = int(sex == "M")
    row["ChestPainType_ATA"] = int(chest_pain == "ATA")
    row["ChestPainType_NAP"] = int(chest_pain == "NAP")
    row["ChestPainType_TA"] = int(chest_pain == "TA")
    row["RestingECG_Normal"] = int(resting_ecg == "Normal")
    row["RestingECG_ST"] = int(resting_ecg == "ST")
    row["ExerciseAngina_Y"] = int(exercise_angina == "Yes")
    row["ST_Slope_Flat"] = int(st_slope == "Flat")
    row["ST_Slope_Up"] = int(st_slope == "Up")

    # 3) First standardization (numeric columns, whole dataset)
    for c in NUMERIC:
        row[c] = (row[c] - CFG["pre_mean"][c]) / CFG["pre_scale"][c]

    # 4) Second standardization (all 15 columns, training split)
    x = np.array([row[c] for c in COLUMNS], dtype=float)
    x = (x - np.array(CFG["scaler_mean"])) / np.array(CFG["scaler_scale"])
    return x.reshape(1, -1)
