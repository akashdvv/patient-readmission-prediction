"""
Train and evaluate 30-day readmission models.

- Patient-level split: every encounter of a given patient is in train OR test, never both.
  (Splitting by encounter lets the model memorise repeat patients and inflates scores.)
- Models: class-balanced logistic regression and random forest, compared with 5-fold
  GroupKFold cross-validation on the training patients, then evaluated once on held-out patients.
- Metrics suited to an 11% positive rate: ROC-AUC, PR-AUC, and "capture rate" - the share of all
  readmissions found if a care-transitions team can follow up only the top-risk X% of discharges.

    python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, ROOT, load_dataset

SEED = 42
RESULTS, IMAGES, TABLEAU = ROOT / "results", ROOT / "images", ROOT / "tableau"
BLUE, ORANGE, AQUA, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
})


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), CATEGORICAL),
    ])


def models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(
            C=0.3, class_weight="balanced", max_iter=3000))]),
        "Random Forest": Pipeline([("pre", preprocessor()), ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=40, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))]),
    }


def capture_rate(y: np.ndarray, p: np.ndarray, top_frac: float) -> float:
    """Share of all readmissions that fall in the top `top_frac` highest-risk discharges."""
    k = int(len(p) * top_frac)
    idx = np.argsort(-p)[:k]
    return y[idx].sum() / y.sum()


def main() -> None:
    for d in (RESULTS, IMAGES, TABLEAU):
        d.mkdir(exist_ok=True)
    df = load_dataset()
    X, y, groups = df[CATEGORICAL + NUMERIC], df["readmit_30"].to_numpy(), df["patient_nbr"].to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
    assert not set(groups[tr]) & set(groups[te]), "patient leakage between train and test"
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    print(f"{len(df):,} encounters, {df.patient_nbr.nunique():,} patients, 30-day readmission rate {y.mean():.1%}")
    print(f"train {len(tr):,} / test {len(te):,} encounters (patient-level split)")

    cv_rows, fitted, probs = [], {}, {}
    for name, pipe in models().items():
        s = cross_validate(pipe, Xtr, ytr, groups=groups[tr], cv=GroupKFold(n_splits=5),
                           scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"}, n_jobs=1)
        cv_rows.append({"model": name, "cv_roc_auc": s["test_roc_auc"].mean(), "cv_roc_auc_std": s["test_roc_auc"].std(),
                        "cv_pr_auc": s["test_pr_auc"].mean()})
        fitted[name] = pipe.fit(Xtr, ytr)
        probs[name] = pipe.predict_proba(Xte)[:, 1]
        print(f"{name:<20} CV ROC-AUC {cv_rows[-1]['cv_roc_auc']:.3f} ± {cv_rows[-1]['cv_roc_auc_std']:.3f} | "
              f"test ROC-AUC {roc_auc_score(yte, probs[name]):.3f}")

    test_rows = []
    for name, p in probs.items():
        test_rows.append({"model": name, "test_roc_auc": roc_auc_score(yte, p), "test_pr_auc": average_precision_score(yte, p),
                          **{f"capture_top{int(f*100)}pct": capture_rate(yte, p, f) for f in (0.1, 0.2, 0.3)}})
    results = pd.DataFrame(cv_rows).merge(pd.DataFrame(test_rows), on="model").round(4)
    results.to_csv(RESULTS / "model_comparison.csv", index=False)
    best = results.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]
    p_best = probs[best]

    # operating point: flag the top 20% of discharges for a follow-up call / visit
    k = int(0.2 * len(p_best))
    threshold = float(np.sort(p_best)[::-1][k - 1])
    flagged = p_best >= threshold
    op = {"flag_share": float(flagged.mean()), "recall": float(yte[flagged].sum() / yte.sum()),
          "precision": float(yte[flagged].mean()), "base_rate": float(yte.mean()),
          "lift": float(yte[flagged].mean() / yte.mean())}

    imp = permutation_importance(fitted[best], Xte, yte, scoring="average_precision", n_repeats=5,
                                 random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({"feature": Xte.columns, "importance": imp.importances_mean,
                               "std": imp.importances_std}).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)

    summary = {"encounters": int(len(df)), "patients": int(df.patient_nbr.nunique()),
               "readmission_rate_30d": round(float(y.mean()), 4), "best_model": best,
               "results": results.to_dict(orient="records"), "operating_point_top20pct": {k_: round(v, 4) for k_, v in op.items()},
               "top_features": importance.head(8).feature.tolist()}
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    joblib.dump({"model": fitted[best], "threshold": threshold, "features": CATEGORICAL + NUMERIC},
                ROOT / "results" / "readmission_model.joblib")

    # Tableau extracts
    scored = df.iloc[te][["encounter_id", "age_years", "gender", "race", "admission_type", "discharge_to",
                          "primary_dx", "number_inpatient", "time_in_hospital", "a1c_result", "readmit_30"]].copy()
    scored["risk_score"] = p_best.round(4)
    scored["risk_tier"] = pd.qcut(scored["risk_score"], [0, 0.5, 0.8, 1.0], labels=["Low", "Medium", "High"])
    scored.to_csv(TABLEAU / "scored_test_encounters.csv", index=False)
    for dim in ["primary_dx", "discharge_to", "admission_type", "a1c_result", "number_inpatient"]:
        g = df.groupby(dim).agg(encounters=("readmit_30", "size"), readmit_rate=("readmit_30", "mean")).reset_index()
        g.insert(0, "dimension", dim)
        g = g.rename(columns={dim: "value"})
        g.to_csv(TABLEAU / f"rate_by_{dim}.csv", index=False)
    df["age_band"] = pd.cut(df["age_years"], [0, 40, 60, 70, 80, 100], labels=["<40", "40-59", "60-69", "70-79", "80+"])
    df.groupby("age_band", observed=True).agg(encounters=("readmit_30", "size"), readmit_rate=("readmit_30", "mean")
                                               ).reset_index().to_csv(TABLEAU / "rate_by_age_band.csv", index=False)

    charts(df, yte, probs, best, importance, op)
    print(json.dumps(summary, indent=2))


def charts(df, yte, probs, best, importance, op) -> None:
    # 1. readmission by prior inpatient visits (descriptive)
    g = df.assign(prior=df["number_inpatient"].clip(upper=6)).groupby("prior").readmit_30.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar([("6+" if i == 6 else str(i)) for i in g.index], g["mean"] * 100, color=BLUE, width=0.6)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m * 100 + 0.8, f"{m:.0%}", ha="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Inpatient admissions in the prior year")
    ax.set_ylabel("30-day readmission rate (%)")
    ax.set_title("Prior hospital use is the strongest warning sign")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(IMAGES / "readmission_by_prior_inpatient.png", dpi=150)
    plt.close(fig)

    # 2. ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for (name, p), color in zip(probs.items(), [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(yte, p)
        prec, rec, _ = precision_recall_curve(yte, p)
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
        axes[1].plot(rec, prec, color=color, linewidth=2, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
    axes[0].plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=0.8)
    axes[1].axhline(yte.mean(), color=INK_2, linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (held-out patients)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (held-out patients)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    # 3. capture curve: % of readmissions found vs % of discharges followed up
    p = probs[best]
    order = np.argsort(-p)
    cum = np.cumsum(yte[order]) / yte.sum()
    frac = np.arange(1, len(p) + 1) / len(p)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frac * 100, cum * 100, color=BLUE, linewidth=2.2, label=best)
    ax.plot([0, 100], [0, 100], color=INK_2, linestyle="--", linewidth=0.9, label="Random selection")
    ax.plot([20], [op["recall"] * 100], "o", color=BLUE, markersize=9, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"Follow up top 20% → catch {op['recall']:.0%} of readmissions\n({op['lift']:.1f}× the rate of random selection)",
                (20, op["recall"] * 100), xytext=(14, -38), textcoords="offset points", fontsize=9)
    ax.set_xlabel("% of discharges flagged (highest risk first)")
    ax.set_ylabel("% of 30-day readmissions captured")
    ax.set_title("Targeting a care-transitions program")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMAGES / "capture_curve.png", dpi=150)
    plt.close(fig)

    # 4. permutation importance
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top.feature, top.importance, xerr=top["std"], color=BLUE, height=0.6, error_kw={"ecolor": INK_2})
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title(f"What drives predicted readmission risk ({best})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
