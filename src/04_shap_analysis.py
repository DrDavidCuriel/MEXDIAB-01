from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"

ADULTOS_PATH  = DATA_RAW / "ensadul2022_entrega_w.csv"
ANTRO_PATH    = DATA_RAW / "ensaantro2022_entrega_w.csv"
MERGED_PATH   = DATA_PROC / "merged.csv"
FEATURES_PATH = DATA_PROC / "features.csv"

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import shap
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────

INPUT_PATH = r"C:\Users\User\Documents\ENSANUT\features.csv"

MODELS_DIR = Path(r"C:\Users\User\Documents\ENSANUT\models")
SHAP_DIR = Path(r"C:\Users\User\Documents\ENSANUT\shap")

SHAP_DIR.mkdir(parents=True, exist_ok=True)

# ── Load dataset ────────────────────────────────────────

df = pd.read_csv(INPUT_PATH)

TARGETS = ["dislipidemia", "infarto_mio"]

FEATURES = [
    c for c in df.columns
    if c not in TARGETS
]

# ── Función SHAP ────────────────────────────────────────

def run_shap(target):

    print(f"\n{'='*60}")
    print(f"SHAP ANALYSIS: {target}")
    print(f"{'='*60}")

    # ── cargar modelo ────────────────────────────────

    with open(MODELS_DIR / f"{target}_model.pkl", "rb") as f:
        data = pickle.load(f)

    clf = data["clf"]
    imputer = data["imputer"]

    # ── dataset target ───────────────────────────────

    df_t = df[FEATURES + [target]].dropna(subset=[target])

    X = df_t[FEATURES]

    # ── imputación ───────────────────────────────────

    X_imp = imputer.transform(X)

    # nombres reales de columnas
    feature_names = X.columns.tolist()

    # ajuste defensivo por mismatch
    if X_imp.shape[1] != len(feature_names):

        print("\nWARNING:")
        print(f"X_imp shape = {X_imp.shape}")
        print(f"feature_names = {len(feature_names)}")

        feature_names = feature_names[:X_imp.shape[1]]

    # convertir a DataFrame
    X_imp = pd.DataFrame(
        X_imp,
        columns=feature_names
    )

    # validación
    assert X_imp.shape[1] == len(feature_names)

    print(f"\nDatos imputados: {X_imp.shape}")

    # ── SHAP explainer ───────────────────────────────

    explainer = shap.TreeExplainer(clf)

    shap_values = explainer.shap_values(X_imp)

    print("SHAP calculado")

    # ── Summary plot (beeswarm) ──────────────────────

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_imp,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_DIR / f"{target}_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Summary plot guardado")

    # ── Bar plot importancia global ─────────────────

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_imp,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_DIR / f"{target}_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Bar plot guardado")

    # ── Importancia SHAP ─────────────────────────────

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    # guardar CSV
    importance_df.to_csv(
        SHAP_DIR / f"{target}_importance.csv",
        index=False
    )

    print("\nTop 10 features:")

    print(importance_df.head(10))

    # ── Dependence plots ─────────────────────────────

    top_features = importance_df.head(5)["feature"].tolist()

    for feat in top_features:

        print(f"Dependence plot: {feat}")

        plt.figure()

        shap.dependence_plot(
            feat,
            shap_values,
            X_imp,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            SHAP_DIR / f"{target}_dependence_{feat}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ── Waterfall example ────────────────────────────

    idx = 0

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_imp.iloc[idx],
        feature_names=feature_names
    )

    plt.figure()

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_DIR / f"{target}_waterfall_example.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Waterfall plot guardado")

# ── Ejecutar ───────────────────────────────────────────

for target in TARGETS:

    run_shap(target)

print("\nDONE")
