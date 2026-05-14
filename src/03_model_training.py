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

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, ConfusionMatrixDisplay)
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

INPUT_PATH = r"C:\Users\User\Documents\ENSANUT\features.csv"
MODELS_DIR = Path(r"C:\Users\User\Documents\ENSANUT\models")
FIGURES_DIR = Path(r"C:\Users\User\Documents\ENSANUT\figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

FEATURES = [c for c in df.columns if c not in ["dislipidemia", "infarto_mio"]]

def train_and_evaluate(target: str, df: pd.DataFrame, features: list):
    print(f"\n{'='*60}")
    print(f"TARGET: {target}")
    print(f"{'='*60}")

    df_t = df[features + [target]].dropna(subset=[target])
    X = df_t[features]
    y = df_t[target].astype(int)

    pos = y.sum()
    neg = (y == 0).sum()
    spw = neg / pos  # scale_pos_weight para desbalance
    print(f"Positivos: {pos:,} ({pos/len(y):.1%}) | Negativos: {neg:,}")
    print(f"scale_pos_weight: {spw:.1f}")

    # Pipeline: imputación → XGBoost
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
        ))
    ])

    # Validación cruzada estratificada 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Para early stopping necesitamos fit manual en cada fold
    aucs, aps = [], []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        imp = SimpleImputer(strategy="median")
        X_tr_i  = imp.fit_transform(X_tr)
        X_val_i = imp.transform(X_val)

        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, use_label_encoder=False,
            eval_metric="auc", random_state=42, n_jobs=-1,
            early_stopping_rounds=30, verbosity=0
        )
        clf.fit(X_tr_i, y_tr,
                eval_set=[(X_val_i, y_val)],
                verbose=False)

        proba = clf.predict_proba(X_val_i)[:, 1]
        aucs.append(roc_auc_score(y_val, proba))
        aps.append(average_precision_score(y_val, proba))
        print(f"  Fold {fold}: AUROC={aucs[-1]:.3f} | AP={aps[-1]:.3f}")

    print(f"\n  Media AUROC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Media AP:    {np.mean(aps):.3f}  ± {np.std(aps):.3f}")

    # Entrenamiento final en todos los datos
    imp_final = SimpleImputer(strategy="median")
    X_full = imp_final.fit_transform(X)

    clf_final = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="auc",
        random_state=42, n_jobs=-1, verbosity=0
    )
    clf_final.fit(X_full, y)

    # Guardar modelo e imputer
    with open(MODELS_DIR / f"{target}_model.pkl", "wb") as f:
        pickle.dump({"clf": clf_final, "imputer": imp_final,
                     "features": features, "aucs": aucs}, f)

    print(f"  Modelo guardado en outputs/models/{target}_model.pkl")

    return clf_final, imp_final, X, y, features

# Entrenar ambos modelos
models = {}
for target in ["dislipidemia", "infarto_mio"]:
    clf, imp, X, y, feats = train_and_evaluate(target, df, FEATURES)
    models[target] = {"clf": clf, "imputer": imp, "X": X, "y": y}
    # src/04_shap_analysis.py