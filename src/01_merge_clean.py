from pathlib import Path

import pandas as pd
import numpy as np

# ── Configuración ─────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"

ADULTOS_PATH  = DATA_RAW / "ensadul2022_entrega_w.csv"
ANTRO_PATH    = DATA_RAW / "ensaantro2022_entrega_w.csv"
MERGED_PATH   = DATA_PROC / "merged.csv"
FEATURES_PATH = DATA_PROC / "features.csv"

# Llave correcta
KEY_COLS = [
    "folio_int"
]

# ── Función de carga ──────────────────────────────────────────

def load_and_clean(path):

    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False
    )

    # limpiar nombres
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
    )

    # missing values ENSANUT
    df.replace({
        222.22: np.nan,
        222: np.nan,
        999: np.nan,
        99: np.nan,
        9999: np.nan,
        88: np.nan
    }, inplace=True)

    return df

# ── Carga ────────────────────────────────────────────────────

df_adultos = load_and_clean(ADULTOS_PATH)
df_antro = load_and_clean(ANTRO_PATH)

print("\nCOLUMNAS ADULTOS:")
print(df_adultos.columns.tolist())

print("\nCOLUMNAS ANTRO:")
print(df_antro.columns.tolist())

print(f"Adultos: {df_adultos.shape}")
print(f"Antro/Tensión: {df_antro.shape}")

# ── Validación ───────────────────────────────────────────────

print("\nColumnas KEY:")
print(KEY_COLS)

print("\nDuplicados en adultos:")
print(df_adultos.duplicated(KEY_COLS).sum())

print("\nDuplicados en antro:")
print(df_antro.duplicated(KEY_COLS).sum())

for col in KEY_COLS:
    print(
        col,
        "| adultos:", col in df_adultos.columns,
        "| antro:", col in df_antro.columns
    )

# ── Merge ────────────────────────────────────────────────────

df = pd.merge(
    df_adultos,
    df_antro,
    on=KEY_COLS,
    how="left",
    validate="one_to_one",
    suffixes=("_adu", "_ant")
)

print(f"\nMerged: {df.shape}")

if "entidad_adu" in df.columns:
    df["entidad"] = pd.to_numeric(df["entidad_adu"], errors="coerce")
elif "entidad_ant" in df.columns:
    df["entidad"] = pd.to_numeric(df["entidad_ant"], errors="coerce")
elif "entidad" in df.columns:
    df["entidad"] = pd.to_numeric(df["entidad"], errors="coerce")
else:
    df["entidad"] = np.nan

print(f"Entidad — no nulos: {df['entidad'].notna().sum()} / {len(df)}")

# match rate
if "an01_1" in df.columns:
    print(f"Match rate: {df['an01_1'].notna().mean():.1%}")

# Dislipidemia: colesterol alto (a0604==1) O triglicéridos altos (a0606==1)
df["dislipidemia"] = np.where(
    df["a0604"].isna() & df["a0606"].isna(),
    np.nan,
    (
        (df["a0604"] == 1) |
        (df["a0606"] == 1)
    ).astype(int)
)

# Infarto al miocardio (a0502a==1)
df["infarto_mio"] = np.where(
    df["a0502a"].isna(),
    np.nan,
    (df["a0502a"] == 1).astype(int)
)
# ── Medidas antropométricas ────────────────────────────────────────────────────

# BMI: usamos primera medición de peso (an01_1) y talla (an04_1)
# Peso en kg, talla en cm → convertir talla a metros
df["peso_kg"]  = pd.to_numeric(df.get("an01_1", np.nan), errors="coerce")
df["talla_cm"] = pd.to_numeric(df.get("an04_1", np.nan), errors="coerce")
df.loc[df["talla_cm"] <= 0, "talla_cm"] = np.nan
df["bmi"] = df["peso_kg"] / (df["talla_cm"] / 100) ** 2
df["bmi"] = df["bmi"].clip(10, 70)  # Winsorizar valores extremos

# Cintura (an08_1, primera medición)
df["cintura_cm"] = pd.to_numeric(df.get("an08_1", np.nan), errors="coerce")
df["cintura_cm"] = df["cintura_cm"].clip(40, 200)

# ── Tensión arterial ───────────────────────────────────────────────────────────

for i, suffix in [(1, "01"), (2, "02"), (3, "03")]:
    df[f"pas_{i}"] = pd.to_numeric(df.get(f"an27_{suffix}s", np.nan), errors="coerce")
    df[f"pad_{i}"] = pd.to_numeric(df.get(f"an27_{suffix}d", np.nan), errors="coerce")

# Promedio de las 3 mediciones (ignorar NaN)
df["pas_promedio"] = df[["pas_1","pas_2","pas_3"]].mean(axis=1)
df["pad_promedio"] = df[["pad_1","pad_2","pad_3"]].mean(axis=1)

# Presión de pulso
df["pulso_presion"] = df["pas_promedio"] - df["pad_promedio"]

# HTA medida
# HTA medida
df["hta_medida"] = np.where(
    df["pas_promedio"].isna() & df["pad_promedio"].isna(),
    np.nan,
    (
        (df["pas_promedio"] >= 130) |
        (df["pad_promedio"] >= 80)
    ).astype(int)
)
# ── Variables de morbilidad ────────────────────────────────────────────────────

df["diabetes_dx"]  = df.get("a0301", pd.Series(np.nan, index=df.index)).eq(1).astype(int)
df["hta_dx"]       = df.get("a0401", pd.Series(np.nan, index=df.index)).eq(1).astype(int)

# ── Factores de riesgo ─────────────────────────────────────────────────────────

# Tabaco: 1=diario, 2=algunos días → fuma_actual=1; 3=no fuma → 0
df["tabaco_actual"] = df.get("a1301", pd.Series(np.nan, index=df.index)).isin([1,2]).astype(int)

# Alcohol: escala 1(diario)–6(nunca) → invertir para que mayor = más riesgo
df["alcohol_riesgo"] = (7 - df.get("a1308", pd.Series(np.nan, index=df.index))).clip(0, 6)

# ── Sintomatología depresiva (Escala CES-D abreviada) ─────────────────────────
# ítems a0211–a0217 (escala 1-4, ítem f es positivo → invertir)
dep_items = [f"a02{str(i).zfill(2)}" for i in range(11, 18)]
for col in dep_items:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Ítem f (a0216) es positivo: invertir (5 - valor)
if "a0216" in df.columns:
    df["a0216"] = 5 - df["a0216"]

# Suma de ítems disponibles
df["dep_score"] = df[[c for c in dep_items if c in df.columns]].sum(axis=1, min_count=4)

# ── Antecedentes heredofamiliares ──────────────────────────────────────────────

df["antec_dm_padre"]    = df.get("a0701p", pd.Series(np.nan, index=df.index)).eq(1).astype(int)
df["antec_dm_madre"]    = df.get("a0701m", pd.Series(np.nan, index=df.index)).eq(1).astype(int)
df["antec_hta_padre"]   = df.get("a0702p", pd.Series(np.nan, index=df.index)).eq(1).astype(int)
df["antec_hta_madre"]   = df.get("a0702m", pd.Series(np.nan, index=df.index)).eq(1).astype(int)

# Infarto familiar (padre o madre)
df["antec_infarto_fam"] = (
    df.get("a0703p", pd.Series(np.nan, index=df.index)).eq(1) |
    df.get("a0703m", pd.Series(np.nan, index=df.index)).eq(1)
).astype(int)

# ── Variables demográficas ─────────────────────────────────────────────────────

# Sexo: el N.R. viene en el listado del hogar; asumimos columna 'sexo' (1=H, 2=M)
df["sexo"] = pd.to_numeric(df.get("sexo", np.nan), errors="coerce")
df["es_mujer"] = df["sexo"].eq(2).astype(int)

# Edad
df["edad"] = pd.to_numeric(df.get("edad", np.nan), errors="coerce")

# Entidad federativa
df["entidad"] = pd.to_numeric(df.get("entidad", np.nan), errors="coerce")

# ── Selección del dataset final ────────────────────────────────────────────────

FEATURES = [
    "bmi", "cintura_cm", "pas_promedio", "pad_promedio", "pulso_presion",
    "hta_medida", "diabetes_dx", "hta_dx",
    "tabaco_actual", "alcohol_riesgo", "dep_score",
    "antec_dm_padre", "antec_dm_madre", "antec_hta_padre", "antec_hta_madre",
    "antec_infarto_fam", "es_mujer", "edad", "entidad",
]

TARGETS = ["dislipidemia", "infarto_mio"]

df_out = df[FEATURES + TARGETS].copy()

# Eliminar filas sin ninguna variable objetivo
df_out.dropna(subset=TARGETS, how="all", inplace=True)

print(f"Dataset final: {df_out.shape}")
print("\nPrevalencias:")
for t in TARGETS:
    n = df_out[t].notna().sum()
    p = df_out[t].mean()
    print(f"  {t}: {p:.1%} (n={n:,})")

df.to_csv(MERGED_PATH, index=False)

print(f"\nMerge guardado en: {MERGED_PATH}")
df_out.to_csv(FEATURES_PATH, index=False)

print(f"\nFeatures guardadas en: {FEATURES_PATH}")