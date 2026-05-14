from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import urllib.request
import pickle

# ── Rutas ──────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW      = BASE_DIR / "data" / "raw"
DATA_PROC     = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "outputs" / "models"
OUTPUT_DIR    = BASE_DIR / "outputs"
GEOJSON_PATH  = DATA_RAW / "mexico_estados.geojson"
FEATURES_PATH = DATA_PROC / "features.csv"

for p in [DATA_RAW, DATA_PROC, OUTPUT_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GEOJSON MÉXICO
# =============================================================================

GEOJSON_URL = (
    "https://raw.githubusercontent.com/angelnmara/"
    "geojson/master/mexicoHigh.json"
)

GEOJSON_PATH = DATA_RAW / "mexico_estados.geojson"

if not GEOJSON_PATH.exists():

    print("Descargando GeoJSON de México...")

    urllib.request.urlretrieve(
        GEOJSON_URL,
        GEOJSON_PATH
    )

    print("GeoJSON descargado")

# cargar geojson
with open(GEOJSON_PATH, encoding="utf-8") as f:
    mexico_geo = json.load(f)

# validar estructura
print("\nPropiedades GeoJSON:")
print(mexico_geo["features"][0]["properties"])

# =============================================================================
# CATÁLOGO ESTADOS
# =============================================================================

ESTADOS = {
    1: "Aguascalientes",
    2: "Baja California",
    3: "Baja California Sur",
    4: "Campeche",
    5: "Coahuila",
    6: "Colima",
    7: "Chiapas",
    8: "Chihuahua",
    9: "Ciudad de México",
    10: "Durango",
    11: "Guanajuato",
    12: "Guerrero",
    13: "Hidalgo",
    14: "Jalisco",
    15: "México",
    16: "Michoacán",
    17: "Morelos",
    18: "Nayarit",
    19: "Nuevo León",
    20: "Oaxaca",
    21: "Puebla",
    22: "Querétaro",
    23: "Quintana Roo",
    24: "San Luis Potosí",
    25: "Sinaloa",
    26: "Sonora",
    27: "Tabasco",
    28: "Tamaulipas",
    29: "Tlaxcala",
    30: "Veracruz",
    31: "Yucatán",
    32: "Zacatecas"
}

# =============================================================================
# CARGAR FEATURES
# =============================================================================

df = pd.read_csv(FEATURES_PATH)

print(f"\nDataset cargado: {df.shape}")

# =============================================================================
# AGREGAR PROBABILIDADES MODELO
# =============================================================================

TARGETS = [
    "dislipidemia",
    "infarto_mio"
]

def add_model_proba(df, target):

    print(f"\nAgregando probabilidades: {target}")

    model_path = MODELS_DIR / f"{target}_model.pkl"

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    clf = bundle["clf"]
    imp = bundle["imputer"]
    feats = bundle["features"]

    # copiar features necesarias
    X = df[feats].copy()

    # detectar columnas completamente NaN
    all_nan_cols = X.columns[X.isna().all()].tolist()

    if all_nan_cols:

        print("\nColumnas completamente NaN:")
        print(all_nan_cols)

    # FIX ESPECÍFICO PARA ENTIDAD
    if "entidad" in X.columns:

        if X["entidad"].isna().all():

            print("\nWARNING: entidad completamente NaN")

            # valor temporal
            X["entidad"] = 14

    # asegurar mismo orden de columnas
    X = X[feats]

    # imputación
    X_imp = imp.transform(X)

    print(f"Shape imputado: {X_imp.shape}")

    # predicción
    df[f"proba_{target}"] = (
        clf.predict_proba(X_imp)[:, 1]
    )

    return df

# agregar probabilidades
for t in TARGETS:
    df = add_model_proba(df, t)

# =============================================================================
# RESUMEN POR ESTADO
# =============================================================================

summary = df.groupby("entidad").agg(

    n=("dislipidemia", "count"),

    prev_dislipidemia=(
        "dislipidemia",
        "mean"
    ),

    prev_infarto=(
        "infarto_mio",
        "mean"
    ),

    prob_dislipidemia_media=(
        "proba_dislipidemia",
        "mean"
    ),

    prob_infarto_media=(
        "proba_infarto_mio",
        "mean"
    ),

    bmi_media=(
        "bmi",
        "mean"
    ),

    cintura_media=(
        "cintura_cm",
        "mean"
    ),

    pas_media=(
        "pas_promedio",
        "mean"
    )

).reset_index()

# mapear nombres estados
summary["entidad_nombre"] = (
    summary["entidad"]
    .map(ESTADOS)
)

# eliminar estados desconocidos
summary = summary[
    summary["entidad_nombre"].notna()
]

# =============================================================================
# CONVERTIR A PORCENTAJE
# =============================================================================

pct_cols = [
    "prev_dislipidemia",
    "prev_infarto",
    "prob_dislipidemia_media",
    "prob_infarto_media"
]

for col in pct_cols:

    summary[col] = (
        summary[col] * 100
    ).round(1)

# guardar resumen
summary_path = OUTPUT_DIR / "prevalencia_por_estado.csv"

summary.to_csv(
    summary_path,
    index=False
)

print(f"\nResumen guardado:")
print(summary_path)

# =============================================================================
# FUNCIÓN MAPA
# =============================================================================

def mapa_coropleta(
    df_summary,
    variable,
    titulo,
    colorscale="RdYlGn_r",
    unidad="%"
):

    fig = go.Figure(

        go.Choropleth(

            geojson=mexico_geo,

            locations=df_summary[
                "entidad_nombre"
            ],

            featureidkey="properties.name",

            z=df_summary[variable],

            colorscale=colorscale,

            zmin=df_summary[variable].min(),

            zmax=df_summary[variable].max(),

            colorbar=dict(
                title=f"<b>{unidad}</b>",
                thickness=15
            ),

            text=df_summary.apply(

                lambda r:
                    f"<b>{r['entidad_nombre']}</b><br>"
                    f"{titulo}: {r[variable]:.1f}%<br>"
                    f"n={r['n']:,}",

                axis=1
            ),

            hoverinfo="text",

            marker_line_color="white",

            marker_line_width=0.5
        )
    )

    fig.update_geos(

        fitbounds="locations",

        visible=False,

        bgcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(

        title=dict(
            text=f"<b>{titulo}</b>",
            x=0.5,
            font_size=18
        ),

        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        ),

        paper_bgcolor="#F8F9FA",

        height=600
    )

    return fig

# =============================================================================
# MAPAS
# =============================================================================

print("\nGenerando mapas...")

# -------------------------------------------------------------------------
# PREVALENCIA DISLIPIDEMIA
# -------------------------------------------------------------------------

fig1 = mapa_coropleta(
    summary,
    "prev_dislipidemia",
    "Prevalencia de Dislipidemia (%)",
    "RdYlGn_r"
)

fig1_path = OUTPUT_DIR / "mapa_dislipidemia.html"

fig1.write_html(fig1_path)

print(f"Guardado: {fig1_path}")

# -------------------------------------------------------------------------
# PREVALENCIA INFARTO
# -------------------------------------------------------------------------

fig2 = mapa_coropleta(
    summary,
    "prev_infarto",
    "Prevalencia de Infarto al Miocardio (%)",
    "OrRd"
)

fig2_path = OUTPUT_DIR / "mapa_infarto.html"

fig2.write_html(fig2_path)

print(f"Guardado: {fig2_path}")

# -------------------------------------------------------------------------
# RIESGO PREDICHO DISLIPIDEMIA
# -------------------------------------------------------------------------

fig3 = mapa_coropleta(
    summary,
    "prob_dislipidemia_media",
    "Probabilidad Media Predicha — Dislipidemia (%)",
    "RdYlBu_r"
)

fig3_path = OUTPUT_DIR / "mapa_dislipidemia_modelo.html"

fig3.write_html(fig3_path)

print(f"Guardado: {fig3_path}")

# -------------------------------------------------------------------------
# RIESGO PREDICHO INFARTO
# -------------------------------------------------------------------------

fig4 = mapa_coropleta(
    summary,
    "prob_infarto_media",
    "Probabilidad Media Predicha — Infarto (%)",
    "PuRd"
)

fig4_path = OUTPUT_DIR / "mapa_infarto_modelo.html"

fig4.write_html(fig4_path)

print(f"Guardado: {fig4_path}")

# =============================================================================
# TOP ESTADOS
# =============================================================================

print("\nTop prevalencia dislipidemia:\n")

print(

    summary[
        [
            "entidad_nombre",
            "prev_dislipidemia",
            "prev_infarto",
            "n"
        ]
    ]

    .sort_values(
        "prev_dislipidemia",
        ascending=False
    )

    .to_string(index=False)
)

print("\nDONE")