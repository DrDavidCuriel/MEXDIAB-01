from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"

ADULTOS_PATH  = DATA_RAW / "ensadul2022_entrega_w.csv"
ANTRO_PATH    = DATA_RAW / "ensaantro2022_entrega_w.csv"
MERGED_PATH   = DATA_PROC / "merged.csv"
FEATURES_PATH = DATA_PROC / "features.csv"