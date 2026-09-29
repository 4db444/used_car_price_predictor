from pathlib import Path
from os import environ
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
PRECLEANING_STATS_PATH = PROJECT_ROOT / "reports" / "tables" / "pre-cleaning"
PRECLEANING_FIGURES_PATH = PROJECT_ROOT / "reports" / "figures" / "pre-cleaning"
POSTCLEANING_STATS_PATH = PROJECT_ROOT / "reports" / "tables" / "post-cleaning"
POSTCLEANING_FIGURES_PATH = PROJECT_ROOT / "reports" / "figures" / "post-cleaning"
UNOPTIMIZED_MODELS_STATS = PROJECT_ROOT / "reports" / "tables" / "models" / "pre-optimization"
OPTIMIZED_MODELS_STATS = PROJECT_ROOT / "reports" / "tables" / "models" / "post-optimization"
MODELS_EVALUATION_FIGURES_PATH = PROJECT_ROOT / "reports" / "figures" / "models"
MODELS_PATH = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_PATH / "final_model.joblib"

CATEGORICAL_COLS = ["fuel", "seller_type", "transmission", "owner"]
NUMERICAL_COLS = ["year", "selling_price", "km_driven"]
KEY_GROUPING_COLS = {
    "fuel" : "name",
    "transmission" : "name",
    "owner" : "year"
}

RANDOM_SEED = int(environ["RANDOM_SEED"])