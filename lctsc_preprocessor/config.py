from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_FILE = BASE_DIR / "data/lctsc_metadata.csv"
OUTPUT_DIR = BASE_DIR / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)