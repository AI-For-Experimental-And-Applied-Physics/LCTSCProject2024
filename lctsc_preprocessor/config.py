from pathlib import Path

METADATA_FILE = Path("data/lctsc_metadata.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)