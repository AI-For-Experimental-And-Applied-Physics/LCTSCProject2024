import pandas as pd
from .config import METADATA_FILE

def load_metadata():
    df = pd.read_csv(METADATA_FILE)
    for col in ['PatientID', 'CTPath', 'RTSTRUCTPath', 'ROIName']:
        if col not in df.columns:
            raise ValueError(f"Missing required column in metadata: {col}")
    return df