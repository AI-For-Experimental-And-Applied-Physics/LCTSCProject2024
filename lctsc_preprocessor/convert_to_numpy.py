from .dicom_utils import load_ct_series
from .rtstruct_utils import extract_mask
from .config import PROCESSED_DATA_DIR
import numpy as np
from pathlib import Path

def convert_case(case_id, ct_path, rtstruct_path, roi_name):
    ct_path = Path(ct_path)
    rtstruct_path = Path(rtstruct_path)

    img, _ = load_ct_series(ct_path)
    mask = extract_mask(rtstruct_path, ct_path, roi_name)

    out_file = PROCESSED_DATA_DIR / f"{case_id}.npz"
    np.savez_compressed(out_file, image=img, mask=mask)
    print(f"[✓] Saved: {out_file}")