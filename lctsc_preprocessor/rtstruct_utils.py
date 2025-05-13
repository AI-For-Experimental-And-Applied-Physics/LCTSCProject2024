from rt_utils import RTStructBuilder
from pathlib import Path

def load_rtstruct(rtstruct_path: Path, ct_dir: Path):
    return RTStructBuilder.create_from(
        dicom_series_path=str(ct_dir),
        rt_struct_path=str(rtstruct_path)
    )