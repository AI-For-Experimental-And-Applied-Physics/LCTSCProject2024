import pydicom
import numpy as np
import SimpleITK as sitk
from pathlib import Path

def load_ct_series(ct_dir: Path):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_IDs:
        raise ValueError(f"No DICOM series found in: {ct_dir}")
    series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_dir), series_IDs[0])
    image = sitk.ReadImage(series_files)
    img_array = sitk.GetArrayFromImage(image).astype(np.int16)

    intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0.0  # RescaleIntercept
    slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1.0      # RescaleSlope
    hu_image = (img_array * slope + intercept).transpose(1, 2, 0)

    spacing = image.GetSpacing()  # (z, y, x)
    pixel_dimensions = (spacing[0], spacing[1], spacing[2])  # Convert to (x, y, z)

    return hu_image, pixel_dimensions