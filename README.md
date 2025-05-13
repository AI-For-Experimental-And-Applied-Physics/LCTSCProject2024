# LCTSCProject2024

Dummy repository containing all suggested python modules to use for the training of a NN on the Lung CT Segmentation Challenge Dataset

## Dummy repo

This repository preprocesses the Lung CT Segmentation Challenge dataset by:
- Reading DICOM CT and RTSTRUCT files
- Converting images to Hounsfield Units (HU)
- Extracting segmentation masks from RTSTRUCTs
- Saving image/mask pairs as NumPy `.npz` files

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```