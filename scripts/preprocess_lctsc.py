import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import argparse
from lctsc_preprocessor.utils import preprocess_case

def main():
    parser = argparse.ArgumentParser(description="Preprocess LCTSC cases.")
    parser.add_argument(
        "metadata_file",
        type=str,
        nargs='?',
        help="Path to the metadata CSV file containing case information."
    )
    parser.add_argument(
        "base_path",
        type=str,
        nargs='?',
        help="Path to the raw data"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the ROIs on the CT images."
    )
    args = parser.parse_args()

    metadata_file = args.metadata_file
    metadata_df = pd.read_csv(metadata_file)

    raw_data_path = args.base_path
    plot = args.plot

    # fill the patient dict variable
    patient_dict = {}

    # Method 2: Using itertuples() (More efficient than iterrows())
    for row in metadata_df.itertuples():
        patient_id = getattr(row, '_5') # unique Name
        file_type = getattr(row, 'Modality') # check if RTSTRUCT or CT
        file_location = getattr(row, '_16') # Path to the images
        if patient_id in patient_dict.keys() :
            patient_dict[patient_id][file_type] = file_location
        else:
            patient_dict[patient_id] = {}
            patient_dict[patient_id][file_type] = file_location

    for k,v in patient_dict.items():
        print(Path(raw_data_path).joinpath(v["CT"]))
        preprocess_case(
            case_id = k,
            ct_path = Path(raw_data_path).joinpath(v["CT"]),
            rtstruct_path = Path(raw_data_path).joinpath(v["RTSTRUCT"]).joinpath('1-1.dcm'),
            plot=plot
        )
if __name__ == "__main__":
    main()