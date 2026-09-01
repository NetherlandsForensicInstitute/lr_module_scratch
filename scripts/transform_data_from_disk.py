import hashlib
import os
from pathlib import Path

import confidence
import pandas as pd

# These input files are used to generate the output files for each mark.
# The name of the files that are read are partial based on the input files.
RELEVANT_INPUT_FILES = ["h1_pairs", "h0_pairs", "h0_glocks"]

# These are the relevant marks and their corresponding headers for the output files.
# The name of the files that are read are partial based on the mark names.
RELEVANT_MARKS = {
    "breech_face_impression_mark": {
        "headers": [
            "Weapon1",
            "Weapon2",
            "Material1",
            "Material2",
            "Hypothesis",
            "Total Cells",
            "Matching Cells",
            "Status",
            "Error",
        ],
        "output_file": "breech_face_impression.csv",
    },
    "firing_pin_impression_mark": {
        "headers": [
            "Weapon1",
            "Weapon2",
            "Material1",
            "Material2",
            "Hypothesis",
            "Total Cells",
            "Matching Cells",
            "Status",
            "Error",
        ],
        "output_file": "firing_pin_impression.csv",
    },
    "aperture_shear_striation_mark": {
        "headers": [
            "Weapon1",
            "Weapon2",
            "Material1",
            "Material2",
            "Hypothesis",
            "Score",
            "Status",
            "Error",
        ],
        "output_file": "aperture_shear.csv",
    },
}


def get_and_hash_weapon_id(cartridge_id: str) -> str:
    """Get the weapon ID from the cartridge string and hash it to anonymize it."""
    weapon_id = "/".join(cartridge_id.split("/")[:-1])
    return hashlib.sha256(weapon_id.encode()).hexdigest()[:10]


def get_and_hash_weapon_type(cartridge_id: str) -> str:
    """Get the weapon type from the cartridge string and hash it to anonymize it."""
    weapon_type = cartridge_id.split("/")[1]
    return hashlib.sha256(weapon_type.encode()).hexdigest()[:4]


if __name__ == "__main__":
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    config = confidence.loadf(base_path / "local.yaml")
    disk_path = config.data_root_dir

    for mark in RELEVANT_MARKS.keys():
        output_file = RELEVANT_MARKS[mark]["output_file"]
        headers = RELEVANT_MARKS[mark]["headers"]

        output_file = Path(__file__).parent.parent / "data" / "2026" / f"{output_file}.csv"

        mark_df = pd.DataFrame()

        for input_file_bas in RELEVANT_INPUT_FILES:
            input_file = f"{disk_path}/{input_file_bas}_{mark}.csv"
            df = pd.read_csv(input_file, header=0, skiprows=0, names=headers, dtype=str)

            # If the first row has the same values as the headers, drop it
            if df.iloc[0].tolist() == headers:
                df = df.iloc[1:]

            mark_df = pd.concat([mark_df, df], ignore_index=True)

        mark_df = mark_df.drop(columns=["Status", "Error"], errors="ignore")
        mark_df = mark_df.drop_duplicates()
        mark_df = mark_df.dropna(how="any")

        mark_df["Type1"] = mark_df["Weapon1"].apply(get_and_hash_weapon_type)
        mark_df["Type2"] = mark_df["Weapon2"].apply(get_and_hash_weapon_type)

        mark_df["Weapon1"] = mark_df["Weapon1"].apply(get_and_hash_weapon_id)
        mark_df["Weapon2"] = mark_df["Weapon2"].apply(get_and_hash_weapon_id)

        mark_df.to_csv(output_file, index=False)
