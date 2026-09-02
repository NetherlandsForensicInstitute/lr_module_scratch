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

FRONT_COLS = [
    "Type1",
    "Type2",
]
END_COLS = ["Matching Cells", "Total Cells", "Score"]


def get_weapon_id(cartridge_id: str) -> str:
    """Get the weapon ID from the cartridge string."""
    return "/".join(cartridge_id.split("/")[:-1])


def get_weapon_type(cartridge_id: str) -> str:
    """Get the weapon type from the cartridge string."""
    return cartridge_id.split("/")[1]


def hash_weapon_id(weapon_id: str) -> str:
    """Hash the weapon ID to anonymize it. Limit it to 10 characters for brevity."""
    return hashlib.sha256(weapon_id.encode()).hexdigest()[:10]


def hash_weapon_type(weapon_type: str) -> str:
    """Hash the weapon type to anonymize it. Limit it to 4 characters for brevity."""
    return hashlib.sha256(weapon_type.encode()).hexdigest()[:4]


if __name__ == "__main__":
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    config = confidence.loadf(base_path / "local.yaml")
    disk_path = config.data_from_disk_path

    for mark_name, mark_data in RELEVANT_MARKS.items():
        output_file_name = mark_data["output_file"]
        headers = mark_data["headers"]

        output_file = Path(__file__).parent.parent / "data" / "2026" / f"{output_file_name}"
        debug_output_file = Path(__file__).parent.parent / "data" / "2026" / f"debug_{output_file_name}"

        mark_df = pd.DataFrame()

        for base_input_filename in RELEVANT_INPUT_FILES:
            input_file = f"{disk_path}/{base_input_filename}_{mark_name}.csv"
            df = pd.read_csv(input_file, header=0, skiprows=0, names=headers, dtype=str)

            # If the first row has the same values as the headers, drop it
            if df.iloc[0].tolist() == headers:
                df = df.iloc[1:]

            mark_df = pd.concat([mark_df, df], ignore_index=True)

        mark_df = mark_df.drop(columns=["Status", "Error"], errors="ignore")
        mark_df = mark_df.drop_duplicates()
        mark_df = mark_df.dropna(how="any")

        # These columns are added only in the debug output file.
        # In the final output file, only the hashed values are kept.
        mark_df["UnhashedType1"] = mark_df["Weapon1"].apply(get_weapon_type)
        mark_df["UnhashedType2"] = mark_df["Weapon2"].apply(get_weapon_type)

        mark_df["UnhashedWeapon1"] = mark_df["Weapon1"].apply(get_weapon_id)
        mark_df["UnhashedWeapon2"] = mark_df["Weapon2"].apply(get_weapon_id)

        mark_df.to_csv(debug_output_file, index=False)

        mark_df["Type1"] = mark_df["UnhashedWeapon1"].apply(hash_weapon_type)
        mark_df["Type2"] = mark_df["UnhashedWeapon2"].apply(hash_weapon_type)

        mark_df["Weapon1"] = mark_df["UnhashedWeapon1"].apply(hash_weapon_id)
        mark_df["Weapon2"] = mark_df["UnhashedWeapon2"].apply(hash_weapon_id)

        mark_df = mark_df.drop(columns=["UnhashedType1", "UnhashedType2", "UnhashedWeapon1", "UnhashedWeapon2"])

        # Ensure the FRONT_COLS are at the front and END_COLS are at the end, with the rest of the columns in between.
        # It could be that some of the FRONT_COLS or END_COLS are not present in the DataFrame, so we filter them first.
        front_cols = [col for col in FRONT_COLS if col in mark_df.columns]
        end_cols = [col for col in END_COLS if col in mark_df.columns]
        middle_cols = [col for col in mark_df.columns if col not in front_cols + end_cols]

        mark_df = mark_df[front_cols + middle_cols + end_cols]

        mark_df.to_csv(output_file, index=False)
