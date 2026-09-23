import pandas as pd
import shutil

# These files are needed to revert the hashed file names back to their original.
FULL_DATA_FILE_BF = "data/2026/debug_breech_face_impression.csv"
FULL_DATA_FILE_FP = "data/2026/debug_firing_pin_impression.csv"

# Run 'matlab_vs_python.yaml' without the ${timestamp} variable, so that the output is written to a fixed location.
BF_RESULTS = "output/breech_face_impression_matlab_python/python/modules.csv"
FP_RESULTS = "output/firing_pin_impression_matlab_python/python/modules.csv"

# Shared network folders. May need to be changed depending on the environment/OS.
FROM_FOLDER = "/mnt/scratch/Scratch/Te_converteren_databases"
TO_FOLDER = "/mnt/scratch/Scratch/misleading_glocks"

for type, full_data_file, results_file in [
    ("breech_face", FULL_DATA_FILE_BF, BF_RESULTS),
    ("firing_pin", FULL_DATA_FILE_FP, FP_RESULTS),
]:
    scores = pd.read_csv(full_data_file)
    llrs = pd.read_csv(results_file)

    llrs = llrs[(llrs["label"] == 1) & (llrs["bounding.0"] < 0)]
    scores = scores[scores["Hypothesis"] == 1]
    scores = scores[["HashedWeapon1", "Weapon1", "Weapon2", "Total Cells", "Matching Cells"]]
    scores = scores.drop_duplicates(subset=["HashedWeapon1"])

    llrs = llrs[["source_ids.0", "source_ids.1", "bounding.0"]]

    merged_df = pd.merge(llrs, scores, left_on="source_ids.0", right_on="HashedWeapon1", how="left")

    merged_df.to_csv(f"output/misleading_h1_{type}.csv")

    # Copy the source folders to the new location
    for w in ["Weapon1", "Weapon2"]:
        for weapon in merged_df[w].unique():
            # Copy the entire folder structure for the weapon to the new location
            src = f"{FROM_FOLDER}{weapon}"
            dst = f"{TO_FOLDER}{weapon}"
            shutil.copytree(src, dst, dirs_exist_ok=True)

            # Copy the db.scratch file to the new location
            folder = "/".join(weapon.split("/")[:-1]) + "/db.scratch"
            src = f"{FROM_FOLDER}{folder}"
            dst = f"{TO_FOLDER}{folder}"
            shutil.copyfile(src, dst)
