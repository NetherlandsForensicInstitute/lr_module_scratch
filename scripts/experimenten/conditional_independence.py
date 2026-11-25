from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#  .../lr_module_scratch/ has to be the working directory further on

bf_scores_path = "data/2026-Q4/all_types/breech_face_impression.csv"
fp_scores_path = "data/2026-Q4/all_types/firing_pin_impression.csv"
as_scores_path = "data/2026-Q4/all_types/aperture_shear.csv"

bf_scores = pd.read_csv(bf_scores_path)
fp_scores = pd.read_csv(fp_scores_path)
as_scores = pd.read_csv(as_scores_path)

bf_lrs_path = "output/all_types_independence/bf_cmc/modules.csv"
fp_lrs_path = "output/all_types_independence/fp_cmc/modules.csv"
as_lrs_path = "output/all_types_independence/as_ccf/modules.csv"

bf_lrs = pd.read_csv(bf_lrs_path)
fp_lrs = pd.read_csv(fp_lrs_path)
as_lrs = pd.read_csv(as_lrs_path)

output_folder = Path("output/all_types_independence")
output_folder.mkdir(exist_ok=True)

bf_fp_scores_merged = pd.merge(bf_scores, fp_scores, on=['Weapon1', 'Weapon2'], how='inner')
bf_as_scores_merged = pd.merge(bf_scores, as_scores, on=['Weapon1', 'Weapon2'], how='inner')
fp_as_scores_merged = pd.merge(fp_scores, as_scores, on=['Weapon1', 'Weapon2'], how='inner')

a = 1