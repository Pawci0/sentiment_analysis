import os
import pandas as pd

from util.variables import *
from visualization.functions import get_test_label

OUTPUT_DIR = f"{TABLES_PATH}/final_test_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("../output/agg_tests.csv")

for model_name, model_group in df.groupby(MODEL):
    for opt_name, opt_group in model_group.groupby(OPT):
        parsed_group = opt_group[[A_ACC, B_ACC]]

        model_col = opt_group.apply(lambda r: get_test_label(r[RED], r[CONT], r[RED_WEIGHT]), axis=1)
        parsed_group.insert(loc=0, column=RED, value=model_col)

        parsed_group.to_csv(f"{OUTPUT_DIR}/{model_name}.csv", index=False)
