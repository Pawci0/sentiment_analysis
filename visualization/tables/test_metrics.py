import os
import pandas as pd

from util.variables import *
from visualization.functions import get_model_label

OUTPUT_DIR = f"{TABLES_PATH}/test_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("../output/agg_tests.csv")
df = df.drop_duplicates(subset=MODEL)

for name, group in df.groupby(OPT):
    parsed_group = group[[ACC, PREC, REC, F1]]

    model_col = group.apply(lambda r: get_model_label(r[METHOD], r[BI], r[WL], r[EBD]), axis=1)
    parsed_group.insert(loc=0, column=MODEL, value=model_col)

    print(group.sort_values(ACC, ascending=False).head(1)[ACC].values[0])
    print(group.sort_values(PREC, ascending=False).head(1)[PREC].values[0])
    print(group.sort_values(REC, ascending=False).head(1)[REC].values[0])
    print(group.sort_values(F1, ascending=False).head(1)[F1].values[0])

    print(parsed_group.mean().round(4))
    parsed_group.to_csv(f"{OUTPUT_DIR}/{name}_table.csv", index=False)
