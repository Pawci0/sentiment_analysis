import os
import pandas as pd

from util.variables import *
from visualization.functions import get_model_label, get_test_label

MODEL_LABEL = 'model'
REDUCTION_LABEL = 'prediction function'


def save_df(df, title):
    output_dir = f"{TABLES_PATH}/best_tests"
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f"{output_dir}/{title}", index=False)


def create_top_tests_table(df, column, title):
    best_values = []
    for name, group in df.groupby(REDUCTION_LABEL):
        best_value = group.sort_values(column, ascending=False).head(1).squeeze()
        best_values.append(best_value)

    best_values_df = pd.DataFrame(best_values).sort_values(column, ascending=False)[
        [REDUCTION_LABEL, MODEL_LABEL, column]]
    save_df(best_values_df, title.replace(' ', '_'))


df = pd.read_csv("../output/agg_tests.csv")

parsed_df = df[[A_ACC, B_ACC]]

test_label_col = df.apply(lambda r: get_test_label(r[RED], r[CONT], r[RED_WEIGHT]), axis=1)
model_label_col = df.apply(lambda r: get_model_label(r[METHOD], r[BI], r[WL], r[EBD]), axis=1)
parsed_df.insert(loc=0, column=REDUCTION_LABEL, value=test_label_col)
parsed_df.insert(loc=1, column=MODEL_LABEL, value=model_label_col)

create_top_tests_table(parsed_df, A_ACC, "Highest accuracy for speaker A")
create_top_tests_table(parsed_df, B_ACC, "Highest accuracy for speaker B")
