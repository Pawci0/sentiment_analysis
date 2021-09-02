import matplotlib.pyplot as plt
import pandas as pd
from textwrap import wrap

from util.variables import *
from visualization.functions import get_model_label, get_test_label

LABEL = 'label'


def get_label(r):
    short_label = get_model_label(r[METHOD], r[BI], r[WL], r[EBD]) + ", " + get_test_label(r[RED], r[CONT],
                                                                                           r[RED_WEIGHT])
    return '\n'.join(wrap(short_label, 25))


def create_top_tests_graph(df, columns, title):
    sorted_df = df.sort_values(columns, ascending=True).tail(8)
    ax = sorted_df.plot(x=LABEL, y=[A_ACC, B_ACC, UTT_ACC], kind="barh", linewidth=10, figsize=(7, 8))

    ax.set_xlim(0.4, 0.9)
    ax.set_ylabel("")
    ax.set_title(title)

    ax.legend(loc='lower right')


df = pd.read_csv("../output/agg_tests.csv")

parsed_df = df[[MODEL, A_ACC, B_ACC, UTT_ACC]]

short_label_col = df.apply(lambda r: get_label(r), axis=1)
parsed_df.insert(loc=0, column=LABEL, value=short_label_col)

create_top_tests_graph(parsed_df, [A_ACC, B_ACC, UTT_ACC], "Highest accuracy for speaker A")
create_top_tests_graph(parsed_df, [B_ACC, A_ACC, UTT_ACC], "Highest accuracy for speaker B")
create_top_tests_graph(parsed_df, [UTT_ACC, A_ACC, B_ACC], "Highest utterance level accuracy")

plt.show()
