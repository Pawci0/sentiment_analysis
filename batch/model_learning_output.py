import pandas as pd

from batch.functions import *
from util.variables import *


def get_stats(epoch_line):
    stats = epoch_line.replace(',', '').split()
    loss = stats[3]
    acc = stats[5]
    fscore = stats[7]
    return loss, acc, fscore


for model in get_model_names():
    with open(f"{MODELS_PATH}/{model}/{model}.log") as f:
        lines = f.readlines()
        vars = get_model_vars(model)
        epochs = int(vars.get(EP_VAR))

        last_line = (5 * epochs)
        train_stat_lines = lines[2:last_line:5]
        valid_stat_lines = lines[4:last_line:5]
        epoch_stats = [get_stats(train) + get_stats(valid) for train, valid in zip(train_stat_lines, valid_stat_lines)]

        df = pd.DataFrame(epoch_stats, columns=[TRAIN_LOSS, TRAIN_ACC, TRAIN_F1, VALID_LOSS, VALID_ACC, VALID_F1])
        df.to_csv(f"{MODELS_PATH}/{model}/{model}.csv", index=False)
