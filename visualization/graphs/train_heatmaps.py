import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from batch.functions import *
from util.variables import *
from visualization.functions import get_model_label_from_vars, get_matrix

sn.set(font_scale=0.5)
opts = [NONE, SKIP_ZERO, SKIP_MOSTLY_ZERO]

opt_to_title = {
    NONE: "",
    SKIP_ZERO: "",
    SKIP_MOSTLY_ZERO: ""
}

values = []
for model in get_model_names():
    model_vars = get_model_vars(model)
    with open(f"{MODELS_PATH}/{model}/{model}.log") as f:
        lines = f.readlines()
        matrix = get_matrix(lines[-7:])

        values.append((model_vars, matrix))

for opt in opts:
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(20, 15))

    fig.suptitle(f"DailyDialog confusion matrix heatmap for {opt}", fontsize=25, fontweight='bold', y=0.995)
    fig.tight_layout(pad=4)
    ax_generator = it.chain.from_iterable(axes)

    for model_vars, matrix in values:
        if model_vars[OPT_VAR] == opt:
            ax = next(ax_generator)
            ax.set_title(get_model_label_from_vars(model_vars), fontsize=15, y=-0.1)
            sn.heatmap(np.asarray(matrix), ax=ax, annot=True, cmap=sn.color_palette("rocket_r", as_cmap=True))

plt.show()
