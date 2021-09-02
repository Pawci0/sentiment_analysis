import itertools as it
import matplotlib.pyplot as plt
import pandas as pd

from batch.functions import *
from util.variables import *
from visualization.functions import get_model_label_from_vars, method_to_label

OPTIMAL_EPOCH = 60

for method in [DIALOGUERNN, LSTM]:
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=True, figsize=(20, 15))
    title = f"Accuracy throughout learning epochs for {method_to_label[method]} method"
    fig.suptitle(title, fontsize=25, fontweight='bold', y=0.995)
    fig.tight_layout(pad=4)

    (none_train_acc, zero_train_acc, mostly_train_acc), (none_valid_acc, zero_valid_acc, mostly_valid_acc) = axes

    method_to_axes = {
        NONE: (none_train_acc, none_valid_acc),
        SKIP_ZERO: (zero_train_acc, zero_valid_acc),
        SKIP_MOSTLY_ZERO: (mostly_train_acc, mostly_valid_acc)
    }

    for model in get_model_names():
        df = pd.read_csv(f"{MODELS_PATH}/{model}/{model}.csv")
        df.attrs = get_model_vars(model)
        if df.attrs[METHOD] == method:
            train_acc, valid_acc = method_to_axes[df.attrs[OPT_VAR]]
            train_acc.plot(df[TRAIN_ACC], label=get_model_label_from_vars(df.attrs))
            valid_acc.plot(df[VALID_ACC], label=get_model_label_from_vars(df.attrs))

    none_train_acc.set_title(f"Training accuracy (opt: {NONE})", y=-0.08, fontweight='bold')
    zero_train_acc.set_title(f"Training accuracy (opt: {SKIP_ZERO})", y=-0.08, fontweight='bold')
    mostly_train_acc.set_title(f"Training accuracy (opt: {SKIP_MOSTLY_ZERO})", y=-0.08, fontweight='bold')
    none_valid_acc.set_title(f"Validation accuracy (opt: {NONE})", y=-0.08, fontweight='bold')
    zero_valid_acc.set_title(f"Validation accuracy (opt: {SKIP_ZERO})", y=-0.08, fontweight='bold')
    mostly_valid_acc.set_title(f"Validation accuracy (opt: {SKIP_MOSTLY_ZERO})", y=-0.08, fontweight='bold')

    for ax in it.chain.from_iterable(axes):
        ax.axvline(x=OPTIMAL_EPOCH, color='red', linestyle='dashed', label=f"{OPTIMAL_EPOCH} epochs")

    mostly_valid_acc.legend(loc='lower right')

plt.show()
