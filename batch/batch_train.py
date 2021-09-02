from datetime import datetime

import processing.train_dailydialog
from util.variables import LSTM, DIALOGUERNN
from util.variables import MPNET, MINI
from util.variables import NONE, SKIP_ZERO, SKIP_MOSTLY_ZERO

models = [LSTM, DIALOGUERNN]
embeds = [MINI, MPNET]
opts = [NONE, SKIP_ZERO, SKIP_MOSTLY_ZERO]
bids = [['--bidirectional'], []]
losses = [['--weighted-loss'], []]

i = 0
for embed in embeds:
    for model in models:
        for opt in opts:
            for bid in bids:
                for loss in losses:
                    model_arg = ["--model", model]
                    epoch_arg = ["--epochs", "200"]
                    dropout_arg = ["--dropout", "0.5"]
                    lr_arg = ["--lr", "0.001"]
                    l2_arg = ["--l2", "0.00001"]
                    embed_arg = ["--embeddings", embed]
                    opt_arg = ["--optimizer", opt]
                    ver_arg = ["--verbose"]

                    arguments = model_arg + epoch_arg + dropout_arg + lr_arg + l2_arg + embed_arg + opt_arg + bid + loss + ver_arg

                    i += 1
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(f"[{current_time}] {i}/48: Running {arguments}")

                    processing.train_dailydialog.main(arguments)
