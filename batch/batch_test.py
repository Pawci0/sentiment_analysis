from datetime import datetime

import processing.test_scenariosa
from batch.functions import get_model_names, get_model_vars
from util.variables import *

models = get_model_names()
reducers = [LAST, MOST, WEIGHT, PATTERN]
contexts = [['--context'], []]
weights = ["0.6", "0.7", "0.8", "0.9", "1", "1.1", "1.2", "1.3", "1.4"]

i = 0
for model in models:
    vars = get_model_vars(model)
    embed_arg = ["--embeddings", vars[EBD_VAR]]
    for context in contexts:
        for reducer in reducers:
            for weight in (weights if reducer == WEIGHT else [1]):
                model_arg = ["--model-name", model]
                reducer_arg = ["--reducer", reducer]

                weight_arg = ["--weight", weight] if reducer == WEIGHT else []

                arguments = model_arg + embed_arg + context + weight_arg + reducer_arg

                i += 1
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"[{current_time}] [{i}/1152] Running {arguments}")

                processing.test_scenariosa.main(arguments)
