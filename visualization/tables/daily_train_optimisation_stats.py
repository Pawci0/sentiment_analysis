import pandas as pd

from util.functions import skip_zero, skip_mostly_zero, get_data_dir
from util.variables import TABLES_PATH, NONE, SKIP_ZERO, SKIP_MOSTLY_ZERO
from visualization.functions import get_class_count

type_to_optimizer = {
    NONE: lambda x: False,
    SKIP_ZERO: skip_zero,
    SKIP_MOSTLY_ZERO: skip_mostly_zero
}

labels_list = []
data_dir = get_data_dir(f"DailyDialog/train")
with open(data_dir + 'dialogues_emotion.txt', 'r', encoding='utf8') as label_file:
    label_lines = label_file.readlines()

    for label_line in label_lines:
        utt_labels = list(map(int, label_line.rstrip().split(' ')))
        labels_list.append(utt_labels)

stats = []
weights = []
for opt in [NONE, SKIP_ZERO, SKIP_MOSTLY_ZERO]:
    filtered_labels_list = list(filter(lambda labels: not type_to_optimizer[opt](labels), labels_list))
    class_count = get_class_count(filtered_labels_list)
    stats.append([opt] + class_count)

    min_count = min(class_count)
    weights.append([opt] + [round(min_count / c, 3) for c in class_count])

df_stats = pd.DataFrame(stats,
                        columns=['optimisation', 'no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness',
                                 'surprise'])
df_stats.to_csv(f"{TABLES_PATH}/daily_train_optimisation_stats.csv", index=False)

df_weights = pd.DataFrame(weights,
                          columns=['optimisation', 'no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness',
                                   'surprise'])
df_weights.to_csv(f"{TABLES_PATH}/daily_train_optimisation_weights.csv", index=False)
