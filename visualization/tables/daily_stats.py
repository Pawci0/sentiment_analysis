import pandas as pd

from util.functions import get_data_dir
from util.variables import TABLES_PATH
from visualization.functions import get_class_count

stats = []
for dataset in ['train', 'test', 'validation']:
    data_dir = get_data_dir(f"DailyDialog/{dataset}")
    with open(data_dir + 'dialogues_emotion.txt', 'r', encoding='utf8') as label_file:
        label_lines = label_file.readlines()

        labels_list = []
        for label_line in label_lines:
            utt_labels = list(map(int, label_line.rstrip().split(' ')))
            labels_list.append(utt_labels)

        stats.append([dataset] + get_class_count(labels_list))

df = pd.DataFrame(stats,
                  columns=['dataset', 'no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])
df.to_csv(f"{TABLES_PATH}/daily_stats.csv", index=False)
