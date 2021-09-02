import pandas as pd

from util.functions import get_data_dir
from util.variables import TABLES_PATH
from visualization.functions import get_class_count

labels = []
data_dir = get_data_dir("ScenarioSA")
with open(data_dir + 'dialogues_sentiment.txt', 'r', encoding='utf8') as label_file:
    label_lines = label_file.readlines()

    labels_list = []
    for label_line in label_lines:
        utt_labels = list(map(int, label_line.rstrip().split(' ')))
        labels_list.append(utt_labels)

    labels.append(['dataset'] + get_class_count(labels_list))

with open(data_dir + 'dialogues_final_sentiment.txt', 'r', encoding='utf8') as label_file:
    label_lines = label_file.readlines()

    a_labels, b_labels = [], []
    for label_line in label_lines:
        a_label, b_label = list(map(int, label_line.rstrip().split(' ')))
        a_labels.append([a_label])
        b_labels.append([b_label])

    labels.append(['A final sentiment'] + get_class_count(a_labels))
    labels.append(['B final sentiment'] + get_class_count(b_labels))

df = pd.DataFrame(labels, columns=['', 'neutral', 'positive', 'negative'])
df.to_csv(f"{TABLES_PATH}/scenario_stats.csv", index=False)
