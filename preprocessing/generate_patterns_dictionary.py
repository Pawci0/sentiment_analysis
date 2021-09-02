from util.functions import get_data_dir, save_pkl_file


def get_patterns_from_dataset(patterns_dict, dataset):
    data_dir = get_data_dir(f"DailyDialog/{dataset}")
    with open(data_dir + 'dialogues_emotion.txt', 'r', encoding='utf8') as label_file:
        label_lines = label_file.readlines()

        for label_line in label_lines:
            utt_labels = list(map(int, label_line.rstrip().split(' ')))
            for i in range(len(utt_labels) - 2):
                pattern = tuple(utt_labels[i:i + 3])

                if pattern in patterns_dict:
                    patterns_dict[pattern] = patterns_dict[pattern] + 1
                else:
                    patterns_dict[pattern] = 1


patterns_dict = {}
get_patterns_from_dataset(patterns_dict, 'train')
get_patterns_from_dataset(patterns_dict, 'test')
get_patterns_from_dataset(patterns_dict, 'validation')

sorted_patterns = sorted(patterns_dict, key=patterns_dict.get, reverse=True)

dict = {}
for one, two, three in sorted_patterns:
    key = (one, two)
    if key not in dict:
        dict[key] = three

save_pkl_file(dict, get_data_dir("DailyDialog") + 'patterns.pkl')
