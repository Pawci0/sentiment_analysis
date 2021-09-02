import os
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

EOU = '__eou__'


def parse_words(words):
    parsed = []
    last_word = EOU
    for word in words:
        if last_word == EOU and word == EOU:
            parsed.append('.')
        if word not in stop_words:
            parsed.append(word)
            last_word = word

    return parsed


def parse_lines(lines):
    parsed_lines = []
    i = len(lines)
    for line in lines:
        print(f"{i} remaining")
        i -= 1
        words = parse_words(line.lower().split(' ')[:-1])
        parsed_lines.append(' '.join(words))

    return parsed_lines


def process_dialogues(in_dir, out_dir):
    print(f"Processing {in_dir} started:")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(in_dir + 'dialogues_text.txt', 'r', encoding="utf8") as f:
        file_lines = f.readlines()
        new_lines = parse_lines(file_lines)

    with open(out_dir + 'dialogues_text.txt', 'w', encoding="utf8") as f:
        f.write('\n'.join(new_lines))
    print(f"Processing {in_dir} done!")


raw_dir = '../data/DailyDialog-raw'
processed_dir = '../data/DailyDialog'
train_dir = 'train'
test_dir = 'test'
valid_dir = 'validation'

process_dialogues(f"{raw_dir}/{train_dir}/", f"{processed_dir}/{train_dir}/")
process_dialogues(f"{raw_dir}/{test_dir}/", f"{processed_dir}/{test_dir}/")
process_dialogues(f"{raw_dir}/{valid_dir}/", f"{processed_dir}/{valid_dir}/")
