import torch
from sentence_transformers import SentenceTransformer

from util.argument_parser import get_embed_args
from util.functions import skip_zero, skip_mostly_zero, get_data_dir, get_output_dir, save_pkl_file
from util.variables import NONE, SKIP_ZERO, SKIP_MOSTLY_ZERO

type_to_optimizer = {
    NONE: lambda x: False,
    SKIP_ZERO: skip_zero,
    SKIP_MOSTLY_ZERO: skip_mostly_zero
}

args = get_embed_args()

transformer_type = args.transformer_type
transformer = args.transformer
dataset_name = args.dataset
optimizer_type = args.optimizer
train = args.train

should_optimize = type_to_optimizer[optimizer_type]
model = SentenceTransformer(transformer)

data_dir = get_data_dir(dataset_name)
output_dir = get_output_dir(data_dir, transformer_type, optimizer_type if train else None)

with open(data_dir + 'dialogues_text.txt', 'r', encoding='utf8') as conv_file, \
        open(data_dir + 'dialogues_emotion.txt', 'r', encoding='utf8') as label_file:
    print(f"Processing {data_dir} started:")

    conv_lines = conv_file.readlines()
    label_lines = label_file.readlines()

    embeddings, labels, all_labels_vector = [], [], []
    i = len(conv_lines)
    for conv_line, label_line in zip(conv_lines, label_lines):
        print(f"{i} remaining")
        i -= 1

        utterances = conv_line.rstrip().split(' __eou__ ')
        utt_labels = list(map(int, label_line.rstrip().split(' ')))

        if should_optimize(utt_labels):
            continue

        embeddings.append(model.encode(utterances, convert_to_tensor=True))
        labels.append(torch.tensor(utt_labels))
        all_labels_vector += utt_labels

    save_pkl_file([embeddings, labels], output_dir + 'embeddings.pkl')

    if train is True:
        all_labels_vector = torch.tensor(all_labels_vector)
        class_count = all_labels_vector.bincount()
        class_weights = class_count.min() / class_count

        save_pkl_file(class_weights, output_dir + 'weights.pkl')

    print(f"Processing {data_dir} done!")
