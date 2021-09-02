import torch
from sentence_transformers import SentenceTransformer

from util.argument_parser import get_embed_args
from util.functions import get_data_dir, get_output_dir, save_pkl_file

args = get_embed_args()

transformer_type = args.transformer_type
transformer = args.transformer
dataset_name = args.dataset
train = args.train

model = SentenceTransformer(transformer)

data_dir = get_data_dir(dataset_name)
output_dir = get_output_dir(data_dir, transformer_type)

with open(data_dir + 'dialogues_text.txt', 'r', encoding='utf8') as conv_file, \
        open(data_dir + 'dialogues_sentiment.txt', 'r', encoding='utf8') as label_file, \
        open(data_dir + 'dialogues_final_sentiment.txt', 'r', encoding='utf8') as final_label_file:
    print(f"Processing {data_dir} started:")

    conv_lines = conv_file.readlines()
    label_lines = label_file.readlines()
    final_label_lines = final_label_file.readlines()

    embeddings, labels, final_labels = [], [], []
    i = len(conv_lines)
    for conv_line, label_line, final_label_line in zip(conv_lines, label_lines, final_label_lines):
        print(f"{i} remaining")
        i -= 1

        utterances = conv_line.rstrip().split(' __eou__ ')
        utt_labels = list(map(int, label_line.rstrip().split(' ')))
        conv_final_labels = list(map(int, final_label_line.rstrip().split(' ')))

        embeddings.append(model.encode(utterances, convert_to_tensor=True))
        labels.append(torch.tensor(utt_labels))
        final_labels.append(torch.tensor(conv_final_labels))

    save_pkl_file([embeddings, labels, final_labels], output_dir + 'embeddings.pkl')

    print(f"Processing {data_dir} done!")
