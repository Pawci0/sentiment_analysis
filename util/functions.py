import os
import pickle


def load_pkl_file(path):
    return pickle.load(open(path, 'rb'))


def save_pkl_file(object, path):
    pickle.dump(object, open(path, 'wb'))


def skip_zero(labels):
    return sum(labels) == 0


def skip_mostly_zero(labels):
    return labels.count(0) > len(labels) * 3 / 4


def get_data_dir(dataset_name):
    return f"../data/{dataset_name}/"


def get_output_dir(data_dir, transformer_type, optimizer_type=None):
    optimizer_suffix = f"/{optimizer_type}" if optimizer_type else ""
    output_dir = f"{data_dir}{transformer_type}{optimizer_suffix}/"
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    return output_dir
