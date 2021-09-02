import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from util.functions import load_pkl_file

EMBED_COL = 'embeddings'
EMOTIONS_COL = 'emotions'
SIZE_COL = 'size'
SENTIMENT_COL = 'sentiment'
FINAL_SENTIMENT_COL = 'final_sentiment'


class DailyDialogueDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.embeddings, self.labels = load_pkl_file(path)
        self.len = len(self.embeddings)

    def __getitem__(self, index):
        labels = self.labels[index]
        return self.embeddings[index], labels, len(labels)

    def __len__(self):
        return self.len


class DailyDialoguePadding:

    def __init__(self):
        pass

    def __call__(self, batch):
        dat = pd.DataFrame(batch, columns=[EMBED_COL, EMOTIONS_COL, SIZE_COL])

        return pad_sequence(dat[EMBED_COL].tolist()), dat[EMOTIONS_COL].tolist(), dat[SIZE_COL].tolist()


class ScenarioSADataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.embeddings, self.labels, self.final_labels = pickle.load(open(path, 'rb'))
        self.len = len(self.embeddings)

    def __getitem__(self, index):
        labels = self.labels[index]
        return self.embeddings[index], labels, self.final_labels[index], len(labels)

    def __len__(self):
        return self.len


class ScenarioSAPadding:

    def __init__(self):
        pass

    def __call__(self, batch):
        dat = pd.DataFrame(batch, columns=[EMBED_COL, SENTIMENT_COL, FINAL_SENTIMENT_COL, SIZE_COL])

        return pad_sequence(dat[EMBED_COL].tolist()), dat[SENTIMENT_COL].tolist(), dat[FINAL_SENTIMENT_COL].tolist(), \
               dat[SIZE_COL].tolist()


def get_dailydialog_dataloader(path, batch_size):
    dataset = DailyDialogueDataset(path)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=DailyDialoguePadding())


def get_scenariosa_dataloader(path, batch_size):
    dataset = ScenarioSADataset(path)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=ScenarioSAPadding())
