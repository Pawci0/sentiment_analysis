import torch

from model.functions import load_pattern_dict


class SentimentReducer:

    def __init__(self, reduce_fn, other_speaker_context=False, weight_coef=None):
        self.other_speaker_context = other_speaker_context
        self.weight_coef = weight_coef
        self.reduce_fn = reduce_fn

    def __call__(self, utterances):
        s0_utt, s1_utt = self.get_speakers_utt(utterances)
        if self.weight_coef is not None:
            s0_w, s1_w = self.get_weights(s0_utt), self.get_weights(s1_utt)
            return self.reduce_fn(s0_utt, s0_w), self.reduce_fn(s1_utt, s1_w)
        else:
            return self.reduce_fn(s0_utt), self.reduce_fn(s1_utt)

    def get_weights(self, utterances):
        return torch.tensor([self.weight_coef ** i for i in range(len(utterances))])

    def get_speakers_utt(self, utterances):
        if self.other_speaker_context:
            return self.get_speakers_utt_with_context(utterances)

        else:
            return utterances[::2], utterances[1::2]

    def get_speakers_utt_with_context(self, utterances):
        last_utt_speaker = (len(utterances) - 1) % 2
        if last_utt_speaker == 0:
            return utterances, utterances[:-1]
        else:
            return utterances[:-1], utterances


class label_based_on_emotion_pattern:

    def __init__(self):
        self.pattern_dict = load_pattern_dict()
        self.pattern_size = len(list(self.pattern_dict.keys())[0])

    def __call__(self, labels, _=None):
        if labels.size()[0] > self.pattern_size:
            pattern = tuple(labels[-self.pattern_size:].tolist())
            if pattern in self.pattern_dict:
                return self.pattern_dict[pattern]
            else:
                return last_label(labels)
        else:
            return last_label(labels)


def most_common_label(labels, _=None):
    return torch.mode(labels).values.item()


def weighted_most_common_label(labels, weights):
    return torch.argmax(torch.bincount(labels, weights)).item()


def last_label(labels, _=None):
    return labels[-1].item()
