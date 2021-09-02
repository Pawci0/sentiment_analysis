import numpy as np
import os
import sys
import torch
from sklearn.metrics import confusion_matrix, classification_report

from model.dataset import get_scenariosa_dataloader
from model.functions import get_predictions, calculate_utterance_level_accuracy
from model.reduction import SentimentReducer, last_label, most_common_label, weighted_most_common_label, \
    label_based_on_emotion_pattern
from util.argument_parser import get_test_args
from util.logger import Logger
from util.variables import LAST, MOST, WEIGHT, PATTERN

emotion_to_sentiment = {
    0: 0,
    1: -1,
    2: -1,
    3: -1,
    4: 1,
    5: -1,
    6: 0
}

reducer_type_to_method = {
    LAST: last_label,
    MOST: most_common_label,
    WEIGHT: weighted_most_common_label,
    PATTERN: label_based_on_emotion_pattern()
}


def get_model(name):
    model_path = f"../output/models/{name}/{name}.pth"
    return torch.load(model_path)


def get_test_name(reducer, context, weight):
    cont = "__cont" if context else ""
    w = f"__w={weight}" if weight else ""
    name = f"red={reducer}{cont}{w}"

    return name.replace('.', ',')


def prepare_output(model_name, output_name):
    out_dir = f"../output/tests/{model_name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return f"{out_dir}/{output_name}.log"


def get_reducer(reducer_type, reducer_context, reducer_weight_coef):
    return SentimentReducer(reducer_type_to_method[reducer_type], reducer_context, reducer_weight_coef)


def get_dataloader(path, batch_size):
    return get_scenariosa_dataloader(f"../data/ScenarioSA/{path}/embeddings.pkl", batch_size)


def get_flat_utt_level_sentiment(emotion_preds, labels):
    emotion_preds_flat = np.concatenate(emotion_preds)
    labels_flat = np.concatenate(labels)

    sentiment_preds_flat = list(map(lambda e: emotion_to_sentiment[e], emotion_preds_flat))

    return sentiment_preds_flat, labels_flat


def accuracy(acc_list):
    return round(sum(ac for ac in acc_list) / len(acc_list), 4)


def final_sentiment_accuracies(emotion_preds, final_labels, reducer):
    s0_acc, s1_acc = [], []
    for preds, labels in zip(emotion_preds, final_labels):
        s0_label, s1_label = labels
        s0_emotion_pred, s1_emotion_pred = reducer(preds)
        s0_sent_pred, s1_sent_pred = emotion_to_sentiment[s0_emotion_pred], emotion_to_sentiment[s1_emotion_pred]

        s0_acc.append(s0_sent_pred == s0_label.item())
        s1_acc.append(s1_sent_pred == s1_label.item())

    return accuracy(s0_acc), accuracy(s1_acc)


def main(argv):
    args = get_test_args(argv)

    model_name = args.model_name
    data_embeddings = args.embeddings
    batch_size = args.batch_size
    reducer_type = args.reducer
    reducer_context = args.context
    reducer_weight_coef = args.weight
    verbose = args.verbose

    reducer = get_reducer(reducer_type, reducer_context, reducer_weight_coef)

    model = get_model(model_name)
    model.cuda()

    output_file_name = get_test_name(reducer_type, reducer_context, reducer_weight_coef)
    output_log_file = prepare_output(model_name, output_file_name)
    sys.stdout = Logger(output_log_file, verbose)

    test_loader = get_dataloader(data_embeddings, batch_size)

    emotion_preds, labels, final_labels = get_predictions(model, test_loader)
    utt_pred_flat, utt_label_flat = get_flat_utt_level_sentiment(emotion_preds, labels)

    utt_acc = calculate_utterance_level_accuracy(utt_pred_flat, utt_label_flat)
    s0_final_acc, s1_final_acc = final_sentiment_accuracies(emotion_preds, final_labels, reducer)

    output = f'''
Utterance level accuracy: {utt_acc}
Final sentiment accuracy for speaker A: {s0_final_acc}
Final sentiment accuracy for speaker B: {s1_final_acc}
'''
    print(output)

    print('\nUtterance level classification report:')
    print(classification_report(utt_label_flat, utt_pred_flat, labels=[-1, 0, 1], digits=4, zero_division=0))

    print('\nUtterance level confussion matrix:')
    print(confusion_matrix(utt_label_flat, utt_pred_flat))

    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main(sys.argv[1:])
