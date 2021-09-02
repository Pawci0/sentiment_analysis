import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from util.functions import load_pkl_file, get_data_dir


def flatten(probabilities, targets, sizes):
    predictions_flat = torch.zeros(0).cuda()
    targets_flat = torch.zeros(0, dtype=torch.int32).cuda()

    for pred, target, size in zip(probabilities, targets, sizes):
        predictions_flat = torch.cat([predictions_flat, pred[:size].cuda()])
        targets_flat = torch.cat([targets_flat.cuda(), target.cuda()])

    return predictions_flat, targets_flat


def train_model(model, loss_function, dataloader, optimizer):
    model.train()

    losses, preds, labels = [], [], []
    for data in dataloader:
        optimizer.zero_grad()
        batch_preds, batch_labels, loss = process_batch(model, data, loss_function)

        preds.append(batch_preds)
        labels.append(batch_labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    avg_accuracy, avg_fscore, avg_loss = calculate_scores(labels, losses, preds)

    return avg_loss, avg_accuracy, avg_fscore


def eval_model(model, loss_function, dataloader):
    model.eval()

    losses, preds, labels = [], [], []
    for data in dataloader:
        batch_preds, batch_labels, loss = process_batch(model, data, loss_function)

        preds.append(batch_preds)
        labels.append(batch_labels)
        losses.append(loss.item())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    avg_accuracy, avg_fscore, avg_loss = calculate_scores(labels, losses, preds)

    return avg_loss, avg_accuracy, avg_fscore, labels, preds


def process_batch(model, data, loss_function):
    batch_sequence, batch_labels, batch_sizes = data

    log_prob = model(batch_sequence, batch_sizes)

    prob_flat, labels_flat = flatten(log_prob.transpose(0, 1), batch_labels, batch_sizes)
    batch_preds = torch.argmax(prob_flat, 1)
    loss = loss_function(prob_flat, labels_flat)

    return batch_preds.cpu().numpy(), labels_flat.cpu().numpy(), loss


def get_predictions(model, dataloader):
    model.eval()

    predictions_ = []
    labels_ = []
    final_labels_ = []
    for data in dataloader:
        batch_sequence, batch_labels, final_labels, batch_sizes = data

        log_prob = model(batch_sequence, batch_sizes)

        for prob, labels, size, final in zip(log_prob.transpose(0, 1), batch_labels, batch_sizes, final_labels):
            pred = torch.argmax(prob, 1)[:size]

            predictions_.append(pred.cpu())
            labels_.append(labels.cpu())
            final_labels_.append(final.cpu())

    return predictions_, labels_, final_labels_


def calculate_scores(labels, losses, preds):
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = calculate_utterance_level_accuracy(preds, labels)
    avg_fscore = round(f1_score(labels, preds, average='micro', labels=[0, 1, 2, 3, 4, 5, 6]), 4)

    return avg_accuracy, avg_fscore, avg_loss


def calculate_utterance_level_accuracy(preds, labels):
    return round(accuracy_score(labels, preds), 4)


def load_pattern_dict():
    return load_pkl_file(get_data_dir('DailyDialog') + 'patterns.pkl')
