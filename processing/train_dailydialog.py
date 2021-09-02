import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

from model.dataset import get_dailydialog_dataloader
from model.functions import train_model, eval_model
from model.model import DialogueRNNModel, LSTMModel
from util.argument_parser import get_train_args
from util.functions import load_pkl_file
from util.logger import Logger
from util.variables import LSTM, DIALOGUERNN


def get_dataloaders(train_path, valid_path, test_path, batch_size=32):
    trainloader = get_dailydialog_dataloader(f"../data/DailyDialog/{train_path}/embeddings.pkl", batch_size)
    validloader = get_dailydialog_dataloader(f"../data/DailyDialog/{valid_path}/embeddings.pkl", batch_size)
    testloader = get_dailydialog_dataloader(f"../data/DailyDialog/{test_path}/embeddings.pkl", batch_size)

    return trainloader, validloader, testloader


def get_model(model,
              D_m, D_g, D_p, D_e, D_h,
              n_classes,
              rec_dropout, dropout,
              bidirectional):
    if model == LSTM:
        return LSTMModel(D_m, D_e, D_h,
                         n_classes=n_classes,
                         dropout=dropout,
                         bidirectional=bidirectional)
    elif model == DIALOGUERNN:
        return DialogueRNNModel(D_m, D_g, D_p, D_e, D_h,
                                n_classes=n_classes,
                                dropout_rec=rec_dropout,
                                dropout=dropout,
                                bidirectional=bidirectional)
    else:
        raise ValueError('No model chosen')


def load_class_weights(train_path='train'):
    return load_pkl_file(f"../data/DailyDialog/{train_path}/weights.pkl")


def get_loss_function(weighted_loss=False, train_path='train'):
    if weighted_loss:
        class_weights = load_class_weights(train_path).cuda()
        return nn.NLLLoss(weight=class_weights)
    else:
        return nn.NLLLoss()


def prepare_model_name(model_type, data_embeddings,
                       data_optimizer, n_epochs,
                       rec_dropout, dropout,
                       lr, l2,
                       weighted_loss, bidirectional):
    wl = "__wl" if weighted_loss else ""
    bi = "__bi" if bidirectional else ""
    name = f"{model_type}{bi}{wl}__ebd={data_embeddings}__opt={data_optimizer}__ep={n_epochs}__rd={rec_dropout}__d={dropout}__lr={lr}__l2={l2}"
    return name.replace('.', ',')


def prepare_output(name):
    out_dir = f"../output/models/{name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return f"{out_dir}/{name}.pth", f"{out_dir}/{name}.log"


def main(argv):
    args = get_train_args(argv)

    data_embeddings = args.embeddings
    data_optimizer = args.optimizer
    batch_size = args.batch_size
    n_epochs = args.epochs
    rec_dropout = args.rec_dropout
    dropout = args.dropout
    bidirectional = args.bidirectional
    model_type = args.model
    lr = args.lr
    l2 = args.l2
    weighted_loss = args.weighted_loss
    verbose = args.verbose

    train_path = f"train/{data_embeddings}/{data_optimizer}"
    test_path = f"test/{data_embeddings}"
    valid_path = f"validation/{data_embeddings}"

    n_classes = 7

    D_m = args.input
    D_g = 300
    D_p = 300
    D_e = 200
    D_h = 200

    model_name = prepare_model_name(model_type, data_embeddings,
                                    data_optimizer, n_epochs,
                                    rec_dropout, dropout,
                                    lr, l2,
                                    weighted_loss, bidirectional)
    model_save_file, model_log_file = prepare_output(model_name)
    sys.stdout = Logger(model_log_file, verbose)

    model = get_model(model_type, D_m, D_g, D_p, D_e, D_h, n_classes, rec_dropout, dropout, bidirectional)
    model.cuda()
    loss_function = get_loss_function(weighted_loss, train_path)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2)

    train_loader, valid_loader, test_loader = get_dataloaders(train_path, valid_path, test_path, batch_size)

    test_loss, test_label, test_pred = None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, train_fscore = train_model(model, loss_function, train_loader, optimizer)
        valid_loss, valid_acc, val_fscore, _, _ = eval_model(model, loss_function, valid_loader)
        test_loss, test_acc, test_fscore, test_label, test_pred = eval_model(model, loss_function, test_loader)

        output = f'''
Epoch {e + 1}/{n_epochs} (time elapsed {round(time.time() - start_time, 2)}s):
Training stats: train_loss {train_loss}, train_acc {train_acc} train_fscore {train_fscore}
Validation stats: valid_loss {valid_loss}, valid_acc {valid_acc}, valid_fscore {val_fscore}
Test stats: test_loss {test_loss}, test_acc {test_acc}, test_fscore {test_fscore}'''
        print(output)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\nClassification report:')
    print(classification_report(test_label, test_pred, labels=[0, 1, 2, 3, 4, 5, 6], digits=4, zero_division=0))

    print('\nConfussion matrix:')
    print(confusion_matrix(test_label, test_pred))

    torch.save(model, model_save_file)

    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main(sys.argv[1:])
