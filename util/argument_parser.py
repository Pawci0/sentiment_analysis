import argparse

from util.variables import *


def get_train_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--embeddings', type=str, choices=[MINI, MPNET], default=MINI, metavar='M',
                        help=f"which embeddings to choose: {MINI} or {MPNET}")
    parser.add_argument('--optimizer', type=str, choices=[SKIP_ZERO, SKIP_MOSTLY_ZERO, NONE], default=NONE, metavar='M',
                        help=f"which training set optimizer to choose: {NONE}, {SKIP_ZERO}, {SKIP_MOSTLY_ZERO}")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of epochs')
    parser.add_argument('--input', type=int, default=-1, metavar='IS',
                        help='input sentence embedding size')
    parser.add_argument('--bidirectional', action='store_true',
                        help='is model bidirectional')
    parser.add_argument('--weighted-loss', action='store_true',
                        help='are class weights used in loss function')
    parser.add_argument('--model', type=str, choices=[LSTM, DIALOGUERNN], default=DIALOGUERNN, metavar='M',
                        help=f"DialogueRNN ({DIALOGUERNN}) or LSTM ({LSTM})")
    parser.add_argument('--verbose', action='store_true',
                        help='log output to terminal')

    parse_args = parser.parse_args(args)
    if parse_args.input == -1:
        parse_args.input = transformer_to_input_size[parse_args.embeddings]

    return parse_args


def get_test_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, metavar='M',
                        help='saved model name for testing')
    parser.add_argument('--embeddings', type=str, choices=[MINI, MPNET], default=MINI, metavar='M',
                        help=f"which embeddings to choose: {MINI} or {MPNET}")
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size')
    parser.add_argument('--reducer', type=str, choices=[LAST, MOST, WEIGHT, PATTERN], default=LAST, metavar='R',
                        help=f"method for final sentiment prediction: final utterance ({LAST}), most common ({MOST}), weighted most common ({WEIGHT}), common emotion pattern ({PATTERN})")
    parser.add_argument('--context', action='store_true',
                        help='use other speaker utterances for final sentiment prediction')
    parser.add_argument('--weight', type=float, default=None, metavar='W',
                        help='coef for weights generation (only for weighted reducer)')
    parser.add_argument('--verbose', action='store_true',
                        help='log output to terminal')

    return parser.parse_args(args)


def get_embed_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, metavar='D',
                        help='chosen dataset')
    parser.add_argument('--train', action='store_true',
                        help='is embedding done for training set - if so, optimisation methods can be applied')
    parser.add_argument('--optimizer', type=str, choices=[SKIP_ZERO, SKIP_MOSTLY_ZERO, NONE], default=NONE, metavar='M',
                        help='use optimizer for balancing imbalanced dataset')
    parser.add_argument('--transformer', type=str, choices=[MINI, MPNET], default=MINI, metavar='M',
                        help=f"set embedding transformer: paraphrase-MiniLM-L6-v2 ({MINI}) or paraphrase-mpnet-base-v2 ({MPNET})")

    args = parser.parse_args()

    args.transformer_type = args.transformer
    args.transformer = type_to_transformer[args.transformer_type]

    return args
