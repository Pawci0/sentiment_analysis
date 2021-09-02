import itertools
import numpy as np

from util.variables import *

method_to_label = {
    DIALOGUERNN: "DialogueRNN",
    LSTM: "LSTM",
}

reduction_to_label = {
    LAST: "Last label",
    MOST: "Most common label",
    WEIGHT: "Weighted most common label",
    PATTERN: "Label based on common emotion patterns"
}


def get_matrix_line(matrix_line):
    parsed_line = matrix_line.translate(str.maketrans({'[': '', ']': ''}))
    arr = np.array(parsed_line.split(), dtype=int)
    return arr / arr.sum()


def get_matrix(matrix_lines):
    return [get_matrix_line(matrix_line) for matrix_line in matrix_lines]


def get_model_label(method, bid, weigh, ebd):
    method = method_to_label[method]
    bidirectional = "Bi" if bid else ""
    weighted = f" + Weighted Loss" if weigh else ""

    return f"{bidirectional}{method}{weighted} ({ebd})"


def get_model_label_from_vars(model_vars):
    return get_model_label(model_vars[METHOD],
                           model_vars.get(BI_VAR, False),
                           model_vars.get(WL_VAR, False),
                           model_vars[EBD_VAR])


def get_test_label(red, cont, weight):
    reduction = reduction_to_label[red]
    context = " with context" if cont else ""
    weighted = f" (γ={weight.replace(',', '.')})" if red == WEIGHT else ""

    return f"{reduction}{context}{weighted}"


def get_class_count(labels_list):
    labels = list(itertools.chain.from_iterable(labels_list))
    return [labels.count(x) for x in set(labels)]
