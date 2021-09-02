import os

from util.variables import *


def get_model_metrics(acc_line, avg_line):
    acc = acc_line.split()[1]
    prec, rec, f1 = avg_line.split()[2:5]

    return acc, prec, rec, f1


def get_test_metrics(utt_acc_line, a_acc_line, b_acc_line):
    utt_acc = utt_acc_line.split()[3]
    a_acc = a_acc_line.split()[6]
    b_acc = b_acc_line.split()[6]

    return utt_acc, a_acc, b_acc


def get_model_names():
    if os.path.isdir(MODELS_PATH):
        _, dirnames, _ = next(os.walk(MODELS_PATH))
        return dirnames
    else:
        return []


def get_test_names(model_name):
    path = f"{TESTS_PATH}/{model_name}"
    if os.path.isdir(path):
        _, _, filenames = next(os.walk(path))
        return filenames
    else:
        return []


def get_model_vars(model):
    var_dict = {}
    method_name, vars_string = model.replace(',', '.').split('__', 1)
    var_dict[METHOD] = method_name

    var_dict.update(get_vars(vars_string))

    return var_dict


def get_vars(var_string):
    var_dict = {}
    for var in var_string.split('__'):
        _var = var.split('=')
        _var.extend([True] * (2 - len(_var)))

        name, value = _var
        var_dict[name] = value

    return var_dict
