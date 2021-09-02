import pandas as pd

from batch.functions import *
from util.variables import *

df = pd.DataFrame()
model_list, method_list, bi_list, wl_list, ebd_list = [], [], [], [], []
opt_list, ep_list, rd_list, d_list, lr_list, l2_list = [], [], [], [], [], []
acc_list, prec_list, rec_list, f1_list = [], [], [], []
red_list, cont_list, weight_list = [], [], []
utt_acc_list, a_acc_list, b_acc_list = [], [], []
for model in get_model_names():
    for test in get_test_names(model):
        with open(f"{TESTS_PATH}/{model}/{test}") as f:
            lines = f.readlines()
            acc, prec, rec, f1 = get_model_metrics(lines[13], lines[14])
            utt_acc, a_acc, b_acc = get_test_metrics(lines[1], lines[2], lines[3])

            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)

            utt_acc_list.append(utt_acc)
            a_acc_list.append(a_acc)
            b_acc_list.append(b_acc)

            model_vars = get_model_vars(model)

            model_list.append(model)
            method_list.append(model_vars.get(METHOD))
            bi_list.append(model_vars.get(BI_VAR, False))
            wl_list.append(model_vars.get(WL_VAR, False))
            ebd_list.append(model_vars.get(EBD_VAR))

            opt_list.append(model_vars.get(OPT_VAR))
            ep_list.append(model_vars.get(EP_VAR))
            rd_list.append(model_vars.get(RD_VAR))
            d_list.append(model_vars.get(D_VAR))
            lr_list.append(model_vars.get(LR_VAR))
            l2_list.append(model_vars.get(L2_VAR))

            test_vars = get_vars(test.split('.')[0])

            red_list.append(test_vars.get(RED_VAR))
            cont_list.append(test_vars.get(CONT_VAR, False))
            weight_list.append(test_vars.get(WEIGHT_VAR, '-'))

df[MODEL] = model_list
df[METHOD] = method_list
df[BI] = bi_list
df[WL] = wl_list
df[EBD] = ebd_list

df[ACC] = acc_list
df[PREC] = prec_list
df[REC] = rec_list
df[F1] = f1_list

df[OPT] = opt_list
df[EP] = ep_list
df[RD] = rd_list
df[D] = d_list
df[LR] = lr_list
df[L2] = l2_list

df[RED] = red_list
df[CONT] = cont_list
df[RED_WEIGHT] = weight_list

df[UTT_ACC] = utt_acc_list
df[A_ACC] = a_acc_list
df[B_ACC] = b_acc_list

df.to_csv("../output/agg_tests.csv", index=False)

print(df)
