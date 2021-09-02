import pandas as pd

from batch.functions import *
from util.variables import *

df = pd.DataFrame()
model_list, method_list, bi_list, wl_list, ebd_list = [], [], [], [], []
opt_list, ep_list, rd_list, d_list, lr_list, l2_list = [], [], [], [], [], []
acc_list, prec_list, rec_list, f1_list = [], [], [], []
for model in get_model_names():
    with open(f"{MODELS_PATH}/{model}/{model}.log") as f:
        lines = f.readlines()
        acc, prec, rec, f1 = get_model_metrics(lines[-13], lines[-12])

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        vars = get_model_vars(model)

        model_list.append(model)
        method_list.append(vars.get(METHOD))
        bi_list.append(vars.get(BI_VAR, False))
        wl_list.append(vars.get(WL_VAR, False))
        ebd_list.append(vars.get(EBD_VAR))

        opt_list.append(vars.get(OPT_VAR))
        ep_list.append(vars.get(EP_VAR))
        rd_list.append(vars.get(RD_VAR))
        d_list.append(vars.get(D_VAR))
        lr_list.append(vars.get(LR_VAR))
        l2_list.append(vars.get(L2_VAR))

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

df.to_csv("../output/agg_models.csv", index=False)

print(df)
