# %%
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# %%
p = "/home/valentin/workspaces/vfm-geom-xai/output/scorpion_analysis/loso_uncentered/h0-mini_scorpion_224px_0p5mpp_cls/loso_DP200_uncentered/predictions.csv"

df = pd.read_csv(p)

# %%

print(classification_report(df["true_label"], df["predicted_leace"]))

labels = sorted(df["true_label"].unique())
cm = confusion_matrix(df["true_label"], df["predicted_leace"], labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)
# %%
