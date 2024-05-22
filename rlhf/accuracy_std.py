import math

import pandas as pd
import pudb

# df = pd.read_csv(
#     "test_results/io7m8cfg.csv",
#     names=["dataset", "prefix", "prompt", "text", "prediction", "target", "reward"],
# )

# # test_df = df.loc[
# #     ~((df["dataset"] == "positive_prompts") & (df["target"] == 1))
# #     & ~((df["dataset"] == "negative_prompts") & (df["target"] == 0))
# # ]

# test_df = df.loc[(df["dataset"] == "positive_prompts") & (df["target"] == 0)]
# print(len(test_df))

# accuracies = test_df["prediction"] == test_df["target"]

# std = accuracies.std()

# p = accuracies.sum() / len(accuracies)
# z = 1.960  # 95% confidence interval

# margin_of_error = z * (std / math.sqrt(len(accuracies)))

# print(margin_of_error)

# Perplexity
run_id = "io7m8cfg"
df = pd.read_csv(
    f"test_results/{run_id}_perplexity.csv",
    names=[
        "dataset",
        "prefix",
        "prompt",
        "text",
        "prediction",
        "target",
        "reward",
        "perplexity",
    ],
)
test_df = df.loc[
    ~((df["dataset"] == "positive_prompts") & (df["target"] == 1))
    & ~((df["dataset"] == "negative_prompts") & (df["target"] == 0))
]

perplexities = test_df.loc[(test_df["target"] == 1)]["perplexity"]

std = perplexities.std()

p = perplexities.sum() / len(perplexities)
z = 1.960  # 95% confidence interval

margin_of_error = z * (std / math.sqrt(len(perplexities)))
print(margin_of_error)
