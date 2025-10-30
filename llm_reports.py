import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results/eval_results.csv")
means = df.groupby("model")["semantic_score"].mean()
means.plot(kind="bar", title="Average Semantic Score by Model")
plt.ylabel("Semantic Score")
plt.show()
