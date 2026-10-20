"""サンプルCSV（合成データ）を作る。実在の人や組織のデータではない。"""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
n = 600
age = rng.integers(18, 75, n)
visits = rng.poisson(6, n)
minutes = np.round(rng.gamma(2.0, 12.0, n), 1)
plan = rng.choice(["free", "basic", "pro"], n, p=[.5, .35, .15])
region = rng.choice(["Kanto", "Kansai", "Chubu", "Kyushu", "Tohoku"], n)
support = rng.integers(0, 6, n)
logit = -0.2 + 0.7*support - 0.35*visits - 0.03*minutes + np.select([plan=="pro", plan=="basic"], [-1.0, -0.4], 0.0) + rng.normal(0, .6, n)
churn = np.where(rng.random(n) < 1/(1+np.exp(-logit)), "yes", "no")
spend = np.round(500 + 40*visits + 8*minutes + np.select([plan=="pro", plan=="basic"], [3000, 1200], 0) + rng.normal(0, 400, n))
df = pd.DataFrame(dict(customer_id=[f"C{i:04d}" for i in range(n)], age=age, region=region, plan=plan,
                       monthly_visits=visits, avg_minutes=minutes, support_tickets=support,
                       monthly_spend=spend, churned=churn))
for c in ["age", "avg_minutes", "region"]:
    df.loc[rng.random(n) < .04, c] = np.nan
df.to_csv("sample/sample_customers.csv", index=False)
print(df.head(), df.churned.value_counts(), sep="\n")
