import pandas as pd
from datetime import date, timedelta

start = date(2018, 11, 1)
end = date(2019, 12, 1)


metadata_df = pd.read_csv("./Data.csv", encoding="utf-8")

metadata_df["date"] = pd.to_datetime(metadata_df["date"])

metadata_df = pd.read_csv("./LDA_Distibution.csv", encoding="utf-8")

lda_df = lda_df.drop(columns=["account", "author", "censor", "date", "forprofit", "license", "official", "segs", "title", "week", "month"])

cols = ["_id"]

topics = ["Topic" + str(i) for i in range(1, 26)]

cols.extend(topics)

lda_df.columns = cols

now = start

dt = list()

value = list()

while True:
    if now >= end:
        break

    select_data = metadata_df.loc[metadata_df["date"] == pd.Timestamp(now)]

    ids = list(select_data["id"])

    dt_lda_df = lda_df[lda_df["_id"].isin(ids)]

    only_distribution = dt_lda_df.drop(columns=["_id"])

    mean = list(only_distribution.mean(axis=0))

    value.append(mean)

    dt.append(now)

    now += timedelta(days=1)

res = pd.DataFrame(value, index=dt)

res.columns = topics

res.to_csv("daily_distribution.csv", encoding="utf-8")