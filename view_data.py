import pandas as pd

df = pd.read_csv("data/historical_races.csv")
print("📊 歷史賽馬資料總筆數:", len(df))
print("\n欄位名稱:")
print(df.columns.tolist())
print("\n前 10 筆資料:")
print(df.head(10).to_string(index=False))
print("\n統計摘要:")
print(df[['actual_weight', 'win_odds', 'is_top3']].describe())
