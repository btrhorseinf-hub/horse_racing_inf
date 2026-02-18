import pandas as pd

df = pd.read_csv("data/next_race.csv")
print("✅ 資料載入成功！")
print(f"總共 {len(df)} 匹馬")
print("\n前3筆：")
print(df.head(3))
