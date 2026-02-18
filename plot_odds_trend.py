import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# 讀取資料
file_path = "data/historical_races.csv"
if not os.path.exists(file_path):
    print("❌ 找不到歷史資料檔案！請先執行 generate_sample_data.py")
    exit()

df = pd.read_csv(file_path)
df['race_date'] = pd.to_datetime(df['race_date'])
df = df.sort_values('race_date').reset_index(drop=True)

# 列出所有馬匹
horses = sorted(df['horse_name'].unique())
print("🐎 可選馬匹：")
for i, h in enumerate(horses, 1):
    count = len(df[df['horse_name'] == h])
    print(f"{i}. {h} ({count} 場)")

# 選擇馬匹
try:
    choice = int(input("\n請輸入馬匹編號: ")) - 1
    selected_horse = horses[choice]
except (ValueError, IndexError):
    print("⚠️ 輸入無效，預設使用第一匹馬")
    selected_horse = horses[0]

# 過濾該馬資料
horse_df = df[df['horse_name'] == selected_horse].copy()
horse_df = horse_df.sort_values('race_date')

if horse_df.empty:
    print(f"❌ 找不到 {selected_horse} 的記錄")
    exit()

# 繪圖
plt.figure(figsize=(10, 6))
ax = plt.gca()

# 賠率是「越低越好」，所以 y 軸反向
plt.gca().invert_yaxis()

# 區分入位與否
top3 = horse_df[horse_df['is_top3'] == 1]
not_top3 = horse_df[horse_df['is_top3'] == 0]

plt.plot(horse_df['race_date'], horse_df['win_odds'], 
         marker='o', linestyle='-', color='lightgray', zorder=1)

plt.scatter(top3['race_date'], top3['win_odds'], 
            color='green', s=100, label='✅ 入位', zorder=2, edgecolor='black')
plt.scatter(not_top3['race_date'], not_top3['win_odds'], 
            color='red', s=100, label='❌ 未入位', zorder=2, edgecolor='black')

# 標註賠率數值
for _, row in horse_df.iterrows():
    plt.text(row['race_date'], row['win_odds'] + 0.3, 
             f"{row['win_odds']:.1f}", 
             ha='center', va='bottom', fontsize=9)

plt.title(f"🏇 {selected_horse} — 獨贏賠率走勢圖", fontsize=16)
plt.xlabel("比賽日期")
plt.ylabel("獨贏賠率（數值越低表示越被看好）")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 格式化 x 軸日期
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()

# 儲存圖片
output_file = f"plots/{selected_horse}_odds_trend.png"
os.makedirs("plots", exist_ok=True)
plt.savefig(output_file, dpi=150)
print(f"\n✅ 圖表已儲存至: {output_file}")

# 顯示圖表（在 Codespace 中可能不彈出，但可下載）
plt.show()
