# plot_interactive.py
import pandas as pd
import plotly.express as px
import os

# 讀取資料
df = pd.read_csv("data/historical_races.csv")
df['race_date'] = pd.to_datetime(df['race_date'])

# 選擇一匹馬（可改為輸入或參數）
horse = "浪漫勇士"
if horse not in df['horse_name'].values:
    horse = df['horse_name'].iloc[0]  # 改用第一匹馬
    print(f"⚠️ 指定馬匹不存在，改用: {horse}")

horse_df = df[df['horse_name'] == horse].sort_values('race_date')
horse_df['result'] = horse_df['is_top3'].map({1: '入位', 0: '未入位'})

# 繪製互動圖
fig = px.line(
    horse_df, 
    x='race_date', 
    y='win_odds',
    markers=True,
    color='result',
    color_discrete_map={'入位': 'green', '未入位': 'red'},
    title=f"🏇 {horse} — 互動式獨贏賠率走勢",
    hover_data=['jockey', 'trainer', 'actual_weight', 'draw']
)
fig.update_yaxes(autorange="reversed")  # 賠率越低越被看好
fig.update_layout(xaxis_title="比賽日期", yaxis_title="獨贏賠率")

# 儲存 HTML
os.makedirs("plots", exist_ok=True)
output_path = "plots/interactive_odds.html"
fig.write_html(output_path)
print(f"✅ 互動圖表已生成！路徑: {output_path}")