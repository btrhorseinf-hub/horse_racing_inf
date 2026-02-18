# generate_historical_data.py —— 高擬真模擬歷史賽馬資料（強化版）

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ====== 設定參數 ======
N_RACES = 200          # 賽事場數
HORSES_PER_RACE = 12   # 每場馬匹數
TOTAL_RECORDS = N_RACES * HORSES_PER_RACE

# ====== 定義選項 ======
jockeys = ["莫雷拉", "潘頓", "田泰安", "蔡明紹", "何澤堯", "布文", "艾道拿", "周俊樂", "梁家俊"]
top_jockeys = {"莫雷拉", "潘頓"}

trainers = ["呂健威", "蔡約翰", "方嘉柏", "沈集成", "葉楚航", "大衛希斯", "告東尼", "蘇偉賢"]
top_trainers = {"蔡約翰", "呂健威"}

track_conditions = ["好地", "好至快", "軟地", "黏地"]
classes = ["第一班", "第二班", "第三班", "普通賽", "精英班", "盃賽"]
distances = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]

# 生成日期範圍（過去一年）
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(N_RACES)]

data = []

for i, race_date in enumerate(dates):
    distance = np.random.choice(distances)
    track = np.random.choice(track_conditions)
    race_class = np.random.choice(classes)
    
    for j in range(HORSES_PER_RACE):
        horse_name = f"馬_{i:03d}_{j:02d}"
        jockey = np.random.choice(jockeys)
        trainer = np.random.choice(trainers)
        weight = np.random.randint(100, 135)
        draw = np.random.randint(1, 15)
        age = np.random.randint(2, 9)
        
        # 基礎賠率：根據實力反推（越強賠率越低）
        base_odds = 15.0
        if jockey in top_jockeys:
            base_odds *= 0.6
        if trainer in top_trainers:
            base_odds *= 0.7
        if draw <= 3:
            base_odds *= 0.85
        elif draw >= 12:
            base_odds *= 1.2
        if race_class in ["第一班", "盃賽"]:
            base_odds *= 0.9  # 高班次競爭激烈，但強馬集中
        
        win_odds = max(1.5, np.random.normal(base_odds, 2.5))
        win_odds = round(win_odds, 2)

        # === 強化 is_top3 生成邏輯 ===
        score = 0.0
        # 1. 賠率是核心指標（賠率越低，實力越強）
        score += max(0, 12 - win_odds) * 0.9
        
        # 2. 頂級騎師加成
        if jockey in top_jockeys:
            score += 3.0
        elif jockey in ["田泰安", "蔡明紹", "何澤堯"]:
            score += 1.5
        
        # 3. 頂級練馬師加成
        if trainer in top_trainers:
            score += 2.5
        
        # 4. 檔位影響（內檔優勢）
        if draw <= 4:
            score += 2.0
        elif draw <= 8:
            score += 0.5
        else:
            score -= 1.0
        
        # 5. 跑道狀況
        if track in ["好地", "好至快"]:
            score += 1.0
        
        # 6. 馬齡黃金期
        if age in [4, 5]:
            score += 1.5
        elif age == 3 or age == 6:
            score += 0.5
        
        # 7. 班次影響（高班次競爭大，但入前三仍較可能）
        if race_class in ["第一班", "盃賽"]:
            score += 1.0
        elif race_class == "普通賽":
            score -= 0.5

        # 轉換為機率（Sigmoid），控制難度
        prob = 1 / (1 + np.exp(-0.35 * (score - 7.0)))
        prob = np.clip(prob, 0.05, 0.95)  # 避免極端
        
        is_top3 = 1 if np.random.rand() < prob else 0

        data.append({
            "race_date": race_date.strftime("%Y-%m-%d"),
            "horse_name": horse_name,
            "jockey": jockey,
            "trainer": trainer,
            "actual_weight": weight,
            "draw": draw,
            "win_odds": win_odds,
            "race_distance": distance,
            "track_condition": track,
            "horse_age": age,
            "class": race_class,
            "is_top3": is_top3
        })

# ====== 儲存為 CSV ======
df = pd.DataFrame(data)
df.to_csv("historical_races.csv", index=False, encoding="utf-8-sig")
print(f"✅ 已生成 historical_races.csv（共 {len(df)} 筆記錄）")
print("📁 欄位：", list(df.columns))

# 顯示正負樣本比例
top3_rate = df["is_top3"].mean()
print(f"📊 入前三比例: {top3_rate:.2%}")