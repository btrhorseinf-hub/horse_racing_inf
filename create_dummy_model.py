import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 建立最小可行資料集
data = {
    'horse_name': ['浪漫勇士', '金鑽貴人', '加州星球', '美麗同享'],
    'jockey': ['潘頓', '莫雷拉', '蔡明紹', '田泰安'],
    'trainer': ['沈集成', '呂健威', '告東尼', '韋達'],
    'actual_weight': [133, 128, 130, 126],
    'draw': [5, 3, 7, 2],
    'win_odds': [2.2, 4.5, 3.8, 6.0],
    'top3': [1, 1, 1, 0]  # 虛擬標籤：是否進入前三
}
df = pd.DataFrame(data)

# 訓練編碼器
jockey_encoder = LabelEncoder()
trainer_encoder = LabelEncoder()
jockey_encoder.fit(df['jockey'].tolist() + ['未知騎師'])  # 加入未知類別
trainer_encoder.fit(df['trainer'].tolist() + ['未知練馬師'])

# 編碼
df['jockey_encoded'] = jockey_encoder.transform(df['jockey'])
df['trainer_encoded'] = trainer_encoder.transform(df['trainer'])

# 特徵與標籤
X = df[['actual_weight', 'draw', 'win_odds', 'jockey_encoded', 'trainer_encoded']]
y = df['top3']

# 使用簡單分類器（實際應換成 RandomForest/XGBoost）
model = DummyClassifier(strategy="prior").fit(X, y)

# 儲存模型與編碼器
joblib.dump(model, "model.pkl")
joblib.dump(jockey_encoder, "jockey_encoder.pkl")
joblib.dump(trainer_encoder, "trainer_encoder.pkl")

print("✅ 虛擬模型已成功生成！")
print("   - model.pkl")
print("   - jockey_encoder.pkl")
print("   - trainer_encoder.pkl")
