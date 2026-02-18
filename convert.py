import pandas as pd

df = pd.read_csv('next_race.csv')
df.rename(columns={
    '馬名': 'horse_name',
    '騎師': 'jockey',
    '練馬師': 'trainer',
    '實際負磅': 'actual_weight',
    '檔位': 'draw',
    '獨贏賠率': 'win_odds'
})[['horse_name','jockey','trainer','actual_weight','draw','win_odds']].to_csv('input_horses.csv', index=False, encoding='utf-8')
print("✅ 已生成 input_horses.csv")