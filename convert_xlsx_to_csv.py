# convert_xlsx_to_csv.py
import pandas as pd
from pathlib import Path
import glob

def extract_race_data_from_excel(file_path):
    """
    從單個 HKJC 賽果 Excel 檔中提取所有 race 的資料
    """
    df_list = []
    xls = pd.ExcelFile(file_path)
    
    for sheet_name in xls.sheet_names:
        try:
            # 讀取每個工作表（每場 race）
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # 跳過空表或無效表
            if df.empty or '馬號' not in df.columns:
                continue
                
            # 標準化欄位名稱
            col_mapping = {
                '名次': 'finish_position',
                '馬名': 'horse_name',
                '騎師': 'jockey',
                '練馬師': 'trainer',
                '實際 負磅': 'actual_weight',
                '檔位': 'draw',
                '獨贏 賠率': 'win_odds',
                '排位 體重': 'body_weight',
                '完成 時間': 'finish_time',
                '頭馬 距離': 'distance_behind',
                '沿途 走位': 'running_position'
            }
            
            df = df.rename(columns=col_mapping)
            cols_needed = ['finish_position', 'horse_name', 'jockey', 'trainer', 
                          'actual_weight', 'draw', 'win_odds']
            
            if not all(col in df.columns for col in cols_needed):
                continue
                
            df = df[cols_needed].copy()
            
            # 推斷 race_date 和 race_id（從檔名）
            file_name = Path(file_path).stem
            date_str = file_name.split('_')[1]  # 取得日期部分（如 20230910）
            race_id = f"{date_str}_{sheet_name}"
            df['race_date'] = date_str
            df['race_id'] = race_id
            
            # 清理數據
            df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
            df['actual_weight'] = pd.to_numeric(df['actual_weight'], errors='coerce')
            df['draw'] = pd.to_numeric(df['draw'], errors='coerce')
            df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
            
            df_list.append(df)
            
        except Exception as e:
            print(f"⚠️ 處理 {file_path} / {sheet_name} 時出錯: {e}")
            continue
            
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def main():
    # 正確路徑：data/raw/
    xlsx_dir = Path("data/raw")
    xlsx_files = list(xlsx_dir.glob("HKJ_local_results_*.xlsx"))
    
    if not xlsx_files:
        print("❌ 找不到任何 HKJ_local_results_*.xlsx 檔案")
        return
        
    print(f"🔍 找到 {len(xlsx_files)} 個 Excel 檔案，開始合併...")
    
    all_races = []
    for file in sorted(xlsx_files):
        print(f"  處理 {file.name}...")
        df_race = extract_race_data_from_excel(file)
        if not df_race.empty:
            all_races.append(df_race)
    
    if not all_races:
        print("❌ 未成功提取任何有效賽事資料")
        return
        
    # 合併所有資料
    full_df = pd.concat(all_races, ignore_index=True)
    
    # 移除完全無效的行
    full_df = full_df.dropna(subset=['horse_name', 'finish_position'])
    
    print(f"✅ 成功提取 {len(full_df)} 筆馬匹賽果")
    
    # 保存為歷史資料集
    output_path = "data/historical_races.csv"
    Path("data").mkdir(exist_ok=True)
    full_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 已保存至 {output_path}")
    
    # 顯示前幾筆
    print("\n📋 前 5 筆資料:")
    print(full_df.head().to_string())

if __name__ == "__main__":
    main()
