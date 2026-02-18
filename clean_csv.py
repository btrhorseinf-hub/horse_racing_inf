# clean_csv.py
import pandas as pd
import sys
import os

def clean_next_race_csv(input_path: str, output_path: str = None):
    """
    自動清理 next_race.csv：
    - 移除 actual_weight, draw, win_odds, race_distance 中的逗號
    - 強制轉為數值
    - 保留原始文字欄位
    """
    if not os.path.exists(input_path):
        print(f"❌ 錯誤：找不到檔案 {input_path}")
        return False

    try:
        # 讀取 CSV（自動偵測編碼）
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_path, encoding='gbk')
        except:
            df = pd.read_csv(input_path, encoding='latin1')

    print(f"📊 原始資料形狀: {df.shape}")
    print("原始欄位:", list(df.columns))

    # 必要欄位檢查
    required_cols = ['horse_name', 'jockey', 'trainer', 'actual_weight', 'draw', 'win_odds', 'race_distance']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ 缺失必要欄位: {missing}")
        return False

    # 數值欄位清單
    numeric_cols = ['actual_weight', 'draw', 'win_odds', 'race_distance']

    # 清理步驟：移除逗號 + 轉為數值
    for col in numeric_cols:
        if col in df.columns:
            # 轉為字串 → 移除逗號 → 轉為 float
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '')      # 移除千位分隔符
                .str.replace(' ', '')      # 移除空格
                .str.replace('–', '')      # 移除破折號（常見於無數據）
                .str.replace('-', '')      # 移除減號
            )
            # 轉為數值，無效值變 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 移除含 NaN 的行（可選：你也可以選擇填入預設值）
    original_len = len(df)
    df.dropna(subset=numeric_cols, inplace=True)
    cleaned_len = len(df)

    print(f"🧹 已清理 {original_len - cleaned_len} 筆無效資料")
    print(f"✅ 最終有效資料筆數: {cleaned_len}")

    # 設定輸出路徑
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"

    # 輸出為 UTF-8 CSV（無索引）
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"💾 已儲存至: {output_path}")

    # 預覽前 3 行
    print("\n🔍 清理後預覽（前 3 行）:")
    print(df.head(3))

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python clean_csv.py <input_file.csv> [output_file.csv]")
        print("範例: python clean_csv.py next_race.csv cleaned_next_race.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = clean_next_race_csv(input_file, output_file)
    if success:
        print("\n🎉 清理完成！現在可以上傳 cleaned_next_race.csv 到你的 Streamlit 應用。")
    else:
        print("\n💥 清理失敗，請檢查輸入檔案格式。")