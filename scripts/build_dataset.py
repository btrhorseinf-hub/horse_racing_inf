#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a clean, structured dataset from HKJC racing result Excel files.
Handles column mapping and mixed-type data safely.
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===== 欄位映射表 =====
COLUMN_MAPPING = {
    '名次': 'finish_position',
    '馬號': 'horse_number',
    '馬名': 'horse_name',
    '騎師': 'jockey',
    '練馬師': 'trainer',
    '實際 負磅': 'actual_weight',
    '排位 體重': 'declared_weight',
    '檔位': 'draw',
    '獨贏 賠率': 'win_odds',
    # 其他可能寫法
    '實際負磅': 'actual_weight',
    '排位體重': 'declared_weight',
    '獨贏賠率': 'win_odds',
}

def flatten_columns(columns):
    """處理 MultiIndex 欄位"""
    flattened = []
    for i, col in enumerate(columns):
        if isinstance(col, tuple):
            clean_parts = []
            for c in col:
                if pd.notna(c) and str(c).strip() and 'Unnamed' not in str(c):
                    clean_parts.append(str(c).strip())
            name = "_".join(clean_parts) if clean_parts else f"col_{i}"
        else:
            name = str(col).strip() if pd.notna(col) else f"col_{i}"
        flattened.append(name)
    return flattened

def load_all_races(raw_dir: str):
    all_races = []
    xlsx_files = list(Path(raw_dir).rglob("*.xlsx"))
    logger.info(f"🔍 找到 {len(xlsx_files)} 個 Excel 檔案")
    
    for file_path in xlsx_files:
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                if "Race" in str(sheet_name):
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                    df.columns = flatten_columns(df.columns)
                    df = df.rename(columns=COLUMN_MAPPING)  # 應用映射
                    df["source_file"] = file_path.stem
                    df["race_sheet"] = sheet_name
                    all_races.append(df)
        except Exception as e:
            logger.warning(f"⚠️ 跳過 {file_path.name}: {e}")
            continue
    
    if not all_races:
        raise ValueError("沒有找到任何有效的 Race 表格！")
    combined = pd.concat(all_races, ignore_index=True)
    logger.info(f"📊 合併完成：共 {len(combined)} 筆賽馬記錄")
    return combined

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    REQUIRED_COLUMNS = [
        'finish_position', 'horse_name', 'jockey', 'trainer',
        'actual_weight', 'declared_weight', 'draw', 'win_odds'
    ]
    cols_to_keep = [col for col in REQUIRED_COLUMNS if col in df.columns]
    cols_to_keep += ['source_file', 'race_sheet']
    df = df[cols_to_keep].copy()
    
    # 數值清洗
    for col in ['actual_weight', 'declared_weight', 'draw', 'win_odds']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 名次處理
    if 'finish_position' not in df.columns:
        raise KeyError("❌ 找不到 'finish_position' 欄位！請檢查 COLUMN_MAPPING")
    
    df['finish_position'] = (
        df['finish_position'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    )
    df = df.dropna(subset=['finish_position'])
    df['is_top3'] = df['finish_position'] <= 3.0
    
    # 生成 ID
    for col in ['horse_name', 'jockey', 'trainer']:
        if col in df.columns:
            df[f"{col}_id"] = pd.Categorical(df[col]).codes
    
    return df

def main(raw_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df_raw = load_all_races(raw_dir)
    df_clean = clean_data(df_raw)
    
    parquet_path = os.path.join(output_dir, "hkjc_racing_data.parquet")
    csv_path = os.path.join(output_dir, "hkjc_racing_data.csv")
    
    df_clean.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df_clean.to_parquet(parquet_path, index=False)
    
    logger.info(f"✅ CSV 已儲存至: {csv_path}")
    logger.info(f"✅ Parquet 已儲存至: {parquet_path}")
    logger.info(f"📈 最終記錄數：{len(df_clean)}")
    logger.info(f"🎯 入位比例：{df_clean['is_top3'].mean():.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="../data/raw")
    parser.add_argument("--output", default="../data/processed")
    args = parser.parse_args()
    main(args.raw, args.output)