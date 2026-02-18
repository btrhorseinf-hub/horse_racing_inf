#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動填入獨贏賠率到賽事 CSV
支援：
  ✅ 真實抓取 HKJC 即時賠率（賽馬日）
  ✅ 模擬賠率（非賽馬日 / 測試用）
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
import argparse
import random
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 預設賠率映射（測試用）
DEFAULT_ODDS_MAP = {
    "浪漫勇士": 2.2,
    "金鑽貴人": 4.5,
    "加州星球": 3.8,
    "美麗同享": 6.0,
    "賢者無敵": 5.5,
    "好眼光": 7.0,
    "包裝猛將": 12.0,
    "飛鷹翱翔": 15.0,
    "嘉應高昇": 8.5,
    "永遠美麗": 9.0,
    "自勝者強": 5.0,
    "發財先鋒": 10.0,
}

def fetch_hkjc_win_odds_by_horse_names(horse_names):
    """
    從 HKJC 動態賠率頁面抓取指定馬名的獨贏賠率
    因 HKJC 不提供直接 API，需解析 HTML
    """
    url = "https://bet.hkjc.com/racing/pages/odds_wp.aspx?lang=ch&date=today"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            logger.warning("⚠️ 今日無賠率資料（非賽馬日）")
            return {}
        response.raise_for_status()
        response.encoding = 'utf-8'
    except Exception as e:
        logger.error(f"❌ 無法載入賠率頁面: {e}")
        return {}
    
    soup = BeautifulSoup(response.text, 'html.parser')
    odds_map = {}
    
    # 找所有馬匹賠率區塊（根據實際 HTML 結構）
    horse_blocks = soup.find_all('div', class_='horseInfo')
    for block in horse_blocks:
        try:
            # 馬名（中文）
            name_elem = block.find('span', class_='horseName')
            if not name_elem:
                continue
            horse_name = name_elem.get_text(strip=True).replace('\u3000', ' ')  # 全形空格
            
            # 賠率（可能有「停售」等文字）
            odds_elem = block.find('div', class_='win')
            if not odds_elem:
                continue
            odds_text = odds_elem.get_text(strip=True)
            
            # 清理賠率（只保留數字和小數點）
            if '停' in odds_text or '不' in odds_text or odds_text == '-':
                continue
            try:
                win_odds = float(odds_text)
                odds_map[horse_name] = win_odds
            except:
                continue
        except Exception as e:
            continue
    
    logger.info(f"✅ 從 HKJC 抓取 {len(odds_map)} 匹馬的賠率")
    return odds_map

def generate_simulated_odds(horse_names):
    """生成合理的模擬賠率（基於馬名或隨機）"""
    odds_map = {}
    for name in horse_names:
        if name in DEFAULT_ODDS_MAP:
            odds_map[name] = DEFAULT_ODDS_MAP[name]
        else:
            # 隨機生成 2.0 ~ 20.0 的賠率（熱門低，冷門高）
            odds = round(random.uniform(2.0, 20.0), 1)
            odds_map[name] = odds
    return odds_map

def main(input_csv: str, output_csv: str = None):
    if not os.path.exists(input_csv):
        logger.error(f"❌ 輸入文件不存在: {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    
    if '馬名' not in df.columns and 'horse_name' not in df.columns:
        logger.error("❌ CSV 必須包含 '馬名' 或 'horse_name' 欄位")
        return
    
    # 統一欄位名
    horse_col = '馬名' if '馬名' in df.columns else 'horse_name'
    horse_names = df[horse_col].tolist()
    
    # 嘗試抓取真實賠率
    logger.info("使用網路抓取 HKJC 即時賠率...")
    real_odds = fetch_hkjc_win_odds_by_horse_names(horse_names)
    
    if not real_odds:
        logger.info("使用網路失敗，啟用模擬賠率模式...")
        real_odds = generate_simulated_odds(horse_names)
    
    # 填入賠率
    df['獨贏賠率'] = df[horse_col].map(real_odds).fillna(999.0)
    
    # 輸出
    out_file = output_csv or input_csv.replace('.csv', '_with_odds.csv')
    df.to_csv(out_file, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 賠率已填入並儲存至: {out_file}")
    
    # 顯示結果
    print("\n🎯 賠率填入結果:")
    for _, row in df.iterrows():
        print(f"  • {row[horse_col]:<12} → {row['獨贏賠率']:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動填入獨贏賠率")
    parser.add_argument("--input", required=True, help="輸入 CSV 路徑（需含馬名）")
    parser.add_argument("--output", help="輸出 CSV 路徑（預設: 覆蓋原文件或加 _with_odds）")
    
    args = parser.parse_args()
    main(args.input, args.output)