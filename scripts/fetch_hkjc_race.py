#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動抓取 HKJC 最近一場賽事（支援非賽馬日）
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import argparse
import logging
import os
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 改用「本週賽程」作為入口（更穩定）
WEEKLY_RACE_URL = "https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResultsAll.aspx"

def get_next_race_info():
    """從本週賽程中找出最近一場未開跑的賽事"""
    logger.info("正在查詢 HKJC 最近一場賽事...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(WEEKLY_RACE_URL, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except Exception as e:
        logger.error(f"❌ 無法連接 HKJC 賽程頁面: {e}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 找所有賽事連結（包含 RaceDate, Venue, RaceNo）
    race_links = soup.find_all('a', href=True)
    future_races = []
    
    for link in race_links:
        href = link['href']
        if 'DisplayRaceCard.aspx' in href and 'RaceDate=' in href:
            # 解析參數
            params = {}
            for part in href.split('&'):
                if '=' in part:
                    key, val = part.split('=', 1)
                    params[key] = val
            
            if 'RaceDate' in params and 'Venue' in params:
                # 檢查是否為未來或今天的賽事
                race_date = params['RaceDate']
                try:
                    race_dt = datetime.strptime(race_date, "%Y/%m/%d")
                    today = datetime.now()
                    if race_dt.date() >= today.date():
                        # 找到第一場就返回（最新一場）
                        return {
                            'date': race_date.replace('/', ''),
                            'venue': params['Venue'],
                            'race_no': '1'  # 從第 1 場開始
                        }
                except:
                    continue
    
    logger.warning("⚠️ 未找到近期賽事（可能本週無賽馬）")
    return None

def fetch_race_card(date, venue, race_no):
    """抓取單場賽事馬匹資料（兼容新舊 HTML）"""
    url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/DisplayRaceCard.aspx?RaceDate={date[:4]}/{date[4:6]}/{date[6:]}&Venue={venue}&RaceNo={race_no}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except Exception as e:
        logger.error(f"❌ 無法載入賽事頁面 ({url}): {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    horses = []
    
    # 新版表格選擇器（2024–2026 年常用結構）
    tables = soup.find_all('table', class_='table_bd_dr')
    if not tables:
        logger.warning("⚠️ 未找到馬匹表格（可能頁面結構變更）")
        return []
    
    # 取第一個表格（通常就是馬匹名單）
    table = tables[0]
    rows = table.find_all('tr')[2:]  # 跳過前兩行（標題）
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 8:
            continue
        
        try:
            draw = cols[0].get_text(strip=True)  # 檔位
            horse_name = cols[3].find('a').get_text(strip=True) if cols[3].find('a') else cols[3].get_text(strip=True)
            jockey = cols[5].get_text(strip=True)
            trainer = cols[6].get_text(strip=True)
            weight = cols[7].get_text(strip=True)
            
            # 清理負磅（移除非數字）
            weight = ''.join(filter(str.isdigit, weight)) or '120'
            
            horses.append({
                '檔位': draw,
                '馬名': horse_name,
                '騎師': jockey,
                '練馬師': trainer,
                '實際負磅': int(weight)
            })
        except Exception as e:
            logger.debug(f"跳過無效行: {e}")
            continue
    
    return horses

def main(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 找最近一場賽事
    race_info = get_next_race_info()
    if not race_info:
        logger.error("❌ 無法取得近期賽事資訊")
        return
    
    date, venue = race_info['date'], race_info['venue']
    logger.info(f"✅ 找到賽事: {date} @ {venue}")
    
    # 2. 抓取所有場次（最多 12 場）
    all_races = []
    for race_no in range(1, 13):
        logger.info(f"抓取第 {race_no} 場...")
        horses = fetch_race_card(date, venue, str(race_no))
        if not horses:
            break  # 無更多場次
        
        # 臨時賠率（因 XML API 不穩定，先用預設值）
        for i, h in enumerate(horses):
            h['獨贏賠率'] = 999.0  # 表示未提供
        
        df = pd.DataFrame(horses)
        filename = f"{output_dir}/hkjc_{date}_{venue}_race{race_no}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 儲存: {filename}")
        all_races.append(filename)
        time.sleep(1)  # 避免過快請求
    
    if all_races:
        logger.info(f"🎉 共抓取 {len(all_races)} 場賽事！")
        # 同時生成一個合併文件方便測試
        sample_file = all_races[0]
        os.system(f"cp {sample_file} {output_dir}/next_race.csv")
        logger.info("📎 已複製第一場為 next_race.csv（供 predict.py 使用）")
    else:
        logger.warning("⚠️ 未抓取到任何賽事")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/predictions")
    args = parser.parse_args()
    main(args.output)