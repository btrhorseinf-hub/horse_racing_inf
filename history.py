# history.py
import sqlite3
import os
from datetime import datetime
from typing import List, Dict

DB_PATH = "predictions_history.db"

def init_db():
    """初始化資料庫（若不存在）"""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_date TEXT NOT NULL,
                horse_name TEXT NOT NULL,
                jockey TEXT,
                trainer TEXT,
                win_odds REAL,
                predicted_top3_prob REAL,
                value_score REAL,
                kelly_fraction REAL,
                actual_result TEXT DEFAULT 'unknown'  -- 'top3', 'not_top3', 'unknown'
            )
        ''')
        conn.commit()
        conn.close()

def save_predictions(results: List[Dict], race_date: str = None):
    """儲存一批預測結果到歷史資料庫"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    current_date = race_date or datetime.now().strftime("%Y-%m-%d")
    
    for r in results:
        cursor.execute('''
            INSERT INTO predictions 
            (race_date, horse_name, jockey, trainer, win_odds, 
             predicted_top3_prob, value_score, kelly_fraction, actual_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_date,
            r["horse_name"],
            r["jockey"],
            r["trainer"],
            r["win_odds"],
            r["predicted_top3_prob"],
            r["value_score"],
            r["kelly_fraction"],
            "unknown"
        ))
    
    conn.commit()
    conn.close()

def get_all_predictions() -> List[Dict]:
    """取得所有歷史預測（用於前端顯示）"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 讓結果可用 dict 方式取值
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY race_date DESC, value_score DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]