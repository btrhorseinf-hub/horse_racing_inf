# test_history.py
import sqlite3
from datetime import datetime

# 建立/連接資料庫
conn = sqlite3.connect("test.db")
cursor = conn.cursor()

# 建表
cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_table (
        id INTEGER PRIMARY KEY,
        name TEXT,
        created_at TEXT
    )
''')

# 插入資料
cursor.execute("INSERT INTO test_table (name, created_at) VALUES (?, ?)",
               ("金鑽貴人", datetime.now().isoformat()))

# 查詢
cursor.execute("SELECT * FROM test_table")
print(cursor.fetchall())

conn.commit()
conn.close()

print("✅ SQLite 測試成功！")