import sqlite3

conn = sqlite3.connect("data/predictions.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM predictions LIMIT 5")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
