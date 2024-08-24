import sqlite3

conn = sqlite3.connect('trader_database.db')
conn.execute('''DROP TABLE days''')
