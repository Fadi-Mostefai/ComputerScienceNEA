import sqlite3

def create_database():
    conn = sqlite3.connect('search_app.db')
    c = conn.cursor()

    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL
                 )''')

    # Create search_history table with a link column
    c.execute('''CREATE TABLE IF NOT EXISTS search_history (
                 id INTEGER PRIMARY KEY,
                 user_id INTEGER,
                 link TEXT NOT NULL,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                 )''')

    conn.commit()
    conn.close()



create_database()
print("Successfull")