import sqlite3
from contextlib import closing

DB_NAME = "chat_history.db"

def init_db():
    """Creates the database and tables if they don't exist on boot."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    sender TEXT,
                    persona TEXT,
                    content TEXT,
                    latency_seconds REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

def log_message(session_id, sender, persona, content, latency=0.0):
    """Inserts a new chat message into the database."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                INSERT INTO Messages (session_id, sender, persona, content, latency_seconds)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, sender, persona, content, latency))
            conn.commit()

def get_chat_history(session_id):
    """Retrieves the chronological chat history for a specific session."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        conn.row_factory = sqlite3.Row  # Returns rows as dictionaries
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                SELECT sender, persona, content, latency_seconds, timestamp 
                FROM Messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            ''', (session_id,))
            return [dict(row) for row in cursor.fetchall()]