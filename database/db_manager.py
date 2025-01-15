import sqlite3
from contextlib import contextmanager
import json
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="meetings.db"):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        with self.get_db() as conn:
            cursor = conn.cursor()
            # Create meetings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meetings (
                    meeting_id TEXT PRIMARY KEY,
                    transcript TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create websocket_connections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS websocket_connections (
                    ws_id TEXT PRIMARY KEY,
                    meeting_id TEXT,
                    user_id TEXT,
                    is_primary BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
                )
            """)
            conn.commit()

    def get_meeting(self, meeting_id: str) -> Optional[dict]:
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,))
            result = cursor.fetchone()
            return dict(result) if result else None

    def create_meeting(self, meeting_id: str):
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO meetings (meeting_id) VALUES (?)",
                (meeting_id,)
            )
            conn.commit()

    def update_transcript(self, meeting_id: str, transcript: str):
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE meetings SET transcript = ? WHERE meeting_id = ?",
                (transcript, meeting_id)
            )
            conn.commit()

    def get_primary_user(self, meeting_id: str) -> Optional[str]:
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ws_id FROM websocket_connections WHERE meeting_id = ? AND is_primary = TRUE",
                (meeting_id,)
            )
            result = cursor.fetchone()
            return result['ws_id'] if result else None

    def set_primary_user(self, meeting_id: str, ws_id: str):
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE websocket_connections 
                SET is_primary = CASE WHEN ws_id = ? THEN TRUE ELSE FALSE END 
                WHERE meeting_id = ?
                """,
                (ws_id, meeting_id)
            )
            conn.commit()

    def add_connection(self, ws_id: str, meeting_id: str, user_id: str):
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO websocket_connections (ws_id, meeting_id, user_id)
                VALUES (?, ?, ?)
                """,
                (ws_id, meeting_id, user_id)
            )
            conn.commit()

    def remove_connection(self, ws_id: str):
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM websocket_connections WHERE ws_id = ?",
                (ws_id,)
            )
            conn.commit()

    def add_user_to_meeting(self, meeting_id: str, user_id: str):
        """Add a user to a meeting in SQLite"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO websocket_connections (ws_id, meeting_id, user_id)
                VALUES (?, ?, ?)
                """,
                (ws_id, meeting_id, user_id)
            )
            conn.commit() 