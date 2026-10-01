import sqlite3
from datetime import datetime

DB_PATH = "database/visitors.db"


class Database:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.create_tables()
        self.migrate_embeddings()

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                FOREIGN KEY (visitor_id) REFERENCES visitors(id)
            )
        """)

        # Multiple face embeddings/templates for each visitor
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visitor_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (visitor_id) REFERENCES visitors(id)
            )
        """)

        self.conn.commit()

    def migrate_embeddings(self):
        """
        Move the original embedding of existing visitors into
        the visitor_embeddings table.

        Existing database data is preserved.
        """

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT id, embedding
            FROM visitors
            WHERE embedding IS NOT NULL
        """)

        visitors = cursor.fetchall()

        for visitor_id, embedding_blob in visitors:

            cursor.execute("""
                SELECT id
                FROM visitor_embeddings
                WHERE visitor_id = ?
                LIMIT 1
            """, (visitor_id,))

            existing = cursor.fetchone()

            if existing is None:

                cursor.execute("""
                    INSERT INTO visitor_embeddings
                    (visitor_id, embedding, created_at)
                    VALUES (?, ?, ?)
                """, (
                    visitor_id,
                    embedding_blob,
                    datetime.now().isoformat()
                ))

        self.conn.commit()

    def add_visitor(self, embedding):
        now = datetime.now().isoformat()

        embedding_blob = embedding.astype(
            "float32"
        ).tobytes()

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO visitors
            (first_seen, last_seen, embedding)
            VALUES (?, ?, ?)
        """, (
            now,
            now,
            embedding_blob
        ))

        visitor_id = cursor.lastrowid

        # Also store the first embedding as a template
        cursor.execute("""
            INSERT INTO visitor_embeddings
            (visitor_id, embedding, created_at)
            VALUES (?, ?, ?)
        """, (
            visitor_id,
            embedding_blob,
            now
        ))

        self.conn.commit()

        return visitor_id

    def get_visitors(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT id, first_seen, last_seen, embedding
            FROM visitors
        """)

        return cursor.fetchall()

    def get_all_embeddings(self):
        """
        Return all face templates.

        Returns:
            [
                (visitor_id, embedding_blob),
                ...
            ]
        """

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT visitor_id, embedding
            FROM visitor_embeddings
        """)

        return cursor.fetchall()

    def add_embedding(self, visitor_id, embedding):
        """
        Store another appearance of an already-known visitor.
        """

        now = datetime.now().isoformat()

        embedding_blob = embedding.astype(
            "float32"
        ).tobytes()

        self.conn.execute("""
            INSERT INTO visitor_embeddings
            (visitor_id, embedding, created_at)
            VALUES (?, ?, ?)
        """, (
            visitor_id,
            embedding_blob,
            now
        ))

        self.conn.commit()

    def update_last_seen(self, visitor_id):
        now = datetime.now().isoformat()

        self.conn.execute("""
            UPDATE visitors
            SET last_seen = ?
            WHERE id = ?
        """, (
            now,
            visitor_id
        ))

        self.conn.commit()

    def add_event(
        self,
        visitor_id,
        event_type,
        image_path=None
    ):
        now = datetime.now().isoformat()

        self.conn.execute("""
            INSERT INTO events
            (visitor_id, event_type, timestamp, image_path)
            VALUES (?, ?, ?, ?)
        """, (
            visitor_id,
            event_type,
            now,
            image_path
        ))

        self.conn.commit()

    def close(self):
        self.conn.close()