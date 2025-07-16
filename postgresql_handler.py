import logging
import logging.handlers
import psycopg2
from datetime import datetime

class PostgreSQLHandler(logging.Handler):
    def __init__(self, dsn):
        super().__init__()
        self.connection = psycopg2.connect(dsn)
        self.cursor = self.connection.cursor()

    def emit(self, record):
        try:
            log_time = datetime.fromtimestamp(record.created)
            session_id = getattr(record, "session_id", None)
            user_query = getattr(record, "user_query", None)
            response = getattr(record, "response", None)
            time_elapsed = getattr(record, "time_elapsed", None)

            self.cursor.execute(
                """
                INSERT INTO logs (session_id, log_time, user_query, response, time_elapsed)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, log_time, user_query, response, time_elapsed)
            )
            self.connection.commit()
            # self.conn.commit()
        except Exception as e:
            print(f"Failed to log to PosgreSQL: {e}")
