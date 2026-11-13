import hashlib
import hmac
import os
import secrets
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "heart_rate.db"
ITERATIONS = 600_000


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    with get_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK (role IN ('user', 'admin')),
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                bpm REAL NOT NULL,
                signal_quality TEXT,
                status TEXT NOT NULL,
                measured_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_measurements_user_time
                ON measurements(user_id, measured_at DESC);
        """)


def _hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS
    )
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password, stored_hash):
    try:
        salt_hex, digest_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, ITERATIONS
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def register_user(full_name, username, password):
    full_name = full_name.strip()
    username = username.strip()

    if not full_name or not username or not password:
        return False, "Please fill in every field."
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (full_name, username, password_hash, role)
                VALUES (?, ?, ?, 'user')
                """,
                (full_name, username, _hash_password(password)),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username is already registered."


def authenticate_user(username, password):
    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT id, full_name, username, password_hash, role
            FROM users WHERE username = ? COLLATE NOCASE
            """,
            (username.strip(),),
        ).fetchone()

    if user and _verify_password(password, user["password_hash"]):
        return {
            "id": user["id"],
            "full_name": user["full_name"],
            "username": user["username"],
            "role": user["role"],
        }
    return None


def create_admin(full_name, username, password):
    """Create an administrator using explicitly supplied credentials."""
    full_name = full_name.strip()
    username = username.strip()

    if not full_name or not username or len(password) < 12:
        return False, "Provide a name, username, and password of at least 12 characters."

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (full_name, username, password_hash, role)
                VALUES (?, ?, ?, 'admin')
                """,
                (full_name, username, _hash_password(password)),
            )
        return True, "Administrator account created."
    except sqlite3.IntegrityError:
        return False, "That username is already registered."


def save_measurement(user_id, bpm, signal_quality, status):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO measurements (user_id, bpm, signal_quality, status)
            VALUES (?, ?, ?, ?)
            """,
            (int(user_id), float(bpm), str(signal_quality), str(status)),
        )


def get_user_measurements(user_id, limit=100):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT measured_at AS Time,
                   ROUND(bpm, 1) AS BPM,
                   signal_quality AS 'Signal Quality',
                   status AS Status
            FROM measurements
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(user_id), max(1, min(int(limit), 1000))),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_users():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT u.id, u.full_name, u.username, u.role, u.created_at,
                   COUNT(m.id) AS measurement_count
            FROM users u
            LEFT JOIN measurements m ON m.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_measurements(limit=500):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT u.full_name AS Name, u.username AS Username,
                   m.measured_at AS Time, ROUND(m.bpm, 1) AS BPM,
                   m.signal_quality AS 'Signal Quality',
                   m.status AS Status
            FROM measurements m
            JOIN users u ON u.id = m.user_id
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 5000)),),
        ).fetchall()
    return [dict(row) for row in rows]


init_db()
