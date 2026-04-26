from __future__ import annotations

import os
import sqlite3
import time
from typing import Iterable, Mapping, Optional


class SQLiteKVCache:
    """Single-file SQLite-backed key/value cache."""

    def __init__(self, db_path: str, enabled: bool = True):
        self.db_path = db_path
        self.enabled = enabled
        self._conn: Optional[sqlite3.Connection] = None
        if enabled:
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    stage TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (stage, key)
                )
                """
            )
            self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteKVCache is disabled or closed")
        return self._conn

    def get(self, stage: str, key: str) -> bytes | None:
        if not self.enabled:
            return None
        row = self.conn.execute(
            "SELECT value FROM cache WHERE stage = ? AND key = ?",
            (stage, key),
        ).fetchone()
        return bytes(row[0]) if row is not None else None

    def get_many(self, stage: str, keys: Iterable[str]) -> dict[str, bytes | None]:
        keys = list(keys)
        result: dict[str, bytes | None] = {key: None for key in keys}
        if not self.enabled or not keys:
            return result

        unique_keys = list(dict.fromkeys(keys))
        chunk_size = 900
        for start in range(0, len(unique_keys), chunk_size):
            chunk = unique_keys[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT key, value FROM cache WHERE stage = ? AND key IN ({placeholders})",
                [stage, *chunk],
            ).fetchall()
            for key, value in rows:
                result[key] = bytes(value)
        return result

    def set(self, stage: str, key: str, value: bytes) -> None:
        if not self.enabled:
            return
        self.conn.execute(
            """
            INSERT OR REPLACE INTO cache(stage, key, value, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (stage, key, sqlite3.Binary(value), time.time()),
        )
        self.conn.commit()

    def set_many(self, stage: str, values: Mapping[str, bytes] | Iterable[tuple[str, bytes]]) -> None:
        if not self.enabled:
            return
        items = list(values.items() if isinstance(values, Mapping) else values)
        if not items:
            return
        now = time.time()
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO cache(stage, key, value, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [(stage, key, sqlite3.Binary(value), now) for key, value in items],
        )
        self.conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
