"""Create the `support` schema and tables (idempotent).

    DATABASE_URL=... uv run python scripts/migrate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supportbot import db  # noqa: E402

if __name__ == "__main__":
    db.migrate()
    print("schema support ready")
