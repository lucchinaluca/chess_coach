"""Settings. Everything can be overridden with environment variables."""
import os
import sys
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _default_stockfish() -> str:
    """Return the STOCKFISH_PATH env var if set, otherwise look for the exe
    next to the running script (the project root), then fall back to 'stockfish'
    on PATH."""
    if os.environ.get("STOCKFISH_PATH"):
        return os.environ["STOCKFISH_PATH"]
    # look in the same directory as the entry-point script
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else ""
    for candidate in [
        os.path.join(script_dir, "stockfish-windows-x86-64-universal.exe"),
        os.path.join(script_dir, "stockfish.exe"),
        os.path.join(script_dir, "stockfish"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return "stockfish"  # last resort: hope it's on PATH


@dataclass
class Config:
    # Stockfish
    stockfish_path: str = field(default_factory=_default_stockfish)
    depth: int = field(default_factory=lambda: int(_env("COACH_DEPTH", "12")))
    threads: int = field(default_factory=lambda: int(_env("COACH_THREADS", "2")))
    hash_mb: int = field(default_factory=lambda: int(_env("COACH_HASH_MB", "128")))
    cache_path: str = field(default_factory=lambda: _env("COACH_CACHE", "coach_cache.sqlite"))

    # What counts as a mistake, in win-probability points lost (0-100 scale)
    mistake_drop: float = 20.0
    blunder_drop: float = 30.0
    # Do not flag a drop when the player was already winning/losing by a lot
    still_winning_before: float = 92.0
    still_winning_after: float = 70.0
    already_lost_before: float = 8.0

    # Phases
    opening_until_move: int = 10
    endgame_max_material: int = 26     # combined non-pawn material of both sides

    # Lichess
    lichess_token: str = field(default_factory=lambda: _env("LICHESS_TOKEN", ""))

    # Puzzles (optional): path to lichess_db_puzzle.csv (decompressed)
    puzzle_csv: str = field(default_factory=lambda: _env("PUZZLE_CSV", ""))
    puzzle_rows: int = field(default_factory=lambda: int(_env("PUZZLE_ROWS", "400000")))

    # Agent
    model: str = field(default_factory=lambda: _env("COACH_MODEL", "gemini-3.8-flash"))
    max_tool_calls: int = 8
    low_clock_seconds: int = 30
