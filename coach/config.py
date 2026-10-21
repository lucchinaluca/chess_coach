"""Settings. Everything can be overridden with environment variables."""
import os
import sys
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


import shutil

def _find_stockfish() -> str:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env
    for cand in (shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish",
                 "/usr/local/bin/stockfish"):
        if cand and os.path.isfile(cand):
            return cand
    return "stockfish"

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
