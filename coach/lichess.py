"""Download and parse a player's games from the Lichess API."""
from __future__ import annotations

import io
import time

import chess.pgn
import requests

API = "https://lichess.org/api/games/user/{user}"


class LichessError(Exception):
    pass


def fetch_games_pgn(username: str, max_games: int = 30, token: str = "",
                    perf_type: str | None = None, retries: int = 2) -> str:
    """Return the player's most recent games as one PGN string (with clocks and opening)."""
    params = {
        "max": max_games, "clocks": "true", "opening": "true", "evals": "false",
        "moves": "true", "tags": "true", "sort": "dateDesc",
    }
    if perf_type:
        params["perfType"] = perf_type
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": "chess-coach/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(retries + 1):
        r = requests.get(API.format(user=username), params=params, headers=headers, timeout=60)
        if r.status_code == 404:
            raise LichessError(f"Lichess user '{username}' not found.")
        if r.status_code == 429:
            if attempt < retries:
                time.sleep(61)          # Lichess asks clients to wait a minute
                continue
            raise LichessError("Lichess rate limit hit. Wait a minute and try again.")
        if r.status_code != 200:
            raise LichessError(f"Lichess returned HTTP {r.status_code}.")
        r.encoding = "utf-8"
        return r.text
    raise LichessError("Could not download games.")


def parse_games(pgn_text: str, username: str, limit: int = 100) -> list[chess.pgn.Game]:
    """Parse standard-chess games the user played in. Skips variants and odd starts."""
    games = []
    stream = io.StringIO(pgn_text)
    name = username.lower()
    while len(games) < limit:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        h = game.headers
        if h.get("Variant", "Standard") not in ("Standard", "From Position"):
            continue
        if "FEN" in h or h.get("SetUp") == "1":
            continue
        if name not in (h.get("White", "").lower(), h.get("Black", "").lower()):
            continue
        if sum(1 for _ in game.mainline_moves()) < 8:
            continue
        games.append(game)
    return games
