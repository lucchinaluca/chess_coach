"""The tools the coach agent can call, plus their JSON schemas for the Anthropic API.

Every tool returns small, structured, JSON-serialisable facts. None of them
writes advice; that is the LLM's job.
"""
from __future__ import annotations

import csv
import random
from collections import Counter
from typing import Callable, Optional

from .analysis import (Analyzer, AnalyzedGame, MoveRecord, annotate_game, explain)
from .config import Config
from .lichess import fetch_games_pgn, parse_games

FOCUS_VALUES = ("all", "opening", "endgame")
PHASES = ("opening", "middlegame", "endgame")


def _in_focus(m: MoveRecord, focus: str) -> bool:
    return focus == "all" or m.phase == focus


def _moment(m: MoveRecord) -> dict:
    d = {
        "game_id": m.game_id, "url": m.url, "move_number": m.move_number, "color": m.color,
        "played": m.san, "played_uci": m.uci, "fen_before": m.fen_before,
        "best_move": m.best_san, "best_move_uci": m.best_uci,
        "phase": m.phase, "win_pct_before": m.wp_before, "win_pct_after": m.wp_after,
        "win_pct_drop": m.drop, "severity": m.severity, "opening": m.opening,
    }
    if m.endgame_type:
        d["endgame_type"] = m.endgame_type
    if m.clock_s is not None:
        d["clock_seconds_left"] = round(m.clock_s)
    return d


class CoachTools:
    def __init__(self, cfg: Optional[Config] = None, username: str = "",
                 analyzer: Optional[Analyzer] = None,
                 progress: Optional[Callable[[str, int, int], None]] = None):
        self.cfg = cfg or Config()
        self.username = username
        self.an = analyzer or Analyzer(self.cfg)
        self.games: dict[str, AnalyzedGame] = {}
        self.progress = progress
        self._puzzles: Optional[list] = None

    # ------------------------------------------------------------------ tool 1
    def scan_games(self, username: str = "", n_games: int = 30, focus: str = "all") -> dict:
        username = (username or self.username).strip()
        if not username:
            return {"error": "No username given."}
        if focus not in FOCUS_VALUES:
            return {"error": f"focus must be one of {FOCUS_VALUES}"}
        n_games = max(1, min(int(n_games), 100))

        pgn = fetch_games_pgn(username, n_games, token=self.cfg.lichess_token)
        parsed = parse_games(pgn, username, limit=n_games)
        if not parsed:
            return {"error": f"No analysable standard games found for '{username}'."}

        for i, g in enumerate(parsed):
            site = g.headers.get("Site", "")
            gid = site.rstrip("/").split("/")[-1]
            if gid in self.games:
                continue
            if self.progress:
                self.progress("games", i + 1, len(parsed))
            ag = annotate_game(g, username, self.an, self.cfg)
            if ag:
                self.games[ag.id] = ag
        return self.summarize(focus, username)

    def summarize(self, focus: str = "all", username: str = "") -> dict:
        games = list(self.games.values())
        moves = [m for g in games for m in g.moves]
        by_phase = {}
        for ph in PHASES:
            ms = [m for m in moves if m.phase == ph]
            losses = [max(0.0, m.drop) for m in ms]
            by_phase[ph] = {
                "player_moves": len(ms),
                "mistakes": sum(1 for m in ms if m.severity == "mistake"),
                "blunders": sum(1 for m in ms if m.severity == "blunder"),
                "avg_win_pct_loss_per_move": round(sum(losses) / len(losses), 2) if losses else 0.0,
            }

        flagged = sorted([m for m in moves if m.severity and _in_focus(m, focus)],
                         key=lambda m: -m.drop)
        out = {
            "player": username or self.username,
            "games_analyzed": len(games),
            "results": dict(Counter(g.result for g in games)),
            "average_rating": (round(sum(g.rating for g in games if g.rating) /
                                     max(1, sum(1 for g in games if g.rating)))
                               if any(g.rating for g in games) else None),
            "focus": focus,
            "by_phase": by_phase,
            "flagged_mistakes_in_focus": len(flagged),
            "worst_moments": [_moment(m) for m in flagged[:10]],
        }

        with_clock = [m for m in flagged if m.clock_s is not None]
        if with_clock:
            low = sum(1 for m in with_clock if m.clock_s <= self.cfg.low_clock_seconds)
            out["clock"] = {
                "flagged_mistakes_with_clock_data": len(with_clock),
                "share_made_with_under_%ds_left" % self.cfg.low_clock_seconds:
                    round(low / len(with_clock), 2),
            }

        if focus in ("all", "opening"):
            opening_mistakes = Counter(m.opening for m in moves
                                       if m.severity and m.phase == "opening")
            games_by_opening = Counter(g.opening for g in games)
            out["openings_with_most_early_mistakes"] = [
                {"opening": name, "mistakes": c, "games": games_by_opening[name]}
                for name, c in opening_mistakes.most_common(5)]
        if focus in ("all", "endgame"):
            etypes = Counter(m.endgame_type for m in moves
                             if m.severity and m.phase == "endgame")
            out["endgame_types_with_mistakes"] = [
                {"type": t, "mistakes": c} for t, c in etypes.most_common(4)]
        if not flagged:
            out["note"] = "No big mistakes were found in this focus."
        return out

    # ------------------------------------------------------------------ tool 2
    def analyze_game(self, game_id: str, focus: str = "all", max_moments: int = 5) -> dict:
        g = self.games.get(game_id)
        if g is None:
            return {"error": f"Unknown game_id '{game_id}'. Use an id from scan_games."}
        if focus not in FOCUS_VALUES:
            return {"error": f"focus must be one of {FOCUS_VALUES}"}
        flagged = sorted([m for m in g.moves if m.severity and _in_focus(m, focus)],
                         key=lambda m: -m.drop)[: max(1, min(int(max_moments), 8))]
        return {
            "game": {"id": g.id, "url": g.url, "white": g.white, "black": g.black,
                     "player_color": g.color, "result": g.result, "opening": g.opening,
                     "player_rating": g.rating, "opponent_rating": g.opp_rating},
            "moments": [_moment(m) for m in flagged],
            "note": None if flagged else "No big mistakes in this game for this focus.",
        }

    # ------------------------------------------------------------------ tool 3
    def explain_position(self, fen: str, played_move_uci: str) -> dict:
        return explain(self.an, fen, played_move_uci)

    # ------------------------------------------------------------------ tool 4
    def _load_puzzles(self) -> list:
        if self._puzzles is None:
            rows = []
            with open(self.cfg.puzzle_csv, newline="", encoding="utf-8") as f:
                for i, r in enumerate(csv.DictReader(f)):
                    if i >= self.cfg.puzzle_rows:
                        break
                    try:
                        rows.append((r["PuzzleId"], int(r["Rating"]),
                                     int(r.get("Popularity", 0) or 0), r["Themes"].split()))
                    except (KeyError, ValueError):
                        continue
            self._puzzles = rows
        return self._puzzles

    def get_practice(self, theme: str, rating: Optional[int] = None, n: int = 3) -> dict:
        if not self.cfg.puzzle_csv:
            return {"error": "Puzzle database not configured (set PUZZLE_CSV). "
                             "Tell the player to try the Lichess puzzle themes page instead."}
        rows = self._load_puzzles()
        target = int(rating) if rating else 1400
        pool = [r for r in rows if theme in r[3] and abs(r[1] - target) <= 150]
        if not pool:
            return {"error": f"No puzzles found for theme '{theme}' near rating {target}."}
        pool.sort(key=lambda r: -r[2])
        picks = pool[: max(10, n * 3)]
        random.shuffle(picks)
        return {"theme": theme, "target_rating": target,
                "puzzles": [{"url": f"https://lichess.org/training/{p[0]}", "rating": p[1]}
                            for p in picks[: max(1, min(int(n), 5))]]}

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, name: str, args: dict) -> dict:
        try:
            if name == "scan_games":
                return self.scan_games(**args)
            if name == "analyze_game":
                return self.analyze_game(**args)
            if name == "explain_position":
                return self.explain_position(**args)
            if name == "get_practice":
                return self.get_practice(**args)
            return {"error": f"Unknown tool '{name}'."}
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except Exception as e:                       # never crash the agent loop
            return {"error": f"{type(e).__name__}: {e}"}

    def close(self):
        self.an.close()


TOOL_SCHEMAS = [
    {
        "name": "scan_games",
        "description": (
            "Downloads the player's recent Lichess games, analyses every position with Stockfish, "
            "and returns a summary: mistake and blunder counts per phase (opening, middlegame, "
            "endgame), average win-probability loss per phase, the player's clock habits, openings "
            "and endgame types linked to mistakes, and the 10 worst moments (each with a game_id, "
            "fen_before and played_uci). Call this first. It does not explain why a move was bad."),
        "input_schema": {
            "type": "object",
            "properties": {
                "username": {"type": "string", "description": "Lichess username."},
                "n_games": {"type": "integer", "description": "How many recent games (default 30, max 100)."},
                "focus": {"type": "string", "enum": list(FOCUS_VALUES),
                          "description": "Limit the worst moments to this phase. Default 'all'."},
            },
            "required": ["username"],
        },
    },
    {
        "name": "analyze_game",
        "description": (
            "Returns the biggest mistakes in ONE already-scanned game (id from scan_games), with the "
            "move played, Stockfish's best move and the win-probability change. Use it only when the "
            "player asks about a specific game."),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": {"type": "string"},
                "focus": {"type": "string", "enum": list(FOCUS_VALUES)},
                "max_moments": {"type": "integer", "description": "1-8, default 5."},
            },
            "required": ["game_id"],
        },
    },
    {
        "name": "explain_position",
        "description": (
            "Gives the facts needed to explain one mistake: Stockfish's best line, the line that "
            "follows the played move, win probabilities before and after, simple tactical facts "
            "(hanging pieces, pins, forks, missed captures, mates, material loss) and suggested "
            "puzzle themes. Pass fen_before and played_uci from a worst_moments entry. It reports "
            "facts only; there may be no named reason for positional mistakes."),
        "input_schema": {
            "type": "object",
            "properties": {
                "fen": {"type": "string", "description": "fen_before of the moment."},
                "played_move_uci": {"type": "string", "description": "played_uci of the moment."},
            },
            "required": ["fen", "played_move_uci"],
        },
    },
    {
        "name": "get_practice",
        "description": (
            "Returns links to a few Lichess puzzles of a given theme near the player's rating, for "
            "the 'practise this week' part of the report. Use a theme from explain_position's "
            "suggested_puzzle_themes (e.g. fork, pin, hangingPiece, mate). Links only, no solutions."),
        "input_schema": {
            "type": "object",
            "properties": {
                "theme": {"type": "string"},
                "rating": {"type": "integer", "description": "Player's rating (average_rating from scan_games)."},
                "n": {"type": "integer", "description": "How many puzzles, 1-5. Default 3."},
            },
            "required": ["theme"],
        },
    },
]
