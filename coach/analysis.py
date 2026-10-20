"""Everything chess-related lives here. Stockfish is the only judge of move quality.

The LLM never sees raw positions to evaluate; it only sees the structured facts
produced by this module.
"""
from __future__ import annotations

import math
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine
import chess.pgn

from .config import Config

MATE_CP = 10_000
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
          chess.QUEEN: 9, chess.KING: 100}
PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def win_pct(cp: float) -> float:
    """Centipawns -> win probability (0-100) with a logistic curve (Lichess-style constant)."""
    cp = max(-1000.0, min(1000.0, float(cp)))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def material_balance(board: chess.Board, color: chess.Color) -> int:
    """Material (pawns included) from `color`'s point of view."""
    total = 0
    for pt, v in VALUES.items():
        if pt == chess.KING:
            continue
        total += v * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color)))
    return total


def nonpawn_material(board: chess.Board) -> int:
    return sum(VALUES[pt] * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
               for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))


def phase(board: chess.Board, cfg: Config) -> str:
    if board.fullmove_number <= cfg.opening_until_move:
        return "opening"
    if nonpawn_material(board) <= cfg.endgame_max_material:
        return "endgame"
    return "middlegame"


def endgame_type(board: chess.Board) -> str:
    def n(pt):
        return len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
    q, r, minors = n(chess.QUEEN), n(chess.ROOK), n(chess.BISHOP) + n(chess.KNIGHT)
    if q == r == minors == 0:
        return "pawn endgame"
    if q == 0 and r > 0 and minors == 0:
        return "rook endgame"
    if q == 0 and r == 0 and minors > 0:
        return "minor-piece endgame"
    if q > 0 and r == 0 and minors == 0:
        return "queen endgame"
    if q == 0 and r > 0 and minors > 0:
        return "rook and minor-piece endgame"
    return "mixed endgame"


def san_line(board: chess.Board, ucis: list[str]) -> str:
    """Numbered SAN for a list of UCI moves from `board`, e.g. '12. Nxe5 Qd4 13. Nf3'."""
    if not ucis:
        return ""
    moves = []
    b = board.copy(stack=False)
    for u in ucis:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        moves.append(m)
        b.push(m)
    return board.variation_san(moves) if moves else ""


# ---------------------------------------------------------------------------
# Stockfish wrapper with an on-disk cache
# ---------------------------------------------------------------------------

@dataclass
class Evaluation:
    cp_white: int                 # centipawns from White's point of view (mate = +/-10000)
    pv: list[str] = field(default_factory=list)   # best line as UCI moves


class Analyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._lock = threading.Lock()
        self._db = sqlite3.connect(cfg.cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS evals ("
                         "epd TEXT, depth INTEGER, cp INTEGER, pv TEXT, "
                         "PRIMARY KEY (epd, depth))")
        self._db.commit()

    def _engine_ready(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.cfg.stockfish_path)
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Stockfish not found at '{self.cfg.stockfish_path}'. "
                    "Set STOCKFISH_PATH to the engine executable.") from e
            self._engine.configure({"Threads": self.cfg.threads, "Hash": self.cfg.hash_mb})
        return self._engine

    def evaluate(self, board: chess.Board) -> Evaluation:
        if board.is_checkmate():
            return Evaluation(-MATE_CP if board.turn == chess.WHITE else MATE_CP, [])
        if board.is_stalemate() or board.is_insufficient_material():
            return Evaluation(0, [])
        key = board.epd()
        with self._lock:
            row = self._db.execute("SELECT cp, pv FROM evals WHERE epd=? AND depth=?",
                                   (key, self.cfg.depth)).fetchone()
            if row:
                return Evaluation(row[0], row[1].split() if row[1] else [])
            info = self._engine_ready().analyse(board, chess.engine.Limit(depth=self.cfg.depth))
            cp = int(info["score"].white().score(mate_score=MATE_CP))
            pv = [m.uci() for m in info.get("pv", [])[:8]]
            self._db.execute("INSERT OR REPLACE INTO evals VALUES (?,?,?,?)",
                             (key, self.cfg.depth, cp, " ".join(pv)))
            self._db.commit()
            return Evaluation(cp, pv)

    def close(self):
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.quit()
                except Exception:
                    pass
                self._engine = None
            self._db.close()


# ---------------------------------------------------------------------------
# Annotated games
# ---------------------------------------------------------------------------

@dataclass
class MoveRecord:
    game_id: str
    url: str
    ply: int
    move_number: int
    color: str
    san: str
    uci: str
    fen_before: str
    wp_before: float
    wp_after: float
    drop: float
    best_san: Optional[str]
    best_uci: Optional[str]
    phase: str
    endgame_type: Optional[str]
    clock_s: Optional[float]
    severity: Optional[str]       # "blunder", "mistake" or None
    opening: str


@dataclass
class AnalyzedGame:
    id: str
    url: str
    white: str
    black: str
    color: str                    # the player's colour: "white" or "black"
    result: str                   # "win", "loss", "draw" or "unknown" (player's point of view)
    opening: str
    eco: str
    rating: Optional[int]
    opp_rating: Optional[int]
    date: str
    moves: list[MoveRecord] = field(default_factory=list)   # the player's moves only


def _int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def classify(wp_before: float, wp_after: float, cfg: Config) -> Optional[str]:
    drop = wp_before - wp_after
    if wp_before >= cfg.still_winning_before and wp_after >= cfg.still_winning_after:
        return None                          # still clearly winning
    if wp_before <= cfg.already_lost_before:
        return None                          # already lost, nothing to learn here
    if drop >= cfg.blunder_drop:
        return "blunder"
    if drop >= cfg.mistake_drop:
        return "mistake"
    return None


def annotate_game(game: chess.pgn.Game, username: str, an: Analyzer, cfg: Config,
                  progress=None) -> Optional[AnalyzedGame]:
    h = game.headers
    white, black = h.get("White", "?"), h.get("Black", "?")
    if username.lower() == white.lower():
        player = chess.WHITE
    elif username.lower() == black.lower():
        player = chess.BLACK
    else:
        return None

    site = h.get("Site", "")
    game_id = site.rstrip("/").split("/")[-1] if site else f"{white}-{black}-{h.get('Date', '')}"
    res = h.get("Result", "*")
    if res == "1/2-1/2":
        result = "draw"
    elif res in ("1-0", "0-1"):
        result = "win" if (res == "1-0") == (player == chess.WHITE) else "loss"
    else:
        result = "unknown"
    me, opp = ("WhiteElo", "BlackElo") if player == chess.WHITE else ("BlackElo", "WhiteElo")

    ag = AnalyzedGame(
        id=game_id, url=site, white=white, black=black,
        color="white" if player == chess.WHITE else "black", result=result,
        opening=h.get("Opening", "Unknown"), eco=h.get("ECO", ""),
        rating=_int(h.get(me)), opp_rating=_int(h.get(opp)), date=h.get("UTCDate", h.get("Date", "")),
    )

    board = game.board()
    nodes = list(game.mainline())
    boards = [board.copy(stack=False)]
    evals = [an.evaluate(board)]
    for i, node in enumerate(nodes):
        board.push(node.move)
        boards.append(board.copy(stack=False))
        evals.append(an.evaluate(board))
        if progress:
            progress(i + 1, len(nodes))

    for i, node in enumerate(nodes):
        b = boards[i]
        if b.turn != player:
            continue
        sign = 1 if player == chess.WHITE else -1
        wp_b = win_pct(sign * evals[i].cp_white)
        wp_a = win_pct(sign * evals[i + 1].cp_white)
        best_uci = evals[i].pv[0] if evals[i].pv else None
        played_uci = node.move.uci()
        drop = 0.0 if played_uci == best_uci else wp_b - wp_a
        best_san = b.san(chess.Move.from_uci(best_uci)) if best_uci else None
        ph = phase(b, cfg)
        severity = classify(wp_b, wp_b - drop, cfg)
        ag.moves.append(MoveRecord(
            game_id=game_id, url=site, ply=i + 1, move_number=b.fullmove_number,
            color=ag.color, san=b.san(node.move), uci=played_uci, fen_before=b.fen(),
            wp_before=round(wp_b, 1), wp_after=round(wp_b - drop, 1), drop=round(drop, 1),
            best_san=best_san, best_uci=best_uci, phase=ph,
            endgame_type=endgame_type(b) if ph == "endgame" else None,
            clock_s=node.clock(), severity=severity, opening=ag.opening,
        ))
    return ag


# ---------------------------------------------------------------------------
# Explaining one position (facts for the LLM; no opinions)
# ---------------------------------------------------------------------------

def _attacker_value(board: chess.Board, sq: int) -> int:
    return VALUES[board.piece_type_at(sq)]


def _hanging(board: chess.Board, color: chess.Color) -> dict[int, str]:
    """Non-pawn pieces of `color` that the opponent can win by capturing."""
    out = {}
    for sq, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type in (chess.PAWN, chess.KING):
            continue
        attackers = board.attackers(not color, sq)
        if not attackers:
            continue
        defenders = board.attackers(color, sq)
        cheapest = min(_attacker_value(board, a) for a in attackers)
        if not defenders:
            out[sq] = f"{PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)} is attacked and undefended"
        elif cheapest < VALUES[piece.piece_type]:
            out[sq] = (f"{PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)} "
                       f"can be captured by a lower-value piece")
    return out


def _pinned(board: chess.Board, color: chess.Color) -> dict[int, str]:
    out = {}
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type not in (chess.PAWN, chess.KING):
            if board.is_pinned(color, sq):
                out[sq] = f"{PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)} is pinned to the king"
    return out


def explain(an: Analyzer, fen: str, played_uci: str, line_plies: int = 5) -> dict:
    board = chess.Board(fen)
    played = chess.Move.from_uci(played_uci)
    if played not in board.legal_moves:
        raise ValueError(f"{played_uci} is not a legal move in this position.")
    color = board.turn
    sign = 1 if color == chess.WHITE else -1

    ev0 = an.evaluate(board)
    best = chess.Move.from_uci(ev0.pv[0]) if ev0.pv else None
    after = board.copy(stack=False)
    after.push(played)
    ev1 = an.evaluate(after)

    wp_b = win_pct(sign * ev0.cp_white)
    wp_a = win_pct(sign * ev1.cp_white)
    if best is not None and best == played:
        wp_a = wp_b
    facts: list[str] = []
    themes: list[str] = []

    # new hanging pieces / pins created by the move
    new_hang = {s: t for s, t in _hanging(after, color).items()
                if s not in _hanging(board, color)}
    for t in new_hang.values():
        facts.append(f"After the move, your {t}.")
    if new_hang:
        themes.append("hangingPiece")
    new_pins = {s: t for s, t in _pinned(after, color).items() if s not in _pinned(board, color)}
    for t in new_pins.values():
        facts.append(f"After the move, your {t}.")
    if new_pins:
        themes.append("pin")

    # the best move took something that the played move did not
    if best is not None and best != played and board.is_capture(best):
        if board.is_en_passant(best):
            captured = chess.PAWN
        else:
            captured = board.piece_type_at(best.to_square)
        facts.append(f"The best move was a capture of a {PIECE_NAMES[captured]} on "
                     f"{chess.square_name(best.to_square)}.")
        themes.append("hangingPiece")

    # mates
    mover_before, mover_after = sign * ev0.cp_white, sign * ev1.cp_white
    if mover_before >= MATE_CP - 100 and mover_after < MATE_CP - 100:
        facts.append("The best move led to a forced checkmate.")
        themes.append("mate")
    if mover_after <= -(MATE_CP - 100):
        facts.append("The move allows a forced checkmate against you.")
        themes.append("mate")

    # opponent's best reply forks several of your pieces
    refutation = ev1.pv[: max(1, line_plies - 1)]
    if refutation:
        reply = chess.Move.from_uci(refutation[0])
        if reply in after.legal_moves:
            b2 = after.copy(stack=False)
            b2.push(reply)
            targets = []
            for sq in b2.attacks(reply.to_square):
                p = b2.piece_at(sq)
                if p and p.color == color and p.piece_type != chess.PAWN and (
                        p.piece_type == chess.KING or VALUES[p.piece_type] >= 3):
                    targets.append(f"{PIECE_NAMES[p.piece_type]} on {chess.square_name(sq)}")
            if len(targets) >= 2:
                facts.append(f"The opponent's best reply attacks several of your pieces at once: "
                             f"{', '.join(targets)}.")
                themes.append("fork")

    # material after a few plies of best play
    swing = 0
    if ev1.pv:
        b3 = after.copy(stack=False)
        for u in ev1.pv[:4]:
            m = chess.Move.from_uci(u)
            if m not in b3.legal_moves:
                break
            b3.push(m)
        swing = material_balance(b3, color) - material_balance(board, color)
        if swing <= -2:
            facts.append(f"Within a few moves of best play you lose about {abs(swing)} points of material.")

    return {
        "fen": fen,
        "played_move": board.san(played),
        "win_pct_before": round(wp_b, 1),
        "win_pct_after": round(wp_a, 1),
        "win_pct_drop": round(wp_b - wp_a, 1),
        "best_move": board.san(best) if best else None,
        "best_move_uci": best.uci() if best else None,
        "best_line": san_line(board, ev0.pv[:line_plies]),
        "line_after_your_move": san_line(after, ev1.pv[: max(1, line_plies - 1)]),
        "material_change_in_best_line": swing,
        "tactical_facts": facts or ["No simple tactic was detected; the evaluation changed "
                                   "for positional reasons the tools cannot name."],
        "suggested_puzzle_themes": sorted(set(themes)),
    }
