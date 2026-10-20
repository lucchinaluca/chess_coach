"""Offline tests. Needs Stockfish (set STOCKFISH_PATH) but no network and no API key.

Run:  python -m pytest -q tests        or        python tests/test_offline.py
"""
import os
import sys
import tempfile

import chess
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coach import tools as tools_module                      # noqa: E402
from coach.agent import unverified_moves                     # noqa: E402
from coach.analysis import (Analyzer, annotate_game, endgame_type, phase,   # noqa: E402
                            win_pct, explain)
from coach.config import Config                              # noqa: E402
from coach.lichess import parse_games                        # noqa: E402

# Black (the player, "tester") hangs the queen on move 3: 3...Qh4?? 4.Nxh4
MOVES = "e4 e5 Nf3 Nc6 Bc4 Qh4 Nxh4 Nf6 d4 exd4 O-O Bc5 Re1 d6 Nf3 O-O".split()


def make_pgn() -> str:
    game = chess.pgn.Game()
    game.headers.update({"Event": "Test", "Site": "https://lichess.org/test0001",
                         "White": "opp", "Black": "tester", "Result": "1-0",
                         "WhiteElo": "1500", "BlackElo": "1400", "Opening": "Italian Game",
                         "ECO": "C50", "UTCDate": "2026.10.01"})
    node = game
    board = chess.Board()
    clock = 300.0
    for san in MOVES:
        move = board.parse_san(san)
        board.push(move)
        node = node.add_variation(move)
        clock -= 6
        node.set_clock(clock)
    return str(game) + "\n\n"


def cfg() -> Config:
    c = Config()
    c.depth = 10
    c.cache_path = os.path.join(tempfile.mkdtemp(), "cache.sqlite")
    return c


def test_helpers():
    assert win_pct(0) == 50.0
    assert win_pct(300) > win_pct(100) > 50 > win_pct(-100)
    c = Config()
    assert phase(chess.Board(), c) == "opening"
    mid = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 11")
    assert phase(mid, c) == "middlegame"
    assert phase(chess.Board("8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40"), c) == "endgame"
    assert endgame_type(chess.Board("8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40")) == "rook endgame"
    assert endgame_type(chess.Board("8/5pk1/8/8/8/8/5PK1/8 w - - 0 40")) == "pawn endgame"


def test_move_checker():
    tool_text = '{"played": "Qh4", "best_move": "Nf6", "best_line": "3... Nf6 4. d4 Bxd4"}'
    assert unverified_moves("You played Qh4; Nf6 was better.", tool_text) == []
    assert unverified_moves("Bxf7+ would have won.", tool_text) == ["Bxf7+"]
    assert unverified_moves("Look at e4 and d5 squares.", tool_text) == []     # squares ignored


def test_annotation_finds_the_blunder():
    c = cfg()
    an = Analyzer(c)
    try:
        games = parse_games(make_pgn(), "tester")
        assert len(games) == 1
        ag = annotate_game(games[0], "tester", an, c)
        assert ag.color == "black" and ag.result == "loss" and ag.rating == 1400
        blunders = [m for m in ag.moves if m.severity == "blunder"]
        assert blunders, [(m.san, m.drop) for m in ag.moves]
        worst = max(blunders, key=lambda m: m.drop)
        assert worst.san == "Qh4" and worst.phase == "opening" and worst.clock_s is not None
        print("worst:", worst.san, "best was", worst.best_san, "drop", worst.drop)

        info = explain(an, worst.fen_before, worst.uci)
        assert any("queen" in f for f in info["tactical_facts"]), info["tactical_facts"]
        assert "hangingPiece" in info["suggested_puzzle_themes"]
        print("facts:", info["tactical_facts"])
        print("best line:", info["best_line"], "| after:", info["line_after_your_move"])
    finally:
        an.close()


def test_scan_and_tools_end_to_end(monkeypatch=None):
    c = cfg()
    pgn = make_pgn()
    tools_module.fetch_games_pgn = lambda *a, **k: pgn          # fake the Lichess download
    t = tools_module.CoachTools(c, username="tester")
    try:
        s = t.scan_games("tester", 5, "all")
        assert s["games_analyzed"] == 1 and s["results"] == {"loss": 1}
        assert s["worst_moments"][0]["played"] == "Qh4"
        assert s["by_phase"]["opening"]["blunders"] >= 1
        assert s["openings_with_most_early_mistakes"][0]["opening"] == "Italian Game"
        assert t.scan_games("tester", 5, "endgame")["worst_moments"] == []

        g = t.analyze_game("test0001")
        assert g["moments"][0]["played"] == "Qh4"
        assert "error" in t.analyze_game("nope")

        m = s["worst_moments"][0]
        e = t.dispatch("explain_position", {"fen": m["fen_before"], "played_move_uci": m["played_uci"]})
        assert e["best_move"] and e["win_pct_drop"] > 20
        assert "error" in t.dispatch("explain_position", {"fen": m["fen_before"], "played_move_uci": "a1a8"})
        assert "error" in t.dispatch("get_practice", {"theme": "fork"})       # no puzzle CSV set

        csv_path = os.path.join(tempfile.mkdtemp(), "p.csv")
        with open(csv_path, "w") as f:
            f.write("PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n")
            for i in range(5):
                f.write(f"ab{i},x,x,{1400 + i * 10},50,90,100,fork short,u,\n")
        c.puzzle_csv = csv_path
        p = t.get_practice("fork", 1400, 3)
        assert len(p["puzzles"]) == 3 and p["puzzles"][0]["url"].startswith("https://lichess.org/training/")
    finally:
        t.close()




# ---------------------------------------------------------------------------
# Agent loop with a scripted fake Anthropic client (no API key, no network)
# ---------------------------------------------------------------------------
class _Block:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason = content, stop_reason


def _last_tool_json(messages):
    import json
    for msg in reversed(messages):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            return json.loads(msg["content"][0]["content"])
    raise AssertionError("no tool result yet")


def test_agent_loop_with_fake_llm():
    import types
    seen = {"calls": 0, "retry_prompt": None}

    def step(messages):
        seen["calls"] += 1
        n = seen["calls"]
        if n == 1:
            return _Resp([_Block("tool_use", id="t1", name="scan_games",
                                 input={"username": "tester", "n_games": 5, "focus": "all"})], "tool_use")
        if n == 2:
            m = _last_tool_json(messages)["worst_moments"][0]
            return _Resp([_Block("tool_use", id="t2", name="explain_position",
                                 input={"fen": m["fen_before"], "played_move_uci": m["played_uci"]})], "tool_use")
        if n == 3:   # hallucinated move -> must be rejected once
            return _Resp([_Block("text", text="You should have played Bxf7+ instead of Qh4.")], "end_turn")
        seen["retry_prompt"] = messages[-1]["content"]
        return _Resp([_Block("text", text="You played Qh4, which hangs the queen. Nf6 was better.")], "end_turn")

    class _Messages:
        def create(self, **kw):
            return step(kw["messages"])

    fake = types.SimpleNamespace(Anthropic=lambda api_key=None: types.SimpleNamespace(messages=_Messages()))
    sys.modules["anthropic"] = fake

    from coach.agent import run_coach
    c = cfg()
    pgn = make_pgn()
    tools_module.fetch_games_pgn = lambda *a, **k: pgn
    t = tools_module.CoachTools(c, username="tester")
    try:
        res = run_coach(t, "coach me", "fake-key", c)
    finally:
        t.close()
    assert res.tool_calls == 2 and res.verified and "Bxf7+" in seen["retry_prompt"]
    assert "Qh4" in res.text and [x[0] for x in res.tool_log] == ["scan_games", "explain_position"]
    print("agent loop ok:", res.text)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
