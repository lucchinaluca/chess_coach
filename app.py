"""Streamlit front end:  streamlit run app.py"""
import os

import chess
import chess.svg
import streamlit as st

from coach.agent import run_coach
from coach.config import Config
from coach.tools import CoachTools

st.set_page_config(page_title="Chess coach", page_icon="♟️", layout="wide")
st.title("♟️ Chess coach")
st.caption("Stockfish judges your games. An AI coach explains the patterns. "
           "The AI never evaluates a position itself, and every move it mentions is checked "
           "against Stockfish's output.")

with st.sidebar:
    username = st.text_input("Lichess username")
    focus = st.selectbox("Focus", ["all", "opening", "endgame"])
    n_games = st.slider("Recent games", 5, 60, 20)
    api_key = st.text_input("Gemini API key", type="password",
                            value=os.environ.get("GEMINI_API_KEY", ""))
    sf_path = st.text_input("Stockfish path", value=Config().stockfish_path)
    depth = st.slider("Stockfish depth", 8, 18, 12)
    token = st.text_input("Lichess token (optional)", type="password")
    st.caption("Keys stay in this session and are never stored.")
    go = st.button("Analyse my games", type="primary")


def draw(moment: dict) -> str:
    board = chess.Board(moment["fen_before"])
    arrows = []
    if moment.get("played_uci"):
        m = chess.Move.from_uci(moment["played_uci"])
        arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#cc3333cc"))
    if moment.get("best_move_uci"):
        m = chess.Move.from_uci(moment["best_move_uci"])
        arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#2e9e44cc"))
    return chess.svg.board(board, arrows=arrows, orientation=board.turn, size=320)


if go:
    if not username.strip():
        st.error("Enter your Lichess username.")
    elif not api_key:
        st.error("Enter a Gemini API key (it is only used for this session).")
    else:
        cfg = Config()
        cfg.stockfish_path, cfg.depth, cfg.lichess_token = sf_path, depth, token
        tools = CoachTools(cfg, username=username.strip())
        try:
            with st.status("Working...", expanded=True) as status:
                tools.progress = lambda kind, i, n: status.update(label=f"Analysing game {i}/{n}...")
                request = (f"Please coach me. My Lichess username is {username.strip()}. "
                           f"Focus: {focus}. Analyse my last {n_games} games.")
                res = run_coach(tools, request, api_key, cfg,
                                on_event=lambda m: status.update(label=m))
                status.update(label="Done", state="complete")

            st.markdown(res.text)
            if not res.verified:
                st.warning("Some moves in this report could not be matched to Stockfish's output: "
                           + ", ".join(res.unverified))

            moments, seen = [], set()
            for name, _args, out in res.tool_log:
                if name == "scan_games":
                    for m in out.get("worst_moments", [])[:6]:
                        key = (m["game_id"], m["move_number"], m["played"])
                        if key not in seen:
                            seen.add(key)
                            moments.append(m)
            if moments:
                st.subheader("The moments behind the report")
                st.caption("Red arrow: what you played. Green arrow: Stockfish's choice.")
                cols = st.columns(3)
                for i, m in enumerate(moments[:6]):
                    with cols[i % 3]:
                        st.image(draw(m))
                        st.markdown(
                            f"**Move {m['move_number']}** ({m['phase']}): you played "
                            f"`{m['played']}`, best was `{m['best_move']}`. "
                            f"Winning chances {m['win_pct_before']}% → {m['win_pct_after']}%. "
                            f"[Open game]({m['url']})")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
        finally:
            tools.close()
else:
    st.info("Enter your Lichess username and a Gemini API key in the sidebar, then press "
            "**Analyse my games**.")
