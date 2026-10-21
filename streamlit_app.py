"""Streamlit front end for the one-call pipeline:  streamlit run app_pipeline.py

The default writer needs no AI and no key, so a public demo always works.
"""
import os

import chess
import chess.svg
import streamlit as st

from coach.config import Config
from coach.pipeline import gather_facts, write_report
from coach.tools import CoachTools

st.set_page_config(page_title="Chess coach", page_icon="♟️", layout="wide")
st.title("♟️ Chess coach")
st.caption("Stockfish finds your biggest mistakes. An optional AI writer explains the pattern. "
           "Every move in an AI report is checked against Stockfish's output.")

WRITERS = {"Stockfish only (no AI, no key)": "template", "Gemini (your key)": "gemini",
           "Ollama (local model)": "ollama"}

with st.sidebar:
    username = st.text_input("Lichess username", placeholder="e.g. lucchina")
    focus = st.selectbox("Focus", ["all", "opening", "endgame"])
    n_games = st.slider("Recent games", 5, 60, 20)
    writer = st.selectbox("Report writer", list(WRITERS))
    backend = WRITERS[writer]
    api_key = ""
    model = ""
    if backend == "gemini":
        api_key = st.text_input("Gemini API key", type="password",
                                value=os.environ.get("GEMINI_API_KEY", ""))
        st.caption("Get a free key at [aistudio.google.com](https://aistudio.google.com/apikey)")
        model = st.text_input("Gemini model", value=Config().model)
    if backend == "ollama":
        model = st.text_input("Ollama model", value=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"))
    sf_path = st.text_input("Stockfish path", value=Config().stockfish_path)
    depth = st.slider("Stockfish depth", 8, 18, 10)
    st.caption("Keys stay in this session and are never stored.")
    go = st.button("Analyse my games", type="primary")


def draw(m: dict) -> str:
    board = chess.Board(m["fen_before"])
    arrows = []
    for key, color in (("played_uci", "#cc3333cc"), ("best_move_uci", "#2e9e44cc")):
        if m.get(key):
            mv = chess.Move.from_uci(m[key])
            arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=color))
    return chess.svg.board(board, arrows=arrows, orientation=board.turn, size=320)


if go:
    if not username.strip():
        st.error("Enter your Lichess username.")
    elif backend == "gemini" and not api_key:
        st.error("Enter a Gemini API key, or choose the Stockfish-only writer.")
    else:
        cfg = Config()
        cfg.stockfish_path, cfg.depth = sf_path, depth
        tools = CoachTools(cfg, username=username.strip())
        try:
            with st.status("Working...", expanded=True) as status:
                tools.progress = lambda k, i, n: status.update(label=f"Analysing game {i}/{n}...")
                facts = gather_facts(tools, username.strip(), focus, n_games,
                                     say=lambda m: status.update(label=m))
                if "error" not in facts and backend != "template":
                    status.update(label="Writing the report...")
                res = write_report(facts, backend, api_key=api_key, model=model)
                status.update(label="Done", state="complete")

            if res.note:
                st.warning(res.note)
            st.markdown(res.text)
            if not res.verified:
                st.warning("Some moves in this report could not be matched to Stockfish's output: "
                           + ", ".join(res.unverified))

            if "error" not in facts:
                moments = facts["scan"].get("worst_moments", [])[:6]
                if moments:
                    st.subheader("The biggest mistakes")
                    st.caption("Red arrow: what you played. Green arrow: Stockfish's choice.")
                    cols = st.columns(3)
                    for i, m in enumerate(moments):
                        with cols[i % 3]:
                            st.image(draw(m))
                            st.markdown(f"**Move {m['move_number']}** ({m['phase']}): you played "
                                        f"`{m['played']}`, best was `{m['best_move']}`. "
                                        f"Winning chances {m['win_pct_before']}% → {m['win_pct_after']}%. "
                                        f"[Open game]({m['url']})")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
        finally:
            tools.close()
else:
    st.info("Enter your Lichess username in the sidebar, then press **Analyse my games**. "
            "The default writer needs no API key.")
