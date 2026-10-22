"""Streamlit front end for the chess coach pipeline.

Two analysis modes:
  • Lichess   — fetch games by username from Lichess, analyse with Stockfish + AI
  • PGN       — paste / upload a PGN, analyse with Stockfish + AI

Ollama reports are streamed token-by-token so the UI stays responsive.
Gemini and the template writer finish in one shot.
"""
from __future__ import annotations

import json
import os

import chess
import chess.svg
import streamlit as st

from coach.config import Config
from coach.pipeline import (
    BackendError,
    REPORT_PROMPT,
    SINGLE_GAME_PROMPT,
    compact_for_llm,
    gather_facts,
    gather_facts_from_pgn,
    stream_ollama,
    template_report,
    unverified_moves,
    write_report,
)
from coach.tools import CoachTools

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Move-Mentor", page_icon="♟️", layout="wide")
st.title("♟️ Move-Mentor")
st.caption(
    "Stockfish finds your biggest mistakes. An optional AI writer explains the pattern. "
    "Every move in an AI report is checked against Stockfish's output."
)

# ---------------------------------------------------------------------------
# Sidebar — shared settings
# ---------------------------------------------------------------------------
WRITERS = {
    "Stockfish only (no AI, no key)": "template",
    "Gemini (your key)": "gemini",
    "Ollama (local)": "ollama",
}

with st.sidebar:
    st.header("Settings")

    focus = st.selectbox("Focus", ["all", "opening", "endgame"])
    writer = st.selectbox("Report writer", list(WRITERS))
    backend = WRITERS[writer]

    api_key = ""
    model = ""
    ollama_url = ""

    if backend == "gemini":
        api_key = st.text_input(
            "Gemini API key", type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
        )
        st.caption("Get a free key at [aistudio.google.com](https://aistudio.google.com/apikey)")
        model = st.text_input("Gemini model", value=Config().model)

    if backend == "ollama":
        st.info(
            "Ollama only works when you are running Streamlit locally "
            "on the same machine as Ollama. Output streams in real time."
        )
        model = st.text_input(
            "Ollama model", value=os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        )

    st.divider()
    sf_path = Config().stockfish_path
    depth = st.slider("Stockfish depth", 8, 18, 10)
    st.caption("Keys stay in this session and are never stored.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg() -> Config:
    cfg = Config()
    cfg.stockfish_path = sf_path
    cfg.depth = depth
    return cfg


def draw_board(m: dict) -> str:
    """Return an SVG chess board for a moment dict."""
    board = chess.Board(m["fen_before"])
    arrows = []
    for key, color in (("played_uci", "#cc3333cc"), ("best_move_uci", "#2e9e44cc")):
        if m.get(key):
            mv = chess.Move.from_uci(m[key])
            arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=color))
    return chess.svg.board(board, arrows=arrows, orientation=board.turn, size=320)


def render_results(facts: dict, report_text: str, verified: bool, unverified_list: list) -> None:
    """Render the coaching report and worst-moment boards."""
    if report_text:
        st.markdown(report_text)
        if not verified:
            st.warning(
                "Some moves in this report could not be matched to Stockfish's output: "
                + ", ".join(unverified_list)
            )

    if "error" not in facts:
        moments = facts["scan"].get("worst_moments", [])[:6]
        if moments:
            st.subheader("The biggest mistakes")
            st.caption("Red arrow: what you played · Green arrow: Stockfish's choice")
            cols = st.columns(3)
            for i, m in enumerate(moments):
                with cols[i % 3]:
                    st.image(draw_board(m))
                    st.markdown(
                        f"**Move {m['move_number']}** ({m['phase']}): "
                        f"you played `{m['played']}`, best was `{m['best_move']}`. "
                        f"Winning chances {m['win_pct_before']}% → {m['win_pct_after']}%. "
                        f"[Open game]({m['url']})"
                    )


def run_ollama_streaming(facts: dict, model_name: str, url: str,
                         prompt: str = REPORT_PROMPT) -> tuple[str, bool, list]:
    """Stream an Ollama report directly into the Streamlit page.

    Returns (full_text, verified, unverified_moves).
    """
    sheet = json.dumps(compact_for_llm(facts), indent=1)
    user_prompt = f"FACT SHEET (JSON):\n{sheet}\n\nWrite the coaching report now."

    st.subheader("Coaching report")
    placeholder = st.empty()
    full_text = ""

    try:
        for token in stream_ollama(prompt, user_prompt, model_name, url):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except BackendError as e:
        st.warning(f"{e} Showing the Stockfish-only report instead.")
        full_text = template_report(facts)
        placeholder.markdown(full_text)

    bad = unverified_moves(full_text, sheet)
    return full_text, not bad, bad


def run_report(facts: dict, prompt: str = REPORT_PROMPT) -> tuple[str, bool, list]:
    """Generate the report for non-streaming backends (template / Gemini)."""
    res = write_report(facts, backend, api_key=api_key, model=model,
                       ollama_url=ollama_url, system_prompt=prompt)
    if res.note:
        st.warning(res.note)
    return res.text, res.verified, res.unverified


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_lichess, tab_pgn = st.tabs(["🌐 Lichess username", "📋 Paste / upload PGN"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Lichess
# ══════════════════════════════════════════════════════════════════════════════
with tab_lichess:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        username = st.text_input("Lichess username", placeholder="e.g. lucchina", key="lc_user")
    with col_r:
        n_games = st.slider("Recent games", 5, 60, 20, key="lc_ngames")

    go_lichess = st.button("Analyse my games", type="primary", key="go_lichess")

    if go_lichess:
        if not username.strip():
            st.error("Enter your Lichess username.")
        elif backend == "gemini" and not api_key:
            st.error("Enter a Gemini API key, or choose the Stockfish-only writer.")
        else:
            cfg = make_cfg()
            tools = CoachTools(cfg, username=username.strip())
            try:
                # ── gather facts (Stockfish, no LLM) ─────────────────────
                with st.status("Working…", expanded=True) as status:
                    tools.progress = lambda k, i, n: status.update(
                        label=f"Analysing game {i}/{n}…"
                    )
                    facts = gather_facts(
                        tools, username.strip(), focus, n_games,
                        say=lambda m: status.update(label=m),
                    )
                    if "error" in facts:
                        status.update(label="Error", state="error")
                        st.error(facts["error"])
                        st.stop()
                    status.update(label="Stockfish done ✓", state="complete")

                # ── write report (LLM or template) ────────────────────────
                if backend == "ollama":
                    report_text, verified, bad = run_ollama_streaming(
                        facts, model or "llama3.2:3b",
                        ollama_url or "http://localhost:11434",
                    )
                else:
                    with st.spinner("Writing the report…" if backend != "template" else ""):
                        report_text, verified, bad = run_report(facts)

                render_results(facts, report_text, verified, bad)

            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
            finally:
                tools.close()
    else:
        st.info(
            "Enter your Lichess username in the sidebar and press **Analyse my games**. "
            "The default writer needs no API key."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PGN
# ══════════════════════════════════════════════════════════════════════════════
with tab_pgn:
    st.markdown(
        "Paste a PGN below **or** upload a `.pgn` file. "
        "Works with Chess.com exports, Lichess exports, or any standard PGN."
    )

    pgn_col, up_col = st.columns([3, 1])
    with pgn_col:
        pgn_text_input = st.text_area(
            "PGN",
            height=220,
            placeholder='[Event "..."]\n[White "Alice"]\n[Black "Bob"]\n\n1. e4 e5 2. Nf3 ...',
            key="pgn_text",
        )
    with up_col:
        uploaded = st.file_uploader("Or upload a .pgn file", type=["pgn"], key="pgn_upload")
        if uploaded is not None:
            pgn_text_input = uploaded.read().decode("utf-8", errors="replace")
            st.success(f"Loaded **{uploaded.name}**")

    pgn_user = st.text_input(
        "Your name in the PGN  (leave blank to auto-detect from White header)",
        placeholder="e.g. Alice",
        key="pgn_user",
    )

    go_pgn = st.button("Analyse PGN", type="primary", key="go_pgn")

    if go_pgn:
        pgn_str = pgn_text_input.strip()
        if not pgn_str:
            st.error("Paste a PGN or upload a file first.")
        elif backend == "gemini" and not api_key:
            st.error("Enter a Gemini API key, or choose the Stockfish-only writer.")
        else:
            cfg = make_cfg()
            tools = CoachTools(cfg, username=pgn_user.strip() or "Player")
            try:
                # ── gather facts (Stockfish, no LLM) ─────────────────────
                with st.status("Working…", expanded=True) as status:
                    facts = gather_facts_from_pgn(
                        pgn_str,
                        pgn_user.strip(),
                        tools,
                        focus=focus,
                        say=lambda m: status.update(label=m),
                    )
                    if "error" in facts:
                        status.update(label="Error", state="error")
                        st.error(facts["error"])
                        st.stop()
                    status.update(label="Stockfish done ✓", state="complete")

                # ── write report (LLM or template) ────────────────────────
                if backend == "ollama":
                    report_text, verified, bad = run_ollama_streaming(
                        facts, model or "llama3.2:3b",
                        ollama_url or "http://localhost:11434",
                        prompt=SINGLE_GAME_PROMPT,
                    )
                else:
                    with st.spinner("Writing the report…" if backend != "template" else ""):
                        report_text, verified, bad = run_report(facts, SINGLE_GAME_PROMPT)

                render_results(facts, report_text, verified, bad)

            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
            finally:
                tools.close()
    else:
        st.info(
            "Paste your PGN above and press **Analyse PGN**. "
            "For Chess.com games: **Game Archive → Export → PGN**."
        )
