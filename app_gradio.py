"""Gradio front end for the one-call pipeline (for Hugging Face Spaces).

  python app_gradio.py          # then open the local URL it prints

The default writer needs no AI and no key, so a public demo always works.
"""
import html
import os
import shutil
import tempfile

# Keep the evaluation cache somewhere writable on hosted machines (must be set before Config is built).
os.environ.setdefault("COACH_CACHE", os.path.join(tempfile.gettempdir(), "coach_cache.sqlite"))

import chess
import chess.svg
import gradio as gr

from coach.config import Config
from coach.pipeline import gather_facts, write_report
from coach.tools import CoachTools

MAX_GAMES = int(os.environ.get("MAX_GAMES", "15"))
WRITERS = {"Stockfish only (no AI, no key)": "template", "Gemini (your own key)": "gemini"}


def find_stockfish() -> str:
    """STOCKFISH_PATH if set, else PATH, else the usual Linux install locations."""
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env
    for cand in (shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish",
                 "/usr/local/bin/stockfish"):
        if cand and os.path.isfile(cand):
            return cand
    return "stockfish"


def board_svg(m: dict) -> str:
    board = chess.Board(m["fen_before"])
    arrows = []
    for key, color in (("played_uci", "#cc3333cc"), ("best_move_uci", "#2e9e44cc")):
        if m.get(key):
            mv = chess.Move.from_uci(m[key])
            arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=color))
    return chess.svg.board(board, arrows=arrows, orientation=board.turn, size=300)


def boards_html(moments: list) -> str:
    if not moments:
        return ""
    esc = html.escape
    cards = []
    for m in moments:
        caption = (f"<b>Move {m['move_number']}</b> ({esc(m['phase'])}): you played "
                   f"<code>{esc(str(m['played']))}</code>, best was <code>{esc(str(m['best_move']))}</code>. "
                   f"Winning chances {m['win_pct_before']}% &rarr; {m['win_pct_after']}%. "
                   f"<a href=\"{esc(m['url'], quote=True)}\" target=\"_blank\" rel=\"noopener\">Open game</a>")
        cards.append('<div style="width:300px;font-size:14px;line-height:1.4">'
                     f'{board_svg(m)}<div style="margin-top:6px">{caption}</div></div>')
    return ('<h3>The biggest mistakes</h3><p>Red arrow: what you played. Green arrow: Stockfish\'s choice.</p>'
            '<div style="display:flex;flex-wrap:wrap;gap:20px">' + "".join(cards) + "</div>")


def analyse(username, focus, n_games, writer, api_key, model, depth, sf_path,
            progress=gr.Progress()):
    username = (username or "").strip()
    backend = WRITERS.get(writer, "template")
    if not username:
        return "Enter your Lichess username.", ""
    if backend == "gemini" and not (api_key or "").strip():
        return "Enter a Gemini API key, or choose the Stockfish-only writer.", ""

    cfg = Config()
    cfg.depth = int(depth)
    cfg.stockfish_path = (sf_path or "").strip() or find_stockfish()
    tools = None
    try:
        tools = CoachTools(cfg, username=username)
        tools.progress = lambda kind, i, n: progress(0.8 * i / max(n, 1), desc=f"Analysing game {i}/{n}")
        facts = gather_facts(tools, username, focus, int(n_games),
                             say=lambda m: progress(0.85, desc=m))
        if "error" not in facts and backend != "template":
            progress(0.92, desc="Writing the report...")
        res = write_report(facts, backend, api_key=(api_key or "").strip(), model=(model or "").strip())
    except Exception as e:
        return f"**Something went wrong:** {type(e).__name__}: {e}", ""
    finally:
        if tools is not None:
            tools.close()

    text = res.text
    if res.note:
        text = f"> **Note:** {res.note}\n\n" + text
    if not res.verified:
        text += ("\n\n> **Warning:** some moves in this report could not be matched to Stockfish's output: "
                 + ", ".join(res.unverified))
    moments = [] if "error" in facts else facts["scan"].get("worst_moments", [])[:6]
    return text, boards_html(moments)


with gr.Blocks(title="Chess coach") as demo:
    gr.Markdown("# ♟️ Chess coach\n"
                "Stockfish finds your biggest mistakes in your recent Lichess games. An optional AI writer "
                "explains the pattern, and every move it mentions is checked against Stockfish's output.")
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            username = gr.Textbox(label="Lichess username", placeholder="e.g. DrNykterstein")
            focus = gr.Dropdown(["all", "opening", "endgame"], value="all", label="Focus")
            n_games = gr.Slider(5, MAX_GAMES, value=min(10, MAX_GAMES), step=1, label="Recent games")
            writer = gr.Dropdown(list(WRITERS), value=list(WRITERS)[0], label="Report writer")
            api_key = gr.Textbox(label="Gemini API key", type="password", visible=False,
                                 info="Used for this request only and never stored.")
            with gr.Accordion("Advanced", open=False):
                model = gr.Textbox(label="Gemini model", value=Config().model)
                depth = gr.Slider(8, 14, value=int(os.environ.get("COACH_DEPTH", "10")), step=1,
                                  label="Stockfish depth")
                sf_path = gr.Textbox(label="Stockfish path", value=find_stockfish())
            go = gr.Button("Analyse my games", variant="primary")
        with gr.Column(scale=3):
            report = gr.Markdown("Enter a Lichess username and press **Analyse my games**. "
                                 "The default writer needs no API key.")
            boards = gr.HTML()

    writer.change(lambda w: gr.update(visible=WRITERS[w] == "gemini"), writer, api_key)
    go.click(analyse, [username, focus, n_games, writer, api_key, model, depth, sf_path], [report, boards])

demo.queue(max_size=10, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
