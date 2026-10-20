"""Command-line coach.

  python run_cli.py YOUR_LICHESS_NAME --no-llm            # Stockfish scan only, no API key
  python run_cli.py YOUR_LICHESS_NAME --focus endgame     # full coaching report
"""
import argparse
import json
import os
import sys

from coach.agent import run_coach
from coach.config import Config
from coach.tools import CoachTools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("--focus", choices=["all", "opening", "endgame"], default="all")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--stockfish")
    ap.add_argument("--no-llm", action="store_true", help="print the Stockfish scan only")
    ap.add_argument("--game", metavar="GAME_ID",
                    help="focus on one specific game (Lichess game ID or full URL)")
    ap.add_argument("--note", metavar="TEXT",
                    help="extra context sent to the coach, e.g. 'I was low on time' or "
                         "'explain the endgame in detail'")
    args = ap.parse_args()

    cfg = Config()
    if args.depth:
        cfg.depth = args.depth
    if args.stockfish:
        cfg.stockfish_path = args.stockfish

    def progress(kind, i, n):
        print(f"\r  analysing game {i}/{n}...", end="", file=sys.stderr, flush=True)

    tools = CoachTools(cfg, username=args.username, progress=progress)
    try:
        if args.no_llm:
            result = tools.scan_games(args.username, args.games, args.focus)
            print(file=sys.stderr)
            print(json.dumps(result, indent=2))
            return

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Set GEMINI_API_KEY, or use --no-llm to see the Stockfish scan only.")
        request = (f"Please coach me. My Lichess username is {args.username}. "
                   f"Focus: {args.focus}. Analyse my last {args.games} games.")
        if args.game:
            game_id = args.game.rstrip("/").split("/")[-1]  # accept full URL or bare ID
            request += f" I want you to focus specifically on game {game_id} — use analyze_game for it."
        if args.note:
            request += f" Extra context from me: {args.note}"
        res = run_coach(tools, request, key, cfg, on_event=lambda m: print(" ", m, file=sys.stderr))
        print(file=sys.stderr)
        print(res.text)
        if not res.verified:
            print(f"\n[warning] unverified moves in the report: {', '.join(res.unverified)}")
        print(f"\n({res.tool_calls} tool calls)")
    finally:
        tools.close()


if __name__ == "__main__":
    main()
