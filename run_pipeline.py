"""Coaching report with at most ONE LLM call.

  python run_pipeline.py YOUR_NAME                          # Stockfish-only report, no AI, no key
  python run_pipeline.py YOUR_NAME --backend ollama         # local model (ollama must be running)
  python run_pipeline.py YOUR_NAME --backend gemini         # needs GEMINI_API_KEY (1 request)
"""
import argparse
import os
import sys

from coach.config import Config
from coach.pipeline import gather_facts, write_report
from coach.tools import CoachTools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("--backend", choices=["template", "ollama", "gemini"], default="template")
    ap.add_argument("--focus", choices=["all", "opening", "endgame"], default="all")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--stockfish")
    ap.add_argument("--model", default="", help="Ollama or Gemini model name")
    ap.add_argument("--retry", action="store_true", help="one extra LLM call if moves fail verification")
    args = ap.parse_args()

    cfg = Config()
    if args.depth:
        cfg.depth = args.depth
    if args.stockfish:
        cfg.stockfish_path = args.stockfish

    key = os.environ.get("GEMINI_API_KEY", "")
    if args.backend == "gemini" and not key:
        sys.exit("Set GEMINI_API_KEY first, or use --backend template / ollama.")

    tools = CoachTools(cfg, username=args.username,
                       progress=lambda k, i, n: print(f"\r  analysing game {i}/{n}...", end="", file=sys.stderr, flush=True))
    try:
        facts = gather_facts(tools, args.username, args.focus, args.games,
                             say=lambda m: print(f"\n  {m}", file=sys.stderr))
        res = write_report(facts, args.backend, api_key=key, model=args.model, retry=args.retry)
    finally:
        tools.close()

    print(file=sys.stderr)
    print(res.text)
    if res.note:
        print(f"\n[note] {res.note}")
    if not res.verified:
        print(f"\n[warning] moves not found in Stockfish's output: {', '.join(res.unverified)}")


if __name__ == "__main__":
    main()
