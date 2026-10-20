# Chess Coach Agent

Stockfish judges your Lichess games. An LLM turns the results into coaching. The LLM never evaluates a
position itself: it only decides what to look at and explains facts that tools return.

```
Your request  ->  Coach agent (LLM)  ->  Coaching report
                       |
        scan_games / analyze_game / explain_position / get_practice
                       |
              Stockfish + Lichess API  (all chess judgement lives here)
```

## 1. Setup (10 minutes)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Stockfish.** Download the engine (stockfishchess.org), then point the project at the executable:

```powershell
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"      # macOS/Linux: export STOCKFISH_PATH=/usr/games/stockfish
```

**Check it works** (no internet and no API key needed):

```bash
python tests/test_offline.py
```

You should see `PASS` five times. These tests use real Stockfish on a synthetic game with a
deliberate queen blunder.

## 2. Build-and-use order

Work through these in order. Each step works on its own.

1. **Stockfish scan only (no API key).** Prints what the agent will see:
   ```bash
   python run_cli.py YOUR_LICHESS_NAME --no-llm --games 20
   ```
   First run analyses every position (about a minute per ten games at depth 12 on a laptop, depending on
   your CPU). Results are cached in `coach_cache.sqlite`, so re-runs are nearly instant.
2. **Full report in the terminal:**
   ```powershell
   $env:GEMINI_API_KEY = "AIza..."
   python run_cli.py YOUR_LICHESS_NAME --focus endgame
   ```
3. **Web app with boards:**
   ```bash
   streamlit run app.py
   ```
4. **Optional, practice links:** download `lichess_db_puzzle.csv.zst` from database.lichess.org,
   decompress it, then `$env:PUZZLE_CSV = "C:\path\lichess_db_puzzle.csv"`. Without it the report simply
   skips puzzle links.

## 3. The tools

| Tool | Input | Returns |
|---|---|---|
| `scan_games` | username, n_games, focus (`all`, `opening`, `endgame`) | Mistakes and blunders per phase, average loss, clock habits, openings and endgame types linked to mistakes, 10 worst moments |
| `analyze_game` | game_id, focus | Biggest mistakes in one scanned game |
| `explain_position` | fen, played move (UCI) | Best line, line after your move, win % before and after, tactical facts, suggested puzzle themes |
| `get_practice` | theme, rating | Links to a few Lichess puzzles near your rating |

Tool outputs are small and structured. Advice is written only by the LLM.

## 4. How a mistake is decided

Centipawns are converted to a win probability, and a move is flagged by how many points of win probability it
loses. Defaults in `coach/config.py`:

- mistake: lose 20 or more points, blunder: 30 or more
- not flagged if you were already clearly winning (92% or more before, 70% or more after) or already lost (8% or less)
- phases: opening up to move 10, endgame when combined non-pawn material is 26 or less, otherwise middlegame

Tune these against your own games. Depth (`COACH_DEPTH`, default 12) trades speed for accuracy.

## 5. Guards against made-up chess

- The system prompt forbids judging positions and requires every move to come from a tool result.
- `agent.py` extracts the moves in the final report and checks each against the tool output. If any are
  missing, the model gets one chance to rewrite; if it still fails, the UI shows a warning.
  Two-character tokens (pawn pushes, plain squares) are not checked because they can't be told apart from
  squares in ordinary sentences.
- Hard cap of 8 tool calls per request.

## 6. What has and hasn't been tested

Tested here with real Stockfish 16: blunder detection, phase and endgame tagging, tactical facts
(hanging piece, pins, forks, missed captures, mates, material loss), caching, all four tools, the agent loop
with a scripted fake LLM (including the hallucinated-move retry), and the Streamlit app running headless.

**Not tested:** the live Lichess download and real Gemini calls (no network access to either from the
build environment). Expect to fix small things on the first real run: the most likely are the Lichess
request parameters in `coach/lichess.py` and the model name (`COACH_MODEL`, default `gemini-2.5-flash`).
The tactical tags are simple heuristics and can miss or over-report; positional mistakes get no named reason.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Stockfish not found` | Set `STOCKFISH_PATH` to the executable |
| `Lichess user not found` | Check the spelling; the account must have public games |
| Lichess rate limit | The downloader waits a minute and retries; use a smaller `--games`, or set `LICHESS_TOKEN` |
| Very slow first run | Lower depth (`COACH_DEPTH=10`) or fewer games; later runs use the cache |
| No clock data in the report | Only games with clocks (not correspondence) include it |
| Report mentions unverified moves | Shown as a warning; the check is deliberately strict |

## 8. Putting it online

- Do not hard-code your own API key in a public app. The Streamlit app asks each visitor for theirs.
- Stockfish needs a server that can run it. Check the CPU and time limits of any free host before relying on it.
- For a demo post, a short screen recording and a screenshot of a real report on your own games works best.

## 9. Ideas for later

- Progress tracking: store results by date and compare the last 30 games with the previous 30.
- Endgame tablebase lookups for simple endings (check the current Lichess tablebase limits first).
- A "how natural was this move" signal from a model trained on human games.
