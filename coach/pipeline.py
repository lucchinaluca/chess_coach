"""One-call coaching pipeline.

The agent loop in agent.py needs an LLM call for every tool round (about 4-7 per report).
This module does the orchestration in plain Python instead:

    gather_facts()   ->  runs scan_games, explain_position and get_practice itself (no LLM)
    write_report()   ->  turns the facts into a report with ONE of:
                           "template"  no AI at all, always works
                           "ollama"    a local model through Ollama (one request)
                           "gemini"    Google Gemini (one request)

Because the LLM only has to write from a fact sheet (no tool calling), even a small local model can do it,
and a free key with a tiny daily quota covers a whole report. Any backend failure falls back to the
template report, so the user always gets Stockfish's analysis.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from .agent import unverified_moves
from .config import Config
from .tools import CoachTools

REPORT_PROMPT = """You are a chess coach for club-level players (roughly 800-1800 rating).
You are given a FACT SHEET in JSON produced by Stockfish analysis of one player's real games.
Write a short coaching report from it.

RULES
- Use only the numbers and moves that appear in the fact sheet. Copy moves exactly as written there.
- Never invent moves, variations or evaluations. Name a tactic only if tactical_facts lists it.
- You may add a short opinion about why a move was bad, but start it with "My read:" and do not
  introduce any move that is not in the fact sheet.
- If a field is missing, skip it. Do not mention the JSON or the fact sheet.

FORMAT (about 350-450 words)
Pattern: 3-4 sentences with numbers from the summary (phase, clock, openings, endgame types).
Examples: one block per example: the game link, the move number, the move played, the better move,
the change in winning chances, then 2-3 sentences using the tactical facts and best lines.
Practise this week: one specific habit, plus the puzzle links if the sheet has them.
Use plain language and explain any chess term in a few words. Be direct and encouraging, not
flattering. If fewer than 10 games were analysed, say the pattern is tentative.
"""


class BackendError(Exception):
    pass


# ---------------------------------------------------------------------------
# 1. Facts (no LLM)
# ---------------------------------------------------------------------------

def pick_examples(moments: list[dict], n: int) -> list[dict]:
    """Biggest drops first, preferring different games."""
    chosen, used = [], set()
    for m in moments:
        if m["game_id"] not in used:
            chosen.append(m)
            used.add(m["game_id"])
            if len(chosen) == n:
                return chosen
    for m in moments:
        if m not in chosen:
            chosen.append(m)
            if len(chosen) == n:
                break
    return chosen


def gather_facts(tools: CoachTools, username: str, focus: str = "all", n_games: int = 30,
                 n_examples: int = 3, say: Optional[Callable[[str], None]] = None) -> dict:
    say = say or (lambda m: None)
    say("Scanning your games with Stockfish...")
    scan = tools.dispatch("scan_games", {"username": username, "n_games": n_games, "focus": focus})
    if "error" in scan:
        return {"error": scan["error"]}

    examples = []
    for i, m in enumerate(pick_examples(scan.get("worst_moments", []), n_examples), 1):
        say(f"Explaining mistake {i}...")
        e = tools.dispatch("explain_position", {"fen": m["fen_before"], "played_move_uci": m["played_uci"]})
        examples.append({"moment": m, "explanation": e})

    practice = None
    themes = Counter(t for ex in examples for t in ex["explanation"].get("suggested_puzzle_themes", []))
    if themes:
        theme = themes.most_common(1)[0][0]
        p = tools.dispatch("get_practice", {"theme": theme, "rating": scan.get("average_rating") or 1400, "n": 3})
        practice = None if "error" in p else p
    return {"scan": scan, "examples": examples, "practice": practice}


# ---------------------------------------------------------------------------
# 2. Report writers
# ---------------------------------------------------------------------------

@dataclass
class ReportResult:
    text: str
    backend_used: str
    verified: bool = True
    unverified: list = field(default_factory=list)
    note: str = ""


def compact_for_llm(facts: dict) -> dict:
    s = facts["scan"]
    keep = ["player", "games_analyzed", "results", "average_rating", "focus", "by_phase",
            "flagged_mistakes_in_focus", "clock", "openings_with_most_early_mistakes",
            "endgame_types_with_mistakes", "note"]
    examples = []
    for ex in facts["examples"]:
        m, e = ex["moment"], ex["explanation"]
        if "error" in e:
            continue
        item = {k: m[k] for k in ("url", "move_number", "phase", "played", "best_move",
                                  "win_pct_before", "win_pct_after", "severity") if k in m}
        for k in ("endgame_type", "clock_seconds_left"):
            if k in m:
                item[k] = m[k]
        item.update({"best_line": e.get("best_line"), "line_after_your_move": e.get("line_after_your_move"),
                     "tactical_facts": e.get("tactical_facts")})
        examples.append(item)
    out = {"summary": {k: s[k] for k in keep if k in s}, "examples": examples}
    if facts.get("practice"):
        out["practice_puzzles"] = [p["url"] for p in facts["practice"]["puzzles"]]
        out["practice_theme"] = facts["practice"]["theme"]
    return out


def _plural(n: int, word: str, plural: str = "") -> str:
    """'1 mistake', '2 mistakes', '1 loss', '2 losses'."""
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _drop_advice(severity: str, drop: float, phase: str) -> str:
    """Return a one-sentence coaching comment based on the severity of the error."""
    if severity == "blunder":
        if drop >= 40:
            return ("This is a decisive blunder — the position swung by more than 40 winning-chance points "
                    "in a single move, which is the difference between winning and losing.")
        return ("This is a blunder. A swing this large usually means a piece was left undefended, "
                "a fork was missed, or a basic tactical pattern was overlooked.")
    if severity == "mistake":
        if phase == "endgame":
            return ("In the endgame small inaccuracies matter a lot — this mistake let your opponent "
                    "consolidate or create a passed pawn that should not exist.")
        if phase == "opening":
            return ("Opening mistakes often hand the opponent free development or a pawn — "
                    "reviewing this line will help you avoid the same problem in future games.")
        return ("This kind of mistake often comes from calculating one move too few — "
                "the position looks safe but your opponent has a strong reply you didn't account for.")
    # generic fallback
    return ("Even when no simple tactic is visible, a move like this weakens your position "
            "subtly — look at the piece activity and pawn structure before and after.")


def _positional_advice(phase: str) -> str:
    """Fallback advice for moves where no concrete tactic was found."""
    if phase == "endgame":
        return ("No single tactic caused this — it is a positional drift. "
                "In endgames, focus on king activity and pawn advancement before trading pieces.")
    if phase == "opening":
        return ("No forcing tactic here — this is a positional error. "
                "In the opening, ask yourself: does this move develop a piece, control the centre, "
                "or improve king safety? If not, there is usually a better option.")
    return ("No concrete tactic is visible, but the position shifted for positional reasons — "
            "piece co-ordination, weak squares, or pawn structure. Open the game on the board link "
            "and compare where your pieces stand before and after this move.")


def template_report(facts: dict) -> str:
    s = facts["scan"]
    n = s["games_analyzed"]
    res = s.get("results", {})
    lines = ["*Stockfish-only report (no AI writer).*", "", "## Pattern"]
    games_txt = "your last game" if n == 1 else f"your last {n} games"
    record = (f"{_plural(res.get('win', 0), 'win')}, {_plural(res.get('draw', 0), 'draw')}, "
              f"{_plural(res.get('loss', 0), 'loss', 'losses')}")
    lines.append(f"Across {games_txt} ({record}) Stockfish flagged "
                 f"{_plural(s['flagged_mistakes_in_focus'], 'big mistake')}"
                 + (f" in the {s['focus']}." if s["focus"] != "all" else "."))

    phases = s["by_phase"]
    rates = {p: (v["mistakes"] + v["blunders"]) / v["player_moves"] for p, v in phases.items() if v["player_moves"]}
    if rates and s["focus"] == "all":
        worst = max(rates, key=rates.get)
        v = phases[worst]
        avg = v["avg_win_pct_loss_per_move"]
        lines.append(f"Your weakest phase is the **{worst}**: {_plural(v['blunders'], 'blunder')} and "
                     f"{_plural(v['mistakes'], 'mistake')} in {_plural(v['player_moves'], 'move')}, losing "
                     f"{avg} winning-chance points per move on average.")
        # Interpret the magnitude
        if avg >= 10:
            lines.append(f"Losing {avg} points per move in the {worst} is a serious pattern — "
                         "on average every other move contains a significant error. "
                         "This is where your rating points are going.")
        elif avg >= 5:
            lines.append(f"Losing {avg} points per move in the {worst} is above average for your level. "
                         "Focused practice on this phase should have a direct impact on your results.")
        else:
            lines.append(f"Losing {avg} points per move in the {worst} is modest but consistent — "
                         "small improvements here will compound over many games.")

    clock = s.get("clock")
    if clock:
        share = next((v for k, v in clock.items() if k.startswith("share_made")), None)
        if share is not None and share >= 0.25:
            lines.append(f"About {round(share * 100)}% of those mistakes came with very little time on the clock. "
                         "This is a time-management issue — try to reach the middlegame with enough time "
                         "to think through the critical moments.")

    if s.get("openings_with_most_early_mistakes"):
        o = s["openings_with_most_early_mistakes"][0]
        lines.append(f"Most early mistakes came in **{o['opening']}** "
                     f"({_plural(o['mistakes'], 'mistake')} in {_plural(o['games'], 'game')}). "
                     "Learning the key ideas — not just the moves — of this opening will help you "
                     "avoid these recurring early errors.")

    if s.get("endgame_types_with_mistakes"):
        t = s["endgame_types_with_mistakes"][0]
        lines.append(f"Most endgame mistakes came in **{t['type']}s** ({t['mistakes']}). "
                     f"Drilling {t['type'].lower()} technique is the fastest way to convert "
                     "the winning positions you are already reaching.")

    if n < 10:
        lines.append("_Only a few games were analysed, so treat this pattern as tentative._")
    if s.get("note"):
        lines.append(s["note"])

    shown = [ex for ex in facts["examples"] if "error" not in ex["explanation"]]
    if shown:
        lines += ["", "## The moments behind it"]
    for i, ex in enumerate(shown, 1):
        m, e = ex["moment"], ex["explanation"]
        where = m["phase"] + (f", {m['endgame_type']}" if m.get("endgame_type") else "")
        drop = m.get("win_pct_before", 0) - m.get("win_pct_after", 0)
        severity = m.get("severity", "mistake")

        lines.append(f"**{i}. Move {m['move_number']} ({where})** — [open game]({m['url']})")
        lines.append(f"You played `{m['played']}`; Stockfish preferred `{m['best_move']}`. "
                     f"Your winning chances dropped from **{m['win_pct_before']}%** to **{m['win_pct_after']}%** "
                     f"({round(drop)}‑point swing).")

        # Coaching interpretation of the drop
        lines.append(_drop_advice(severity, drop, m["phase"]))

        # Tactical facts — skip the unhelpful positional-fallback string and replace it
        POSITIONAL_FALLBACK = "No simple tactic was detected"
        tactical = e.get("tactical_facts", [])
        concrete = [f for f in tactical if POSITIONAL_FALLBACK not in f]
        if concrete:
            for fact in concrete:
                lines.append(f"- {fact}")
        else:
            lines.append(_positional_advice(m["phase"]))

        if e.get("best_line"):
            lines.append(f"- **Best continuation:** `{e['best_line']}`")
        if e.get("line_after_your_move"):
            lines.append(f"- **After your move:** `{e['line_after_your_move']}` — notice how the position deteriorates.")
        lines.append("")

    if facts.get("practice"):
        p = facts["practice"]
        theme_name = p["theme"].replace("_", " ")
        lines += [
            "## Practise this week",
            f"The pattern across your mistakes points to **{theme_name}**. "
            f"Spending 10–15 minutes a day on puzzles in this theme will train your eye "
            "to spot these positions before they happen over the board.",
            f"Try these puzzles near your rating:",
        ]
        lines += [f"- {x['url']} (rating {x['rating']})" for x in p["puzzles"]]
    return "\n".join(lines).strip()


def _call_gemini(system: str, user: str, api_key: str, model: str) -> str:
    try:
        from google import genai
        from google.genai import types as gt
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model, contents=user,
            config=gt.GenerateContentConfig(system_instruction=system, max_output_tokens=5000))
        text = (resp.text or "").strip()
    except Exception as e:
        raise BackendError(f"Gemini request failed ({type(e).__name__}: {str(e)[:200]}).") from e
    if not text:
        raise BackendError("Gemini returned no text (the response was empty or blocked).")
    return text


def _call_ollama(system: str, user: str, model: str, url: str) -> str:
    try:
        r = requests.post(f"{url}/api/chat", timeout=900, json={
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.3, "num_ctx": 8192}})
    except requests.RequestException as e:
        raise BackendError(f"Could not reach Ollama at {url}. Is it running? ({type(e).__name__})") from e
    if r.status_code != 200:
        raise BackendError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
    text = (r.json().get("message", {}) or {}).get("content", "").strip()
    if not text:
        raise BackendError("Ollama returned an empty answer.")
    return text


def write_report(facts: dict, backend: str = "template", api_key: str = "", model: str = "",
                 ollama_url: str = "", retry: bool = False) -> ReportResult:
    if "error" in facts:
        return ReportResult(text=facts["error"], backend_used="none")
    if backend == "template":
        return ReportResult(template_report(facts), "template")

    sheet = json.dumps(compact_for_llm(facts), indent=1)
    user = f"FACT SHEET (JSON):\n{sheet}\n\nWrite the coaching report now."

    def call(prompt: str) -> str:
        if backend == "gemini":
            return _call_gemini(REPORT_PROMPT, prompt, api_key, model or Config().model)
        if backend == "ollama":
            return _call_ollama(REPORT_PROMPT, prompt, model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
                                ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434"))
        raise BackendError(f"Unknown backend '{backend}'.")

    try:
        text = call(user)
        bad = unverified_moves(text, sheet)
        if bad and retry:
            text = call(user + f"\n\nYour previous draft mentioned moves that are not in the fact sheet: "
                               f"{', '.join(bad)}. Rewrite the report using only moves from the fact sheet.")
            bad = unverified_moves(text, sheet)
        return ReportResult(text, backend, verified=not bad, unverified=bad)
    except BackendError as e:
        return ReportResult(template_report(facts), "template", note=f"{e} Showing the Stockfish-only report instead.")