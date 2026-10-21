"""The coach agent: an LLM that plans, calls tools, and writes the report.

Stockfish judges (inside the tools). The LLM only decides what to look at and
explains what the tools returned. A code-level check rejects reports that
mention moves no tool produced.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import Config
from .tools import TOOL_SCHEMAS, CoachTools

SYSTEM_PROMPT = """You are a chess coach for club-level players (roughly 800-1800 rating). You help one player understand their own mistakes, using their real games.

HOW YOU WORK
- You have four tools: scan_games, analyze_game, explain_position, get_practice. Stockfish,
  running inside the tools, is your only source of chess judgement.
- Never evaluate a position yourself. Do not call a move good, bad, winning or losing unless a
  tool result says so. Do not invent moves, variations or evaluations.
- Every move you mention must appear in a tool result from this conversation. Copy moves exactly
  as the tool gave them.
- If the tools do not say why a move was bad, state what changed (for example "your winning
  chances fell from 62% to 18%"), you can infer a reason and explanation just preface it saying it is the LLM's opinion . Name a tactic only if the tool's
  tactical_facts list it.

WORKFLOW
1. Start with scan_games for the player's username and the focus they asked for (all, opening or
   endgame). Default to their last 30 games.
2. From the summary, find the pattern: which phase, which kinds of positions, whether mistakes
   cluster when their clock is low, which openings or endgame types keep appearing.
3. Pick at most 3 moments that best illustrate the pattern (prefer the biggest drops) and call
   explain_position for each, passing fen_before and played_uci. Use analyze_game only if the
   player asks about one specific game.
4. For the practice suggestion, call get_practice once with a theme from explain_position's
   suggested_puzzle_themes and the player's average_rating. If no theme fits or the tool errors,
   suggest a general habit instead.
5. Use at most 8 tool calls in total. If a tool fails, say so briefly and continue.

REPORT FORMAT (under about 350 words)
- Pattern: 3-4 sentences with numbers from the scan.
- Examples: up to 3. For each give the game link, the move number, the move played, the better
  move, the change in winning chances, and a 2-3 sentences explanation based on tool facts, make a quick self understanding of the position.
- One thing to practise this week: specific and small, with puzzle links if you have them or more major things like phases of the game for a certain opening the player plays.
Use plain language and explain any chess term in a few words. Be direct and encouraging, not
flattering. If fewer than 10 games were analysed, say the pattern is tentative.

LIMITS
- Only discuss analysis of this player's games. For anything else, say what you can help with.
- If the username is missing, ask for it. Do not ask for anything a tool can provide.
- If the scan finds no big mistakes in the requested focus, say so and report the average loss
  instead of inventing problems, you can give general tips and advice commonly used.
"""

# SAN-looking tokens. Two-character tokens (bare pawn squares like e4, d5) are excluded
# because they are indistinguishable from square references in ordinary sentences.
# Piece moves (Na5, Qd7, Bxe5, etc.) ARE checked even when short — those are unambiguous.
_SAN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"O-O-O|O-O"                                        # castling
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]"               # piece moves: Na5, Nxe5, Nce4, N3e4
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?"                   # pawn captures: exd5, exd8=Q
    r"|[a-h][18]=[QRBN]"                                # pawn promotions: e8=Q
    r")"
    r"[+#]?(?![A-Za-z0-9])"
)


def unverified_moves(report: str, tool_text: str) -> list[str]:
    """Return moves written in the report that never appeared in a tool result.

    Uses whole-word matching so 'Na5' is not falsely cleared by 'Na50' appearing
    in a game ID or URL inside the tool JSON.
    """
    bad = []
    for m in {x.group(0) for x in _SAN.finditer(report)}:
        core = m.rstrip("+#")
        # Require the move to appear as a standalone token in the tool output,
        # not just as a substring of a longer alphanumeric string.
        pattern = r"(?<![A-Za-z0-9])" + re.escape(core) + r"(?![A-Za-z0-9])"
        if not re.search(pattern, tool_text):
            bad.append(m)
    return sorted(bad)


def _build_gemini_tools(tools: list[dict]) -> list:
    """Convert Anthropic-style input_schema tool defs to google-genai Tool objects."""
    from google.genai import types as gtypes

    declarations = []
    for t in tools:
        schema = t["input_schema"]
        props = {}
        for name, spec in schema.get("properties", {}).items():
            typ = spec.get("type", "string")
            if typ == "integer":
                gtype = gtypes.Type.INTEGER
            elif typ == "number":
                gtype = gtypes.Type.NUMBER
            elif typ == "boolean":
                gtype = gtypes.Type.BOOLEAN
            else:
                gtype = gtypes.Type.STRING
            kwargs = {"type": gtype, "description": spec.get("description", "")}
            if "enum" in spec:
                kwargs["enum"] = spec["enum"]
            props[name] = gtypes.Schema(**kwargs)

        declarations.append(gtypes.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties=props,
                required=schema.get("required", []),
            ),
        ))
    return [gtypes.Tool(function_declarations=declarations)]


@dataclass
class CoachResult:
    text: str
    tool_log: list = field(default_factory=list)        # [(name, args, result_dict), ...]
    verified: bool = True
    unverified: list = field(default_factory=list)
    tool_calls: int = 0


def run_coach(tools: CoachTools, request: str, api_key: str, cfg: Optional[Config] = None,
              on_event: Optional[Callable[[str], None]] = None) -> CoachResult:
    from google import genai                          # imported here so the rest works without it
    from google.genai import types as gtypes

    cfg = cfg or tools.cfg
    client = genai.Client(api_key=api_key)
    gemini_tools = _build_gemini_tools(TOOL_SCHEMAS)

    history = []
    log, tool_text, calls, retried = [], [], 0, False

    def say(msg):
        if on_event:
            on_event(msg)

    def send(user_parts) -> object:
        history.append(gtypes.Content(role="user", parts=_to_parts(user_parts)))
        resp = client.models.generate_content(
            model=cfg.model,
            contents=history,
            config=gtypes.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=gemini_tools,
                max_output_tokens=4000,
            ),
        )
        history.append(resp.candidates[0].content)
        return resp

    def _to_parts(payload) -> list:
        if isinstance(payload, str):
            return [gtypes.Part(text=payload)]
        if isinstance(payload, list):
            return payload
        return [payload]

    response = send(request)

    for _ in range(cfg.max_tool_calls + 6):
        # Collect function calls from the response
        fn_calls = [p.function_call for p in response.parts if p.function_call]

        if fn_calls:
            fn_response_parts = []
            for fn_call in fn_calls:
                name = fn_call.name
                args = dict(fn_call.args)
                if calls >= cfg.max_tool_calls:
                    out = {"error": "Tool call limit reached. Write the report now using what you have."}
                else:
                    calls += 1
                    say(f"Running {name}...")
                    out = tools.dispatch(name, args)
                    log.append((name, args, out))
                payload = json.dumps(out)
                tool_text.append(payload)
                fn_response_parts.append(
                    gtypes.Part(
                        function_response=gtypes.FunctionResponse(
                            name=name,
                            response={"content": payload[:20000]},
                        )
                    )
                )
            response = send(fn_response_parts)
            continue

        # No function calls — final text response
        text = response.text.strip() if response.text else ""
        bad = unverified_moves(text, "\n".join(tool_text))
        if bad and not retried:
            retried = True
            say("Checking the report against the tool results...")
            response = send(
                f"Your report mentions moves that no tool returned: {', '.join(bad)}. "
                "Rewrite it using only moves that appear in the tool results."
            )
            continue
        return CoachResult(text=text, tool_log=log, verified=not bad, unverified=bad,
                           tool_calls=calls)

    return CoachResult(text="The agent did not finish within the step limit.", tool_log=log,
                       verified=False, tool_calls=calls)
