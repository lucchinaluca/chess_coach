"""Pipeline tests. Needs Stockfish (STOCKFISH_PATH); no network, no API key.  python tests/test_pipeline.py"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_offline as base                                   # noqa: E402
from coach import pipeline, tools as tools_module            # noqa: E402


def _facts():
    pgn = base.make_pgn()
    tools_module.fetch_games_pgn = lambda *a, **k: pgn
    c = base.cfg()
    t = tools_module.CoachTools(c, username="tester")
    try:
        return pipeline.gather_facts(t, "tester", "all", 5)
    finally:
        t.close()


def _server(replies):
    """Fake Ollama: returns the scripted replies in order; records requests."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            text = replies[min(len(seen) - 1, len(replies) - 1)]
            out = json.dumps({"message": {"role": "assistant", "content": text}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", seen


def test_facts_and_template():
    facts = _facts()
    assert facts["scan"]["games_analyzed"] == 1 and len(facts["examples"]) == 1
    assert facts["examples"][0]["moment"]["played"] == "Qh4"
    r = pipeline.write_report(facts, "template")
    assert r.backend_used == "template" and "Qh4" in r.text and "queen" in r.text
    print(r.text)


def test_error_passthrough():
    r = pipeline.write_report({"error": "No analysable standard games found."}, "gemini", api_key="x")
    assert r.text.startswith("No analysable") and r.backend_used == "none"


def test_ollama_backend_one_call_and_verification():
    facts = _facts()
    srv, url, seen = _server(["You played Qh4, which hangs the queen. Nf6 was better. My read: rushed."])
    r = pipeline.write_report(facts, "ollama", model="fake", ollama_url=url)
    srv.shutdown()
    assert len(seen) == 1 and r.backend_used == "ollama" and r.verified, (r.unverified, r.note)
    assert "FACT SHEET" in seen[0]["messages"][1]["content"] and seen[0]["stream"] is False

    srv, url, seen = _server(["Bxf7+ would have won.", "You played Qh4. Nf6 was better."])
    r = pipeline.write_report(facts, "ollama", model="fake", ollama_url=url)           # no retry
    assert len(seen) == 1 and not r.verified and r.unverified == ["Bxf7+"]
    r = pipeline.write_report(facts, "ollama", model="fake", ollama_url=url, retry=True)  # retry fixes it
    srv.shutdown()
    assert r.verified and "Bxf7" not in r.text


def test_backend_failure_falls_back_to_template():
    facts = _facts()
    r = pipeline.write_report(facts, "ollama", model="x", ollama_url="http://127.0.0.1:9")   # nothing listening
    assert r.backend_used == "template" and "Ollama" in r.note and "Qh4" in r.text

    def boom(*a, **k):
        raise pipeline.BackendError("Gemini request failed (ClientError: 429 quota).")
    pipeline._call_gemini = boom
    r = pipeline.write_report(facts, "gemini", api_key="k")
    assert r.backend_used == "template" and "429" in r.note


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
