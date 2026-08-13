#!/usr/bin/env python3
"""
Safeguard Behavioural Intelligence API

Run this beside detector.py:

    python3 intelligence_server.py

It exposes the read-only behavioural report at:
    http://localhost:8766/intelligence

It reads patrol_events.db and does not modify detector.py or the database.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from behavioural_intelligence import build_report, load_events


class IntelligenceHandler(BaseHTTPRequestHandler):
    db_path = Path("patrol_events.db")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] != "/intelligence":
            self._send_json({"error": "Use /intelligence"}, 404)
            return

        try:
            rows = load_events(self.db_path)
            report = build_report(rows)
            self._send_json(report)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def log_message(self, fmt, *args):
        print(f"[SGT-INTEL] {fmt % args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="patrol_events.db")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"Database not found: {db}")

    IntelligenceHandler.db_path = db

    server = ThreadingHTTPServer(
        (args.host, args.port),
        IntelligenceHandler,
    )

    print(f"[SGT-INTEL] Intelligence API running on http://{args.host}:{args.port}")
    print(f"[SGT-INTEL] Reading: {db}")
    print("[SGT-INTEL] Read-only mode — detector.py is untouched.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SGT-INTEL] Server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
