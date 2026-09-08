import asyncio
import base64
import json
import os
import time

import websockets
from google import genai
from google.genai import types

DETECTOR_WS = "ws://localhost:8765"
BRIDGE_WS_HOST = "0.0.0.0"
BRIDGE_WS_PORT = 8767
MODEL = "gemini-3.1-flash-live-preview"
GEMINI_INTERVAL_S = 1.0

SYSTEM_INSTRUCTION = """
You are GEMINI, the visual intelligence layer of SafeGuardOS.

You receive aerial security-drone imagery from ARGUS.
Your job is to assess ONLY what is visually observable.
SafeGuardOS already has YOLO/ByteTrack object detections and behavioural
tracking. You are an additional semantic scene-understanding layer.

For every requested assessment, output EXACTLY one JSON object:
{
  "scene_status": "NORMAL",
  "activity_level": "LOW",
  "people_count": 0,
  "vehicle_count": 0,
  "interaction": "NONE",
  "concern": "NONE",
  "risk_level": "LOW",
  "confidence": 0.0,
  "summary": "Short description of what is happening."
}

Allowed values:
scene_status: NORMAL, WATCH, CONCERNING
activity_level: LOW, MEDIUM, HIGH
interaction: NONE, POSSIBLE_FOLLOWING, POSSIBLE_PURSUIT, POSSIBLE_CONFRONTATION, GROUP_ACTIVITY, UNKNOWN
concern: NONE, UNUSUAL_MOVEMENT, POSSIBLE_FOLLOWING, POSSIBLE_PURSUIT, POSSIBLE_CONFRONTATION, UNKNOWN
risk_level: LOW, MEDIUM, HIGH

Rules:
1. Return ONLY the JSON object.
2. Do not use Markdown.
3. Do not identify people.
4. Do not claim that someone committed a crime.
5. Do not infer intent without visual evidence.
6. If uncertain, use UNKNOWN and reduce confidence.
7. Confidence must be a number between 0.0 and 1.0.
8. Keep summary under 30 words.
9. Treat your result as an observation, not a final operational decision.
"""


class GeminiBridge:
    def __init__(self):
        self.clients = set()
        self.last_gemini_time = 0.0
        self.latest_intelligence = {
            "scene_status": "NORMAL",
            "activity_level": "LOW",
            "people_count": 0,
            "vehicle_count": 0,
            "interaction": "NONE",
            "concern": "NONE",
            "risk_level": "LOW",
            "confidence": 0.0,
            "summary": "Gemini is starting.",
            "timestamp": time.time(),
        }

    async def broadcast(self, payload):
        if not self.clients:
            return
        message = json.dumps(payload)
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        self.clients -= disconnected

    async def flutter_client(self, websocket):
        print(f"[GEMINI] Flutter client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            await websocket.send(json.dumps({
                "type": "gemini_intelligence",
                "intelligence": self.latest_intelligence,
            }))
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[GEMINI] Flutter client disconnected: {websocket.remote_address}")

    def parse_response(self, text):
        cleaned = text.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

    def validate_intelligence(self, data):
        if not isinstance(data, dict):
            return None
        required = [
            "scene_status", "activity_level", "people_count", "vehicle_count",
            "interaction", "concern", "risk_level", "confidence", "summary",
        ]
        if not all(key in data for key in required):
            return None
        try:
            data["people_count"] = int(data["people_count"])
            data["vehicle_count"] = int(data["vehicle_count"])
            data["confidence"] = float(data["confidence"])
        except (TypeError, ValueError):
            return None
        data["confidence"] = max(0.0, min(1.0, data["confidence"]))
        data["summary"] = str(data["summary"])[:300]
        return data

    async def ask_gemini(self, session, frame_bytes, detector_detections):
        await session.send_realtime_input(
            video=types.Blob(data=frame_bytes, mime_type="image/jpeg")
        )

        prompt = f"""
Assess the current ARGUS frame now.

Use the image as the primary visual evidence.
The following is supporting context from SafeGuardOS YOLO/ByteTrack:
{json.dumps({"detector_detections": detector_detections})}

Return exactly one JSON object following the required SafeGuardOS format.
Do not provide any other text.
"""
        await session.send_realtime_input(text=prompt)

        full_response = ""
        try:
            async for response in session.receive():
                if not response.server_content:
                    continue
                transcription = response.server_content.output_transcription
                if transcription and transcription.text:
                    full_response += transcription.text
                if response.server_content.turn_complete:
                    break
        except Exception as exc:
            print(f"[GEMINI] Response error: {exc}")
            return None

        parsed = self.parse_response(full_response)
        if parsed is None:
            print("[GEMINI] Warning: response was not valid JSON.")
            print("[GEMINI] Raw:", full_response.strip())
            return None
        return self.validate_intelligence(parsed)

    async def run_detector_consumer(self):
        while True:
            try:
                print(f"[GEMINI] Connecting to detector: {DETECTOR_WS}")
                async with websockets.connect(DETECTOR_WS, max_size=20 * 1024 * 1024, ping_interval=None,) as detector_ws:
                    print("[GEMINI] Connected to detector WebSocket.")
                    client = genai.Client()
                    config = {
                        "response_modalities": ["AUDIO"],
                        "output_audio_transcription": {},
                        "system_instruction": SYSTEM_INSTRUCTION,
                    }
                    print(f"[GEMINI] Connecting to Live model: {MODEL}")
                    async with client.aio.live.connect(model=MODEL, config=config) as session:
                        print("[GEMINI] Live connected.")
                        print("[GEMINI] Bridge is running.")
                        async for raw_message in detector_ws:
                            try:
                                packet = json.loads(raw_message)
                            except json.JSONDecodeError:
                                continue
                            if packet.get("type") != "frame":
                                continue
                            frame_b64 = packet.get("frame")
                            if not frame_b64:
                                continue
                            now = time.time()
                            if now - self.last_gemini_time < GEMINI_INTERVAL_S:
                                continue
                            self.last_gemini_time = now
                            try:
                                frame_bytes = base64.b64decode(frame_b64)
                            except Exception:
                                print("[GEMINI] Could not decode detector frame.")
                                continue
                            detections = packet.get("detections", [])
                            print(f"[GEMINI] Assessing frame (detector detections: {len(detections)})...")
                            intelligence = await self.ask_gemini(session, frame_bytes, detections)
                            if intelligence is None:
                                continue
                            intelligence["timestamp"] = time.time()
                            self.latest_intelligence = intelligence
                            print(
                                "[GEMINI] "
                                f"{intelligence.get('scene_status')} | "
                                f"{intelligence.get('interaction')} | "
                                f"{intelligence.get('risk_level')} | "
                                f"{intelligence.get('summary')}"
                            )
                            await self.broadcast({
                                "type": "gemini_intelligence",
                                "intelligence": intelligence,
                            })
            except (ConnectionRefusedError, websockets.exceptions.ConnectionClosed, OSError) as exc:
                print(f"[GEMINI] Detector connection lost: {exc}")
            except Exception as exc:
                print(f"[GEMINI] Bridge error: {exc}")
            print("[GEMINI] Retrying detector connection in 3 seconds...")
            await asyncio.sleep(3)


async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set.")
        return

    bridge = GeminiBridge()
    await websockets.serve(
        bridge.flutter_client,
        BRIDGE_WS_HOST,
        BRIDGE_WS_PORT,
        max_size=20 * 1024 * 1024,
    )
    print("=" * 70)
    print("SafeGuardOS Gemini Bridge")
    print("=" * 70)
    print(f"Detector input : {DETECTOR_WS}")
    print(f"Flutter output : ws://localhost:{BRIDGE_WS_PORT}")
    print(f"Gemini model   : {MODEL}")
    print("Gemini rate    : ~1 frame/sec")
    print("=" * 70)
    await bridge.run_detector_consumer()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[GEMINI] Bridge stopped.")
