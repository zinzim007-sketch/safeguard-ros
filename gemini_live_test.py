import asyncio
import cv2
import json

from google import genai
from google.genai import types


VIDEO_PATH = "/mnt/c/Users/zinzi/Desktop/SGT work/sgt_patrol/assets/demo/patrol_demo.mp4"

MODEL = "gemini-3.1-flash-live-preview"


async def main():
    client = genai.Client()

    config = {
        "response_modalities": ["AUDIO"],
        "output_audio_transcription": {},
        "system_instruction": """
You are GEMINI, the visual intelligence layer of SafeGuardOS.

You receive aerial security-drone imagery.

Your job is to assess ONLY what is visually observable.

For every assessment, output EXACTLY one JSON object using this structure:

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

scene_status:
NORMAL
WATCH
CONCERNING

activity_level:
LOW
MEDIUM
HIGH

interaction:
NONE
POSSIBLE_FOLLOWING
POSSIBLE_PURSUIT
POSSIBLE_CONFRONTATION
GROUP_ACTIVITY
UNKNOWN

concern:
NONE
UNUSUAL_MOVEMENT
POSSIBLE_FOLLOWING
POSSIBLE_PURSUIT
POSSIBLE_CONFRONTATION
UNKNOWN

risk_level:
LOW
MEDIUM
HIGH

Rules:

1. Return ONLY the JSON object.
2. Do not use Markdown.
3. Do not add explanations before or after the JSON.
4. Do not identify people.
5. Do not claim that someone committed a crime.
6. Do not infer intent without visual evidence.
7. If uncertain, use UNKNOWN and reduce confidence.
8. Confidence must be a number between 0.0 and 1.0.
9. Keep summary under 30 words.
""",
    }

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("ERROR: Could not open video.")
        return

    print("Connecting to Gemini Live...")

    try:
        async with client.aio.live.connect(
            model=MODEL,
            config=config,
        ) as session:

            print("Gemini Live connected!")
            print("Streaming structured intelligence...")
            print("-" * 70)

            frame_number = 0

            while True:
                ok, frame = video.read()

                if not ok:
                    print("\nVideo finished.")
                    break

                frame_number += 1

                # Approximately one frame per second.
                if frame_number % 60 != 0:
                    continue

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )

                if not ok:
                    continue

                await session.send_realtime_input(
                    video=types.Blob(
                        data=encoded.tobytes(),
                        mime_type="image/jpeg",
                    )
                )

                print(f"\nFRAME {frame_number}")

                await session.send_realtime_input(
                    text="""
Assess the current frame.

Return exactly one JSON object following the required SafeGuardOS format.
Do not provide any other text.
"""
                )

                full_response = ""

                try:
                    async for response in session.receive():

                        if not response.server_content:
                            continue

                        transcription = (
                            response.server_content.output_transcription
                        )

                        if transcription and transcription.text:
                            full_response += transcription.text

                        if response.server_content.turn_complete:
                            break

                except Exception as e:
                    print(f"Response error: {e}")
                    continue

                cleaned = full_response.strip()

                print("🤖 GEMINI RAW:")
                print(cleaned)

                # Try to parse the response as JSON.
                try:
                    intelligence = json.loads(cleaned)

                    print("\n🧠 SAFEGUARDOS INTELLIGENCE:")
                    print(
                        json.dumps(
                            intelligence,
                            indent=2,
                        )
                    )

                except json.JSONDecodeError:
                    print("\n⚠️ Gemini response was not valid JSON.")

                print("-" * 70)

                await asyncio.sleep(0.5)

    except Exception as e:
        print("\nGemini Live connection error:")
        print(e)

    finally:
        video.release()


if __name__ == "__main__":
    asyncio.run(main())
