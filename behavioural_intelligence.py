#!/usr/bin/env python3
"""
Safeguard Patrol — Behavioural Intelligence v2

READ-ONLY analysis of the existing patrol_events.db.

This version focuses on:
- genuine behavioural episodes
- persistent hotspots
- risk based on frequency + persistence + peak risk
- a concise patrol intelligence summary
- optional JSON output

It does NOT modify detector.py or patrol_events.db.
"""

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


EPISODE_GAP_SECONDS = 20.0
SAME_AREA_DISTANCE_PX = 120.0
HOTSPOT_CELL_SIZE_PX = 75.0


def parse_bbox(value):
    try:
        parts = json.loads(value)
        if len(parts) != 4:
            return None
        return tuple(map(float, parts))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def bbox_center(value):
    bbox = parse_bbox(value)
    if bbox is None:
        return None
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def load_events(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return con.execute("""
            SELECT id, timestamp, class_name, confidence, bbox,
                   operator_action, operator_response_timestamp
            FROM detection_events
            ORDER BY timestamp ASC
        """).fetchall()
    finally:
        con.close()


def build_episodes(loitering):
    """Group nearby loitering events into behavioural episodes."""
    if not loitering:
        return []

    episodes = []
    current = [loitering[0]]

    for row in loitering[1:]:
        previous = current[-1]
        time_gap = float(row["timestamp"]) - float(previous["timestamp"])

        a = bbox_center(previous["bbox"])
        b = bbox_center(row["bbox"])
        distance = (
            math.hypot(a[0] - b[0], a[1] - b[1])
            if a and b else float("inf")
        )

        if time_gap <= EPISODE_GAP_SECONDS and distance <= SAME_AREA_DISTANCE_PX:
            current.append(row)
        else:
            episodes.append(current)
            current = [row]

    episodes.append(current)
    return episodes


def summarise_episode(events):
    timestamps = [float(e["timestamp"]) for e in events]
    risks = [int(e["confidence"]) for e in events if e["confidence"] is not None]

    centres = [bbox_center(e["bbox"]) for e in events]
    centres = [p for p in centres if p]

    if centres:
        cx = sum(p[0] for p in centres) / len(centres)
        cy = sum(p[1] for p in centres) / len(centres)
        centre = [round(cx, 1), round(cy, 1)]
    else:
        centre = None

    peak = max(risks) if risks else 0
    average = round(sum(risks) / len(risks), 1) if risks else 0
    duration = max(timestamps) - min(timestamps)

    # Episode severity deliberately rewards persistence as well as risk.
    persistence_points = min(20, max(0, len(events) - 1) * 4)
    severity = min(100, peak + persistence_points)

    if severity >= 80:
        level = "HIGH"
    elif severity >= 60:
        level = "ELEVATED"
    elif severity >= 45:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "events": len(events),
        "duration_s": round(duration, 1),
        "average_risk": average,
        "peak_risk": peak,
        "severity": severity,
        "level": level,
        "approx_center_px": centre,
    }


def build_hotspots(loitering):
    cells = defaultdict(list)

    for row in loitering:
        centre = bbox_center(row["bbox"])
        if not centre:
            continue

        cell = (
            int(centre[0] // HOTSPOT_CELL_SIZE_PX),
            int(centre[1] // HOTSPOT_CELL_SIZE_PX),
        )
        cells[cell].append(row)

    hotspots = []

    for cell, events in cells.items():
        risks = [
            int(e["confidence"])
            for e in events
            if e["confidence"] is not None
        ]

        centres = [bbox_center(e["bbox"]) for e in events]
        centres = [p for p in centres if p]

        if not centres:
            continue

        avg_x = sum(p[0] for p in centres) / len(centres)
        avg_y = sum(p[1] for p in centres) / len(centres)

        peak = max(risks) if risks else 0
        average = sum(risks) / len(risks) if risks else 0

        # Frequency is the dominant signal; repeated risk increases severity.
        frequency_points = min(55, len(events) * 2)
        risk_points = min(30, max(0, peak - 40))
        persistence_points = min(15, max(0, len(events) // 10))
        hotspot_score = min(
            100,
            frequency_points + risk_points + persistence_points
        )

        if hotspot_score >= 75:
            level = "HIGH"
        elif hotspot_score >= 50:
            level = "ELEVATED"
        elif hotspot_score >= 30:
            level = "MODERATE"
        else:
            level = "LOW"

        hotspots.append({
            "events": len(events),
            "average_risk": round(average, 1),
            "peak_risk": peak,
            "score": hotspot_score,
            "level": level,
            "approx_center_px": [round(avg_x, 1), round(avg_y, 1)],
        })

    hotspots.sort(
        key=lambda h: (h["score"], h["events"], h["peak_risk"]),
        reverse=True,
    )
    return hotspots[:10]


def make_summary(behavioural, hotspots, episodes):
    loitering = behavioural["loitering_events"]

    if not loitering:
        return (
            "No loitering activity was recorded during this patrol. "
            "No recurring behavioural hotspot was identified."
        )

    top = hotspots[0] if hotspots else None
    high_episodes = sum(1 for e in episodes if e["level"] == "HIGH")
    elevated_episodes = sum(1 for e in episodes if e["level"] == "ELEVATED")

    if top:
        location_text = (
            f"The strongest recurring hotspot contains {top['events']} "
            f"loitering events near approximately "
            f"({top['approx_center_px'][0]}, {top['approx_center_px'][1]})."
        )
    else:
        location_text = "No reliable spatial hotspot could be established."

    if high_episodes:
        episode_text = f"{high_episodes} high-severity behavioural episodes were identified."
    elif elevated_episodes:
        episode_text = (
            f"{elevated_episodes} elevated behavioural episodes were identified, "
            "with no high-severity episode."
        )
    else:
        episode_text = "Most behavioural activity remained below elevated severity."

    return (
        f"{loitering} loitering events were recorded across "
        f"{behavioural['episodes']} behavioural episodes. "
        f"The peak detector risk was {behavioural['maximum_risk']}, "
        f"with {behavioural['high_risk_events']} high-risk detector events. "
        f"{location_text} {episode_text}"
    )


def build_report(rows):
    counts = Counter()
    loitering = []

    for row in rows:
        name = (row["class_name"] or "unknown").lower()
        counts[name] += 1
        if name == "loitering":
            loitering.append(row)

    risks = [
        int(r["confidence"])
        for r in loitering
        if r["confidence"] is not None
    ]

    high_risk = sum(1 for r in risks if r >= 65)

    raw_episodes = build_episodes(loitering)
    episodes = [summarise_episode(e) for e in raw_episodes]

    hotspots = build_hotspots(loitering)

    average_risk = sum(risks) / len(risks) if risks else 0

    # Site status is intentionally conservative.
    # Repeated activity is considered more important than a single peak.
    if not loitering:
        site_status = "NORMAL"
    elif high_risk >= 20 or (
        hotspots and hotspots[0]["level"] == "HIGH"
        and average_risk >= 50
    ):
        site_status = "HIGH"
    elif high_risk >= 5 or average_risk >= 50:
        site_status = "ELEVATED"
    else:
        site_status = "MODERATE"

    episodes.sort(
        key=lambda e: (e["severity"], e["duration_s"], e["events"]),
        reverse=True,
    )

    behavioural = {
        "loitering_events": len(loitering),
        "average_risk": round(average_risk, 1),
        "maximum_risk": max(risks) if risks else 0,
        "high_risk_events": high_risk,
        "episodes": len(episodes),
        "site_status": site_status,
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_events": len(rows),
        "class_counts": dict(counts),
        "behavioural": behavioural,
        "intelligence_summary": make_summary(
            behavioural, hotspots, episodes
        ),
        "hotspots": hotspots,
        "top_episodes": episodes[:10],
    }

    return report


def print_report(report):
    b = report["behavioural"]

    print("\n" + "=" * 64)
    print(" SAFEGUARD — BEHAVIOURAL INTELLIGENCE v2")
    print("=" * 64)

    print(f"Site assessment    : {b['site_status']}")
    print(f"Total detections   : {report['total_events']}")
    print(f"Loitering events   : {b['loitering_events']}")
    print(f"Behavioural avg    : {b['average_risk']}")
    print(f"Peak detector risk : {b['maximum_risk']}")
    print(f"High-risk events   : {b['high_risk_events']}")
    print(f"Behavioural        : {b['episodes']} episodes")

    print("\nINTELLIGENCE SUMMARY")
    print("-" * 64)
    print(report["intelligence_summary"])

    print("\nDETECTION MIX")
    print("-" * 64)
    for name, count in sorted(
        report["class_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:<16} {count}")

    print("\nRECURRING BEHAVIOURAL HOTSPOTS")
    print("-" * 64)
    if report["hotspots"]:
        for i, h in enumerate(report["hotspots"], 1):
            x, y = h["approx_center_px"]
            print(
                f"#{i:<2} {h['level']:<9} "
                f"score={h['score']:<3} "
                f"events={h['events']:<4} "
                f"avg={h['average_risk']:<5} "
                f"peak={h['peak_risk']:<3} "
                f"near=({x}, {y})"
            )
    else:
        print("No recurring hotspot identified.")

    print("\nTOP BEHAVIOURAL EPISODES")
    print("-" * 64)
    if report["top_episodes"]:
        for i, e in enumerate(report["top_episodes"], 1):
            x, y = e["approx_center_px"] or ("?", "?")
            print(
                f"#{i:<2} {e['level']:<9} "
                f"severity={e['severity']:<3} "
                f"duration={e['duration_s']:<5}s "
                f"events={e['events']:<3} "
                f"avg={e['average_risk']:<5} "
                f"peak={e['peak_risk']:<3} "
                f"near=({x}, {y})"
            )
    else:
        print("No behavioural episodes identified.")

    print("=" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate behavioural intelligence from patrol_events.db"
    )
    parser.add_argument(
        "--db",
        default="patrol_events.db",
        help="Path to patrol_events.db",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    db_path = Path(args.db)

    if not db_path.exists():
        raise SystemExit(
            f"Database not found: {db_path}\n"
            "Run detector.py first or use --db with the correct path."
        )

    try:
        rows = load_events(db_path)
    except sqlite3.Error as exc:
        raise SystemExit(f"Could not read database: {exc}")

    report = build_report(rows)
    print_report(report)

    if args.json_path:
        output = Path(args.json_path)
        output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"[SGT] Intelligence report written to: {output}")


if __name__ == "__main__":
    main()
