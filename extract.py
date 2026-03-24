"""Extract transcripts from YouTube videos using Innertube API (no API key needed)."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

INNERTUBE_API_URL = "https://www.youtube.com/youtubei/v1/browse"
INNERTUBE_CLIENT = {
    "clientName": "WEB",
    "clientVersion": "2.20240101.00.00",
    "hl": "en",
    "gl": "US",
}


def get_channel_videos(channel_url: str, max_videos: int = 100) -> list[dict]:
    """Get video IDs from a YouTube channel."""
    # Extract channel ID or handle
    resp = requests.get(channel_url, headers={"User-Agent": "Mozilla/5.0"})

    # Find channel ID in page source
    match = re.search(r'"channelId":"(UC[^"]+)"', resp.text)
    if not match:
        raise ValueError(f"Could not find channel ID for {channel_url}")

    channel_id = match.group(1)

    # Use Innertube to browse channel videos
    payload = {
        "context": {"client": INNERTUBE_CLIENT},
        "browseId": channel_id,
        "params": "EgZ2aWRlb3PyBgQKAjoA",  # Videos tab
    }

    resp = requests.post(INNERTUBE_API_URL, json=payload)
    data = resp.json()

    videos = []
    _extract_videos_from_response(data, videos, max_videos)

    print(f"Found {len(videos)} videos from channel")
    return videos[:max_videos]


def _extract_videos_from_response(data: dict, videos: list, max_count: int):
    """Recursively extract video IDs from Innertube response."""
    if isinstance(data, dict):
        if "videoId" in data and len(videos) < max_count:
            vid = {
                "video_id": data["videoId"],
                "title": data.get("title", {}).get("runs", [{}])[0].get("text", ""),
            }
            if vid not in videos:
                videos.append(vid)
        for value in data.values():
            _extract_videos_from_response(value, videos, max_count)
    elif isinstance(data, list):
        for item in data:
            _extract_videos_from_response(item, videos, max_count)


def get_transcript(video_id: str, languages: list[str] = None) -> dict | None:
    """Extract transcript from a YouTube video using Innertube API."""
    languages = languages or ["en"]

    # Step 1: Get video page to find transcript params
    url = f"https://www.youtube.com/watch?v={video_id}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # Find serialized player response
    match = re.search(r"var ytInitialPlayerResponse\s*=\s*({.+?});", resp.text)
    if not match:
        return None

    try:
        player = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    # Check for captions
    captions = player.get("captions", {}).get("playerCaptionsTracklistRenderer", {})
    tracks = captions.get("captionTracks", [])

    if not tracks:
        return None

    # Find preferred language track
    track_url = None
    for lang in languages:
        for track in tracks:
            if track.get("languageCode", "").startswith(lang):
                track_url = track["baseUrl"]
                break
        if track_url:
            break

    if not track_url:
        track_url = tracks[0]["baseUrl"]  # Fallback to first available

    # Fetch transcript XML
    track_url += "&fmt=json3"
    resp = requests.get(track_url)

    try:
        transcript_data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None

    # Parse events into text segments
    segments = []
    for event in transcript_data.get("events", []):
        segs = event.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if text and text != "\n":
            segments.append({
                "text": text,
                "start": event.get("tStartMs", 0) / 1000,
                "duration": event.get("dDurationMs", 0) / 1000,
            })

    title = player.get("videoDetails", {}).get("title", "")

    return {
        "video_id": video_id,
        "title": title,
        "segments": segments,
        "full_text": " ".join(s["text"] for s in segments),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract YouTube transcripts")
    parser.add_argument("--channel", help="YouTube channel URL")
    parser.add_argument("--video", help="Single video URL or ID")
    parser.add_argument("--playlist", help="Playlist URL")
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--output", default="transcripts")
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        # Single video
        video_id = args.video.split("v=")[-1].split("&")[0] if "youtube.com" in args.video else args.video
        print(f"Extracting transcript for {video_id}...")
        result = get_transcript(video_id, args.languages)
        if result:
            out_file = output_dir / f"{video_id}.json"
            out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"Saved: {out_file} ({len(result['segments'])} segments)")
        else:
            print("No transcript available for this video")

    elif args.channel:
        # Full channel
        videos = get_channel_videos(args.channel, args.max_videos)

        success = 0
        for video in tqdm(videos, desc="Extracting transcripts"):
            result = get_transcript(video["video_id"], args.languages)
            if result:
                out_file = output_dir / f"{video['video_id']}.json"
                out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                success += 1
            time.sleep(args.delay)

        print(f"\nExtracted {success}/{len(videos)} transcripts to {output_dir}/")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
