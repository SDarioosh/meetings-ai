#!/usr/bin/env bash
# Generate a short synthetic 3-person meeting with macOS text-to-speech, for demos/smoke tests.
# Usage: scripts/make-demo-meeting.sh [output.m4a]   (macOS only; needs ffmpeg)
set -euo pipefail
out="${1:-demo-meeting.m4a}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

lines=(
  "Samantha|Okay, let's get started. This is the weekly sync for the Falcon dashboard launch. Priya, can you give us an update on exports?"
  "Daniel|Sure. CSV export is finished and merged. PDF export is about seventy percent done. I'll have it finished by next Friday, the twenty fourth."
  "Karen|One thing we never settled is whether PDF export ships in version one, or waits for version two."
  "Samantha|Let's decide that now. Two customers asked for it, so PDF export ships in version one."
  "Karen|Fine with me. But somebody needs to update the pricing page before launch, because PDF is a paid feature."
  "Daniel|The bigger risk is the reporting API rate limit. If we hit it during the customer demo, the charts just won't load."
  "Samantha|Marco, can you add caching in front of the reporting API? Done means every chart loads in under two seconds."
  "Karen|Yes, I'll take the caching work. I can have a first version by Wednesday."
  "Samantha|Great. Do we have a launch date yet? Marketing still hasn't confirmed."
  "Daniel|No, that's still open. Let's revisit it next week once marketing gets back to us."
  "Samantha|Okay. Thanks everyone, that's it for today."
)

i=0
: > "$tmp/list.txt"
for entry in "${lines[@]}"; do
  voice="${entry%%|*}"; text="${entry#*|}"
  say -v "$voice" -o "$tmp/$i.aiff" "$text"
  ffmpeg -loglevel error -y -i "$tmp/$i.aiff" -af "apad=pad_dur=0.6" -ar 16000 -ac 1 "$tmp/$i.wav"
  echo "file '$tmp/$i.wav'" >> "$tmp/list.txt"
  i=$((i + 1))
done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$tmp/list.txt" -c:a aac -b:a 64k "$out"
echo "Wrote $out"
