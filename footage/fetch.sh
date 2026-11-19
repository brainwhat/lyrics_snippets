#!/bin/bash
# fetch.sh id... : download the 1080p Pexels files listed in candidates.tsv, resuming until complete
cd "$(dirname "$0")"
for id in "$@"; do
  url=$(awk -F'\t' -v i=$id '$1==i{print $3}' candidates.tsv)
  out=stock/pexels_$id.mp4
  want=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
  for try in $(seq 1 30); do
    have=$(stat -f %z "$out" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && [ "$want" -gt 0 ] && break
    curl -s -L -C - -o "$out" "$url"
  done
  echo "$id $(stat -f %z "$out") / $want"
done
