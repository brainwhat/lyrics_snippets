"""Whisper word times for a vocal stem: python engine/transcribe.py <vocals.wav> <lang> <out.json>
Writes [{s, e, text, words:[{w, s, e, p}]}] like Конструктор снипеттов/transcribe.py."""
import json, sys, subprocess
import numpy as np
import mlx_whisper

SR = 16000
src, lang, out = sys.argv[1:4]
y = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', src, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                                 capture_output=True, check=True).stdout, np.float32)
res = mlx_whisper.transcribe(y, path_or_hf_repo='mlx-community/whisper-large-v3-turbo', language=lang,
                             word_timestamps=True, condition_on_previous_text=False,
                             compression_ratio_threshold=2.2, verbose=False)
segs = []
for s in res['segments']:
    ws = [dict(w=w['word'].strip(), s=round(w['start'], 3), e=round(w['end'], 3), p=round(w.get('probability', 1), 3))
          for w in s.get('words', []) if w['word'].strip()]
    if ws:
        segs.append(dict(s=ws[0]['s'], e=ws[-1]['e'], text=s['text'].strip(), words=ws))
json.dump(segs, open(out, 'w'), ensure_ascii=False, indent=1)
print(len(segs), 'segments')
