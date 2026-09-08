"""Create verified, lossless seek-safe video derivatives on IDC; never edit raw data.

Canonical CFR video PTS represents ROW INDEX, not measured sensor time. Every
decoded source frame is retained in order. Actual timestamps stay in parquet.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import av

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_temporal_decode import compare_seek


def decoded_digest(video):
    digest = hashlib.sha256()
    count = 0
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            pixels = frame.to_ndarray(format='rgb24')
            digest.update(str(pixels.shape).encode())
            digest.update(pixels.tobytes())
            count += 1
    return dict(frames=count, rgb_sha256=digest.hexdigest())


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def transcode(source, destination, gop=1):
    if gop not in (1, 16):
        raise ValueError('Only explicitly tested candidate GOP sizes1/16 are allowed')
    started = time.monotonic()
    source, destination = Path(source), Path(destination)
    if not source.is_file() or destination.exists():
        raise ValueError('Existing source and new destination required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
        '-threads', '2', '-i', str(source), '-map', '0:v:0', '-an',
        '-vf', 'setpts=N/(30*TB)', '-r', '30', '-vsync', '0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '0', '-g', str(gop), '-bf', '0',
        '-threads', '2', str(destination)]
    subprocess.run(command, check=True)
    original, derived = decoded_digest(source), decoded_digest(destination)
    if original != derived:
        raise ValueError(f'Decoded pixel/frame integrity failure: {destination}; retained for diagnosis')
    seek = compare_seek(destination)
    if not seek['exact_match']:
        raise ValueError(f'Derivative still has seek mismatches: {destination}')
    return dict(source=str(source), destination=str(destination), **derived,
        source_bytes=source.stat().st_size, cache_bytes=destination.stat().st_size,
        seek_exact_match=True, sampled_indices=seek['sampled_indices'],
        video_clock='CFR30 indexed by original row; actual sensor timestamps unchanged in parquet',
        codec=f'lossless libx264 GOP{gop} no B frames', command=command,
        source_file_sha256=file_digest(source), cache_file_sha256=file_digest(destination),
        elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--gop', type=int, choices=(1, 16), default=1)
    args = p.parse_args()
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or args.limit < 1:
        raise ValueError('New absolute output directory and positive limit required')
    report = json.loads(Path(args.audit).read_text())
    selected = sorted(report['decode'], key=lambda r: (r['exact_match'], r['source'], r['camera']))[:args.limit]
    output.mkdir(parents=True)
    with (output/'verified.jsonl').open('x') as handle:
        for row in selected:
            dest = output/row['source']/row['camera']/Path(row['video']).name
            result = transcode(row['video'], dest, args.gop)
            handle.write(json.dumps(result)+'\n')
            handle.flush()
            print(json.dumps(result), flush=True)
