"""Pinned public Qwen artifacts: IDC-only plan/pilot, then serial verified transfer.

No credentials, retries, remote code execution, global proxy or local bulk traffic.
Partial files are retained on failure. Existing final files must match source hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

REPO = 'Qwen/Qwen3-VL-Embedding-8B'
REVISION = '2c4565515e0f265c6511776e7193b22c0968ddc7'
ENDPOINT = 'https://hf-mirror.com'
DATA = Path('/pfs/user/data/host_piper_paper')
OUTPUT = DATA/'weights'/f'qwen3_vl_embedding_8b_{REVISION[:12]}'
PLAN = OUTPUT/'transfer_plan.json'
ALLOWED = {'config.json', 'preprocessor_config.json', 'video_preprocessor_config.json',
           'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
           'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
           'model.safetensors.index.json', 'README.md'}


def validate_host():
    if socket.gethostname() != 'dev-instance-shenrongtian' or not DATA.is_dir():
        raise ValueError('This downloader is restricted to the verified IDC development host')
    if not OUTPUT.resolve().is_relative_to(DATA.resolve()):
        raise ValueError('Artifact output escaped the designated data root')


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=45)


def public_metadata():
    # This endpoint accepts the host's curl TLS client; urllib requests receive403.
    result = subprocess.run(['curl', '-fsSL', '--connect-timeout', '10', '--max-time', '30',
        f'{ENDPOINT}/api/models/{REPO}?blobs=true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(f'Public metadata transport failed: curl exit{result.returncode}')
    return json.loads(result.stdout)


def url(name, source='hf'):
    if source == 'modelscope':
        # Publication alias is mutable, but every final byte is pinned by the HF
        #revision's SHA256 below. A changed mirror artifact fails integrity checks.
        return f'https://modelscope.cn/models/{REPO}/resolve/master/{name}'
    return f'{ENDPOINT}/{REPO}/resolve/{REVISION}/{name}'


def digest(path, row):
    size = path.stat().st_size
    if size != row['size']:
        return False
    if row.get('sha256'):
        hasher, expected = hashlib.sha256(), row['sha256']
    else:
        hasher, expected = hashlib.sha1(), row['blob_id']
        hasher.update(f'blob {size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest() == expected


def plan():
    validate_host()
    if PLAN.exists():
        raise ValueError('Plan already exists; inspect it rather than overwrite')
    print(json.dumps(dict(stage='metadata', repository=REPO)), flush=True)
    metadata = public_metadata()
    if metadata['sha'] != REVISION or metadata.get('private') or metadata.get('gated'):
        raise ValueError('Unexpected revision or non-public model')
    files = []
    for entry in metadata['siblings']:
        name = entry['rfilename']
        if name in ALLOWED or (name.startswith('model-') and name.endswith('.safetensors') and '/' not in name):
            files.append(dict(name=name, size=entry['size'], blob_id=entry['blobId'],
                              sha256=entry.get('lfs', {}).get('sha256')))
    if sum(row['name'].endswith('.safetensors') for row in files) != 4:
        raise ValueError('Expected exactly four public checkpoint shards')
    total = sum(row['size'] for row in files)
    free = shutil.disk_usage(DATA).free
    if free < 2*total+5*1024**3:
        raise ValueError('Insufficient free space for transfer')
    shard = next(row for row in files if row['name'].endswith('.safetensors'))
    print(json.dumps(dict(stage='one_mib_range_pilot', file=shard['name'], total_bytes=total)), flush=True)
    start = time.monotonic()
    size = 1024*1024
    with request(url(shard['name']), {'Range': f'bytes=0-{size-1}'}) as response:
        status = response.status
        content_range = response.headers.get('Content-Range')
        if status != 206 or content_range != f"bytes 0-{size-1}/{shard['size']}":
            raise ValueError(f'Pilot range unsupported: HTTP {status}; no full transfer started')
        payload = response.read(size+1)
    if len(payload) != size:
        raise ValueError('Pilot size mismatch')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'pilot_first_mib.bin').open('xb') as handle:
        handle.write(payload)
    data = dict(repository=REPO, revision=REVISION, output=str(OUTPUT), source=ENDPOINT,
        total_bytes=total, disk_free_bytes=free, hostname=socket.gethostname(),
        proxy_set={name: bool(os.environ.get(name)) for name in
                   ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')},
        pilot=dict(bytes=size, seconds=time.monotonic()-start, status=status,
                   sha256=hashlib.sha256(payload).hexdigest(), content_range=content_range), files=files)
    with PLAN.open('x') as handle:
        json.dump(data, handle, indent=2)
    print(json.dumps({k:v for k,v in data.items() if k != 'files'}), flush=True)


def probe_modelscope():
    validate_host()
    data = json.loads(PLAN.read_text())
    shard = next(row for row in data['files'] if row['name'].endswith('.safetensors'))
    started = time.monotonic()
    with request(url(shard['name'], 'modelscope'), {'Range': 'bytes=0-1048575'}) as response:
        if response.status != 206 or response.headers.get('Content-Range') != f"bytes 0-1048575/{shard['size']}":
            raise ValueError('Alternate source range mismatch')
        payload = response.read(1048577)
    if len(payload) != 1048576 or hashlib.sha256(payload).hexdigest() != data['pilot']['sha256']:
        raise ValueError('Alternate source prefix does not match the pinned HF artifact')
    result = dict(source='modelscope.cn', repository=REPO, pinned_revision=REVISION,
        bytes=len(payload), prefix_sha256=hashlib.sha256(payload).hexdigest(),
        seconds=time.monotonic()-started, range_and_prefix_match=True)
    with (OUTPUT/'modelscope_pilot.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


def download(source='hf', resume=False):
    validate_host()
    data = json.loads(PLAN.read_text())
    if data['revision'] != REVISION or data['output'] != str(OUTPUT) or data['pilot']['status'] != 206:
        raise ValueError('Unverified transfer plan')
    if source == 'modelscope':
        probe = json.loads((OUTPUT/'modelscope_pilot.json').read_text())
        if not probe['range_and_prefix_match'] or probe['pinned_revision'] != REVISION:
            raise ValueError('Unverified alternate source')
    # Small configs/processor files arrive before large weights, allowing independent
    # CPU processing checks while this attached, single-worker transfer continues.
    files = sorted(data['files'], key=lambda row: (row['size'] > 32*1024**2, row['name']))
    started = time.monotonic()
    for row in files:
        dest = OUTPUT/row['name']
        if dest.exists():
            if not digest(dest, row):
                raise ValueError(f"Existing artifact failed integrity: {row['name']}")
            print(json.dumps(dict(file=row['name'], status='verified_existing')), flush=True)
            continue
        partial = dest.with_suffix(dest.suffix+'.partial')
        if partial.exists() and not resume:
            raise ValueError(f"Partial artifact needs explicit recovery review: {partial.name}")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > row['size']:
            raise ValueError('Partial artifact exceeds pinned source size')
        if offset == row['size']:
            if not digest(partial, row):
                raise ValueError('Complete partial failed integrity; retained for diagnosis')
            partial.rename(dest)
            continue
        count, last_report = offset, time.monotonic()
        print(json.dumps(dict(file=row['name'], status='start', expected_bytes=row['size'])), flush=True)
        if row['size'] <= 32*1024**2:
            if offset:
                raise ValueError('Small-file partial requires separate recovery; not overwritten')
            result = subprocess.run(['curl', '-fsSL', '--connect-timeout', '10', '--max-time', '120',
                '--max-filesize', str(row['size']), '--output', str(partial), url(row['name'], source)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode or not partial.exists() or not digest(partial, row):
                raise ValueError(f"Small artifact transfer/integrity failure: {row['name']}; curl exit{result.returncode}")
            partial.rename(dest)
            print(json.dumps(dict(file=row['name'], status='verified', bytes=row['size'])), flush=True)
            continue
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        with request(url(row['name'], source), headers) as response:
            if offset:
                expected_range = f"bytes {offset}-{row['size']-1}/{row['size']}"
                if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                    raise ValueError('Resume response does not match the requested byte range')
            elif response.status != 200:
                raise ValueError('Unexpected full-transfer HTTP status')
            #Response validated before appending to a partial artifact.
            handle = partial.open('ab' if offset else 'xb')
            try:
                while True:
                    chunk = response.read(4*1024*1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > row['size']:
                        raise ValueError('Transfer exceeded pinned source size')
                    handle.write(chunk)
                    if time.monotonic()-last_report >= 20:
                        print(json.dumps(dict(file=row['name'], received_bytes=count)), flush=True)
                        last_report = time.monotonic()
            finally:
                handle.close()
        if not digest(partial, row):
            raise ValueError(f"Downloaded artifact failed integrity: {row['name']}")
        partial.rename(dest)
        print(json.dumps(dict(file=row['name'], status='verified', bytes=count)), flush=True)
    summary = dict(repository=REPO, revision=REVISION, files=len(files), bytes=data['total_bytes'],
                   seconds=time.monotonic()-started, verified=True, final_transfer_source=source)
    with (OUTPUT/'verified_artifacts.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'download', 'probe-modelscope', 'download-modelscope'))
    parser.add_argument('--resume-partials', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'plan':
            plan()
        elif args.mode == 'probe-modelscope':
            probe_modelscope()
        else:
            download('modelscope' if args.mode == 'download-modelscope' else 'hf', args.resume_partials)
    except urllib.error.HTTPError as exc:
        # Do not print signed redirected URLs or response headers in error logs.
        print(json.dumps(dict(error='HTTPError', status=exc.code)), flush=True)
        raise SystemExit(2)
    except urllib.error.URLError:
        print(json.dumps(dict(error='URLError')), flush=True)
        raise SystemExit(2)
