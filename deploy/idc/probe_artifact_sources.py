"""Bounded1MiB public-source probes on IDC. Never downloads a full model file."""
import json
import socket
import time
import urllib.request

NAME = 'model-00001-of-00004.safetensors'
SOURCES = [
    'https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/resolve/2c4565515e0f265c6511776e7193b22c0968ddc7/'+NAME,
    'https://modelscope.cn/models/Qwen/Qwen3-VL-Embedding-8B/resolve/master/'+NAME,
]

if __name__ == '__main__':
    if socket.gethostname() != 'dev-instance-shenrongtian':
        raise ValueError('IDC only')
    for source in SOURCES:
        started = time.monotonic()
        try:
            request = urllib.request.Request(source, headers={'Range': 'bytes=0-1048575'})
            with urllib.request.urlopen(request, timeout=15) as response:
                result = dict(source=source.split('/')[2], status=response.status,
                              content_range=response.headers.get('Content-Range'))
                result['bytes'] = len(response.read(1048576))
                result['seconds'] = time.monotonic()-started
            print(json.dumps(result), flush=True)
        except Exception as exc:
            print(json.dumps(dict(source=source.split('/')[2], error=type(exc).__name__,
                                  status=getattr(exc, 'code', None), seconds=time.monotonic()-started)), flush=True)
