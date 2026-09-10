"""Portable low-memory alignment helpers; no training at import."""
import os
from pathlib import Path


def apply_low_memory_config(root):
    from piper_profile import apply_profile, enable_strict_loading
    cfg = apply_profile(root, context_frames=1)
    enable_strict_loading()
    cfg.DS_CONFIG.scheduler.params.total_num_steps = 3000
    cfg.TRAIN.MAX_ITERS = 3000
    os.environ['HOST_ALIGNMENT_LOGITS_TO_KEEP'] = '1'
    os.environ['HOST_VIDEO_DECODE_MODE'] = 'sequential'
    return cfg


def sequential_load_imgs(paths):
    """Decode in presentation order with PyAV, the existing RGB-certificate backend.
    Never random-seek, silently replace, interpolate, or cache videos to disk.
    """
    import av
    from PIL import Image
    grouped, result = {}, [None]*len(paths)
    for position, path in enumerate(paths):
        if not isinstance(path, str) or not path.startswith('video://'):
            raise ValueError('Explicit video frame paths required')
        video, index, dimensions = path[8:].rsplit('::', 2)
        width, height = map(int, dimensions.split('x'))
        index = int(index)
        if index < 0 or min(width, height) <= 0:
            raise ValueError('Invalid frame index or dimensions')
        grouped.setdefault(video, {}).setdefault(index, []).append((position, width, height))
    for video, wanted in grouped.items():
        last = max(wanted)
        with av.open(video) as container:
            container.streams.video[0].thread_count = 2
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    rgb = Image.fromarray(frame.to_ndarray(format='rgb24'))
                    for position, width, height in wanted[index]:
                        result[position] = rgb.resize((width, height), Image.Resampling.BILINEAR)
                if index >= last:
                    break
    if any(frame is None for frame in result):
        raise ValueError('Missing/out-of-range decoded frame')
    return result


def check_paths(root, weights, output, minimum_free_gib=110):
    import json
    import shutil
    root, weights, output = map(lambda p:Path(p).resolve(), (root, weights, output))
    if output.exists():
        raise ValueError('Use a new output directory')
    if shutil.disk_usage(output.parent).free < minimum_free_gib*1024**3:
        raise ValueError(f'Reserve{minimum_free_gib}GiB free for checkpoint storage and margin')
    verified = json.loads((weights/'verified_artifacts.json').read_text())
    if verified.get('verified') is not True or verified.get('revision') != '2c4565515e0f265c6511776e7193b22c0968ddc7':
        raise ValueError('Pinned verified weights required')
    audit = json.loads((root/'sequential_decode_verified.json').read_text())
    if audit.get('complete') is not True or audit.get('videos') != 2652:
        raise ValueError('Full884-episode sequential decode audit required')
    return audit
