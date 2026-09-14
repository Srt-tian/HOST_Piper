"""Strict presentation-order policy decoding, without disk caches or placeholders."""
from pathlib import Path


def load_frames_strict(paths, target_size):
    import av
    from PIL import Image

    width, height = target_size
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('Positive integer target dimensions required')
    result = [None] * len(paths)
    grouped = {}
    for position, path in enumerate(paths):
        if not isinstance(path, str):
            raise ValueError('String frame paths required')
        if not path.startswith('video://'):
            with Image.open(Path(path)) as source:
                result[position] = source.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
            continue
        parts = path[8:].split('::')
        if len(parts) not in (2, 3) or not parts[0]:
            raise ValueError('Malformed video frame path')
        index = int(parts[1])
        if index < 0:
            raise ValueError('Negative frame index')
        grouped.setdefault(parts[0], {}).setdefault(index, []).append(position)
    for video, wanted in grouped.items():
        with av.open(video) as container:
            if len(container.streams.video) != 1:
                raise ValueError('Exactly one video stream required')
            container.streams.video[0].thread_count = 2
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    rgb = Image.fromarray(frame.to_ndarray(format='rgb24')).resize(
                        (width, height), Image.Resampling.BILINEAR)
                    for position in wanted[index]:
                        result[position] = rgb.copy()
                if index >= max(wanted):
                    break
    if any(frame is None for frame in result):
        raise ValueError('Missing/out-of-range decoded frame; no black-frame fallback')
    return result
