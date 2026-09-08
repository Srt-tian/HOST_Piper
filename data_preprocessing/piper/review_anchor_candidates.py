"""Render bounded candidate contact sheets using sequential decode, no network/GPU.

This is evidence for review, not an automatic success classifier or DTW labeler.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def render(root, output):
    root, output = Path(root), Path(output)
    if not output.is_absolute() or output.exists():
        raise ValueError('Use a new absolute output directory')
    groups = json.loads((root/'anchor_review.json').read_text())['groups']
    output.mkdir(parents=True)
    evidence = []
    for group in groups:
        candidate = group['candidates'][0]
        ep = Path(candidate['path'])
        indices = np.linspace(0, candidate['frames']-1, 8, dtype=int).tolist()
        sheet = Image.new('RGB', (8*320, 3*240+60), '#101923')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f"{group['task']} / {group['split']} / {ep.parent.name} / {ep.name}", fill='white')
        draw.text((12, 25), 'Candidate only. Columns: source frame index. Rows: front / left / right.', fill='white')
        for column, index in enumerate(indices):
            draw.text((column*320+8, 44), str(index), fill='white')
        for row, camera in enumerate(('cam_front', 'cam_left', 'cam_right')):
            observed = set()
            with av.open(str(ep/f'{camera}.mp4')) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index in indices:
                        img = frame.to_image().resize((320, 240), Image.Resampling.LANCZOS)
                        sheet.paste(img, (indices.index(index)*320, 60+row*240))
                        observed.add(index)
            if observed != set(indices):
                raise ValueError('Missing candidate review frames')
        destination = output/f"{group['task']}_{group['split']}.jpg"
        sheet.save(destination, quality=92)
        evidence.append(dict(task=group['task'], split=group['split'], candidate=str(ep),
                             sampled_indices=indices, contact_sheet=str(destination), reviewed=False))
    with (output/'evidence.json').open('x') as handle:
        json.dump(evidence, handle, indent=2)
    print(json.dumps(dict(contact_sheets=len(evidence), reviewed=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    render(args.root, args.output)
