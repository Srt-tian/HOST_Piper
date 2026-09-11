"""Offline compact DTW review: three-camera frames, matched-vs-time baseline and curves."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'alignment'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'data_preprocessing/piper'))

def render(root):
    import numpy as np
    from PIL import Image,ImageDraw
    from piper_low_memory import sequential_load_imgs
    from build_progress import anchor_frames
    root=Path(root)
    output=root/'review'
    output.mkdir(exist_ok=False)
    records=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines() if line]
    cases=[]
    for i,record in enumerate(records):
        directory=output/f'case_{i:02d}'
        directory.mkdir()
        main=anchor_frames(record['frame_paths'],record['main_video_path'],record['main_frames'],1)
        ref=anchor_frames(record['ref_frame_paths'],record['ref_video_path'],record['ref_frames'],1)
        match=np.asarray(record['forward_argmax_indices'],dtype=int)
        baseline_target=main/max(1,record['main_frames']-1)*(record['ref_frames']-1)
        baseline=np.abs(ref[None,:]-baseline_target[:,None]).argmin(axis=1)
        for tag,key in [('m','frame_paths'),('r','ref_frame_paths')]:
            paths=[p.rsplit('::',1)[0]+'::224x168' for p in record[key]]
            frames=sequential_load_imgs(paths)
            for n in range(len(frames)//3):
                canvas=Image.new('RGB',(224,504))
                for cam in range(3): canvas.paste(frames[n*3+cam],(0,168*cam))
                canvas.save(directory/f'{tag}{n:03d}.jpg',quality=78)
        xs=main/max(1,record['main_frames']-1)
        ys=(ref[match]+1)/record['ref_frames']
        points=' '.join(f'{40+x*600:.1f},{240-y*210:.1f}' for x,y in zip(xs,ys))
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="270" viewBox="0 0 680 270"><rect width="680" height="270" fill="#111c2d"/><path d="M40 30 V240 H640 M40 240 L640 30" stroke="#66768b" fill="none"/><polyline points="{points}" fill="none" stroke="#25d7bd" stroke-width="3"/><text x="42" y="22" fill="white">Reference progress: teal=HOST DTW, gray=time baseline</text><text x="430" y="262" fill="white">Main normalized time</text></svg>'
        (directory/'curve.svg').write_text(svg)
        sheet=Image.new('RGB',(1376,2240),'#0a1422')
        draw=ImageDraw.Draw(sheet)
        chosen=np.linspace(0,len(main)-1,8).astype(int)
        for k,n in enumerate(chosen):
            x=16+(k%2)*680;y=16+(k//2)*554
            draw.text((x,y),f'{record["task"]} main frame {main[n]} | DTW {ref[match[n]]} | TIME {ref[baseline[n]]}',fill='white')
            for col,(tag,idx) in enumerate([('m',n),('r',match[n]),('r',baseline[n])]):
                with Image.open(directory/f'{tag}{idx:03d}.jpg') as im: sheet.paste(im,(x+224*col,y+25))
        sheet.save(directory/'contact.jpg',quality=86)
        cases.append(dict(task=record['task'],directory=f'case_{i:02d}',
            main=main.tolist(),reference=ref.tolist(),matched=match.tolist(),baseline=baseline.tolist(),
            diagnostic=record['diagnostics'],main_path=record['main_video_path'],ref_path=record['ref_video_path']))
    payload=json.dumps(cases).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>HOST DTW validation</title>
<style>body{background:#091321;color:#e5edf9;font:15px system-ui;margin:28px auto;max-width:1120px}select,button{background:#20324d;color:white;border:1px solid #49617e;border-radius:7px;padding:9px;margin:4px}h1{font-size:25px}.muted{color:#9eb0c7}#frames{display:flex;gap:16px}#frames>div{flex:1}#frames img{width:100%;max-width:280px;border-radius:9px}#curve{width:680px;max-width:100%}input{width:70%}pre{white-space:pre-wrap;font-size:12px}a{color:#29ddc4}</style>
<h1>HOST · held-out DTW review</h1><p class="muted">Real validation videos. Main / model-matched reference / equal-time reference. Time baseline is not ground truth. Three views: front, left, right.</p>
<select id="cases"></select><button id="prev">◀</button><button id="play">Play</button><button id="next">▶</button><input type="range" id="step" min="0"><span id="counter"></span>
<div id="frames"><div><h3>Main</h3><img id="mainimg"></div><div><h3>HOST DTW reference</h3><img id="dtwimg"></div><div><h3>Time-index reference</h3><img id="timeimg"></div></div>
<p id="numbers"></p><img id="curve"><p><a id="contact" target="_blank">Open8-sample contact sheet</a></p><pre id="info"></pre>
<script>const data=PAYLOAD; const sel=document.getElementById('cases'), slider=document.getElementById('step'); let timer=null;
data.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+c.task;sel.appendChild(o)});
function imagePath(c,tag,n){return c.directory+'/'+tag+String(n).padStart(3,'0')+'.jpg'}
function show(){const c=data[+sel.value],n=+slider.value;document.getElementById('mainimg').src=imagePath(c,'m',n);document.getElementById('dtwimg').src=imagePath(c,'r',c.matched[n]);document.getElementById('timeimg').src=imagePath(c,'r',c.baseline[n]);document.getElementById('counter').textContent=(n+1)+' / '+c.main.length;document.getElementById('numbers').textContent='Frame: main '+c.main[n]+' | DTW '+c.reference[c.matched[n]]+' | time '+c.reference[c.baseline[n]]}
function change(){const c=data[+sel.value];slider.max=c.main.length-1;slider.value=0;document.getElementById('curve').src=c.directory+'/curve.svg';document.getElementById('contact').href=c.directory+'/contact.jpg';document.getElementById('info').textContent=JSON.stringify(c.diagnostic,null,2)+'\\nMain: '+c.main_path+'\\nRef: '+c.ref_path;show()}
slider.oninput=show;sel.onchange=change;document.getElementById('next').onclick=()=>{slider.value=Math.min(+slider.max,+slider.value+1);show()};document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,+slider.value-1);show()};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}else{timer=setInterval(()=>{slider.value=(+slider.value+1)%(+slider.max+1);show()},250);document.getElementById('play').textContent='Pause'}};change();</script>'''
    (output/'index.html').write_text(page.replace('PAYLOAD',payload))
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    print(json.dumps(dict(review=str(output),cases=len(cases),bytes=size)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    render(p.parse_args().root)
