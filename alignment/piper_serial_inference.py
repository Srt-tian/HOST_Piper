"""Inference-only independent Qwen row encoding; preserve all sampled anchors."""
import types
import torch

def split_qwen_rows(inputs, config):
    ids=inputs['input_ids']
    rows=[{k: inputs[k][i:i+1] for k in
           ('input_ids','attention_mask','num_mains','num_refs')}
          for i in range(len(ids))]
    for row in rows:
        row['cls_token_id']=inputs['cls_token_id']
    merge=config.vision_config.spatial_merge_size**2
    for pixels,grid,token in (
        ('pixel_values','image_grid_thw',config.image_token_id),
        ('pixel_values_videos','video_grid_thw',config.video_token_id)):
        values=inputs.get(pixels); grids=inputs.get(grid)
        if values is None:
            if grids is not None: raise ValueError('Grid without pixels')
            for row in rows: row[pixels]=row[grid]=None
            continue
        sizes=[int(g.prod()) for g in grids]
        gi=pi=0
        for row in rows:
            wanted=int((row['input_ids']==token).sum())
            start_g,start_p=gi,pi
            found=0
            while found<wanted and gi<len(sizes):
                if sizes[gi]%merge: raise ValueError('Non-integral visual token count')
                found+=sizes[gi]//merge
                pi+=sizes[gi]; gi+=1
            if found!=wanted: raise ValueError('Visual grids do not match row token count')
            row[pixels]=values[start_p:pi] if pi>start_p else None
            row[grid]=grids[start_g:gi] if gi>start_g else None
        if gi!=len(sizes) or pi!=len(values):
            raise ValueError('Unconsumed visual grids/pixels')
    return rows

def merge_qwen_rows(rows):
    result={}
    for key in rows[0]:
        values=[r[key] for r in rows]
        if key=='cls_token_id':
            result[key]=values[0]
        elif all(v is None for v in values):
            result[key]=None
        else:
            result[key]=torch.cat([v for v in values if v is not None],dim=0)
    return result

def install_serial_cnn(model, rows_per_forward=1):
    if rows_per_forward not in (1,2,4,8):
        raise ValueError('Supported packed-row batch sizes:1,2,4,8')
    original=model.cnn.forward
    def serial(cnn, data):
        if torch.is_grad_enabled():
            raise ValueError('Serial encoding is inference-only')
        mains=[]; refs=[]
        rows=split_qwen_rows(data['qwen_input'],cnn.base_model.config)
        for start in range(0,len(rows),rows_per_forward):
            row=merge_qwen_rows(rows[start:start+rows_per_forward])
            emb=original({'qwen_input':row})
            nm=int(row['num_mains'].sum()); nr=int(row['num_refs'].sum())
            if len(emb)!=nm+nr: raise ValueError('Unexpected extracted anchor count')
            mains.append(emb[:nm]); refs.append(emb[nm:])
        return torch.cat(mains+refs,dim=0)
    model.cnn.forward=types.MethodType(serial,model.cnn)
    return original
