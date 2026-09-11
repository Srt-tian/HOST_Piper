"""CPU tiny-model proof that retaining only last hidden states preserves embeddings/loss."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'alignment'))
def run(root,weights):
    import torch
    import os
    os.environ['HOST_ALIGNMENT_LOGITS_TO_KEEP']='1'
    from piper_smoke import configure,batch,make_model,loss_for
    from piper_dtw_eval_helpers import install_last_hidden_only
    torch.set_num_threads(4); torch.manual_seed(42)
    cfg=configure(Path(root),4)
    data=batch(Path(root),weights,cfg,0)
    model=make_model(weights,True).eval()
    steps=torch.cat([data['chosen_steps'],data['ref_chosen_steps']],dim=0)
    lengths=torch.cat([data['seq_lens'],data['ref_seq_lens']],dim=0)
    with torch.no_grad():
        before_emb=model(copy.deepcopy(data),steps,lengths,training=False)['embs'].clone()
        before=loss_for(model,data,0,training=False)
        handles=install_last_hidden_only(model)
        after_emb=model(copy.deepcopy(data),steps,lengths,training=False)['embs'].clone()
        after=loss_for(model,data,0,training=False)
        from piper_serial_inference import install_serial_cnn
        install_serial_cnn(model)
        serial_emb=model(copy.deepcopy(data),steps,lengths,training=False)['embs'].clone()
        serial_loss=loss_for(model,data,0,training=False)
    torch.testing.assert_close(before_emb,serial_emb,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(before,serial_loss,rtol=1e-5,atol=1e-6)
    print(json.dumps(dict(serial_embedding_max_abs=float((before_emb-serial_emb).abs().max()),serial_loss=float(serial_loss))))
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    torch.testing.assert_close(before_emb,after_emb,rtol=0,atol=0)
    print(json.dumps(dict(passed=True,random_tiny_cpu_only=True,loss=float(before),
                         last_hidden_only_exact_loss_equality=True,exact_embedding_equality=True)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--weights',required=True)
    a=p.parse_args();run(a.root,a.weights)
