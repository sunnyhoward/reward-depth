"""Is the install rank-1 at L20 for the ONE-STAGE arms too, or only for the two-stage ones?
Identical construction for every arm: take that arm's OWN lora_A rows from blocks 21-25
(residual-input modules), SVD, ablate the top-k right-singular directions at block 20's output,
measure the preference margin. Random subspace at matched rank as control."""
import os, sys, json, random, torch
os.environ['SUP_MODEL']='Qwen/Qwen3.5-4B'
os.environ['SUP_BRIT']='/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl'
sys.path.insert(0,'/workspace/rd-branch/probefix'); sys.path.insert(0,'/workspace/rd-branch/supervisor'); sys.path.insert(0,'/workspace/rd-branch')
from sup_common import pair_texts, encode, span_mask, _all_rows
from pf_common import inner
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HF='/workspace/.hf_home/hub/models--sunnyhoward--reward-depth-probefix4b/snapshots/3b55e6a54b017b522c841b8033807ae614bce53e'
SCR='/tmp/claude-0/-workspace/1fa80ebe-7585-4e64-b90f-4e37aa002d72/scratchpad'
DEV='cuda'
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', padding_side='left')
rows=_all_rows(); rng=random.Random(0)
ev=[r for r in rows if r.get('eval_bucket')=='legacy']; rng.shuffle(ev); ev=ev[:96]

def read_subspace(d):
    sd=load_file(f'{d}/adapter_model.safetensors')
    A=torch.cat([sd[k].float() for k in sd if 'lora_A' in k
                 and any(f'.layers.{b}.' in k for b in range(21,26))
                 and any(m in k for m in ('q_proj','k_proj','v_proj','gate_proj','up_proj'))]).to(DEV)
    return torch.linalg.svd(A,full_matrices=False).Vh

def hk_ab(M):
    def hk(m,i,o):
        h=o[0] if isinstance(o,(tuple,list)) else o
        hf=h.float(); hf=hf-(hf@M.T)@M; h2=hf.to(h.dtype)
        return (h2,)+tuple(o[1:]) if isinstance(o,(tuple,list)) else h2
    return hk

def margin(model,hook=None):
    B=list(inner(model).layers); hs=[]
    if hook is not None: hs.append(B[20].register_forward_hook(hook))
    tot=0.0; n=0
    try:
        for i in range(0,len(ev),8):
            trip=pair_texts(tok,ev[i:i+8]); texts=[t for c,j,_ in trip for t in (c,j)]
            plens=[pl for _,_,pl in trip for _ in (0,1)]
            enc=encode(tok,texts,max_length=256).to(DEV); m=span_mask(tok,texts,plens,enc).to(DEV)
            with torch.no_grad(): lg=model(**enc).logits[:,:-1].float()
            lp=((lg.gather(-1,enc.input_ids[:,1:].unsqueeze(-1)).squeeze(-1)-lg.logsumexp(-1))*m).sum(-1)
            d=lp[0::2]-lp[1::2]; tot+=float(d.sum()); n+=d.numel()
    finally:
        for h in hs: h.remove()
    return tot/n

ARMS=[('C0',[f'{HF}/B4_probe_L20/ckpt300',f'{HF}/C0_two_plain/ckpt600']),
      ('C1',[f'{HF}/B4_probe_L20/ckpt300',f'{HF}/C1_two_dpop/ckpt600']),
      ('P0',[f'{HF}/P0_all_plain/ckpt600']),
      ('P1',[f'{HF}/P1_all_dpop/ckpt600']),
      ('D2',['/workspace/probefix4b_rpo/D2_upper_nostage1/ckpt600'])]
KS=[1,2,4,8,32]
g=torch.Generator(device=DEV).manual_seed(0)
out={}
for tag,ads in ARMS:
    b=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',dtype=torch.bfloat16,device_map=DEV)
    m=b
    for a in ads[:-1]: m=PeftModel.from_pretrained(m,a).merge_and_unload()
    mm=PeftModel.from_pretrained(m,ads[-1])
    V=read_subspace(ads[-1])          # each arm's OWN read subspace
    m0=margin(mm); res={'m0':m0}
    print(f'\n{tag}: no ablation {m0:.2f}',flush=True)
    for k in KS:
        real=margin(mm,hk_ab(V[:k]))
        R=torch.linalg.qr(torch.randn(2560,k,generator=g,device=DEV)).Q.T
        rand=margin(mm,hk_ab(R))
        res[k]=dict(real=real,rand=rand)
        print(f'  k={k:3d}  own-subspace {real:8.2f} ({100*real/m0:5.1f}%)   random {rand:8.2f} ({100*rand/m0:5.1f}%)',flush=True)
    out[tag]=res
    del mm,m,b; torch.cuda.empty_cache()
json.dump(out,open(f'{SCR}/rank_sweep_all.json','w'),indent=1)
print('\nDONE')
