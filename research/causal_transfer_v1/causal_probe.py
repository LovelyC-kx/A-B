#!/usr/bin/env python3
"""Frozen-checkpoint local causal update probe for existing NSCO formal ANN experiments.
No optimizer.step, no checkpoint writes, no test images. Uses the immutable experiment snapshot.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def load_tools(repo):
    formal=repo/'formal'
    assert (formal/'p0_gain_sweep.py').is_file(),f'Missing required formal/p0_gain_sweep.py at {formal}'
    sys.path.insert(0,str(formal))
    import p0_gain_sweep as p0
    return p0

def prep_batch(batch,device,side,alpha):
    x=batch['input'].to(device,non_blocking=True)
    y=batch['target'].to(device,non_blocking=True)
    if x.ndim!=4 or x.shape!=y.shape or x.shape[1]!=3:raise ValueError('Expected RGB matched BCHW paired images')
    h,w=x.shape[-2:]
    if min(h,w)<side:raise ValueError(f'Image {h}x{w} smaller than crop {side}')
    a=(h-side)//2;b=(w-side)//2
    x=x[...,a:a+side,b:b+side];y=y[...,a:a+side,b:b+side]
    # A and B are synthetic-strength conditions derived from identical observed cloudy/clear pairs.
    x=(y+alpha*(x-y)).clamp(0,1)
    return x,y

def mse(model,batch,device,side,alpha):
    x,y=prep_batch(batch,device,side,alpha)
    p=model(x)
    if p.shape!=y.shape:raise RuntimeError(f'Model prediction shape {p.shape} differs from target {y.shape}')
    return F.mse_loss(p,y)

def params(model):return [(n,p) for n,p in model.named_parameters() if p.requires_grad]

def grad_for(model,entries,batches,device,side,alpha):
    model.zero_grad(set_to_none=True)
    acc=[torch.zeros_like(p) for _,p in entries]
    for batch in batches:
        loss=mse(model,batch,device,side,alpha)
        gg=torch.autograd.grad(loss,[p for _,p in entries],allow_unused=True)
        for a,g in zip(acc,gg):
            if g is not None:a.add_(g.detach(),alpha=1/len(batches))
    return acc

def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum().item()) for g in gs))
def dot(a,b):return sum(float((x.detach().double()*y.detach().double()).sum().item()) for x,y in zip(a,b))
def group(n):
    # Actual encoder/decoder groups; unexpected names remain visible as groups.
    ss=n.split('.')
    if ss[0] in ('encoders','decoders','downs','ups') and len(ss)>1:return '.'.join(ss[:2])
    return ss[0]

def group_indices(entries):
    d={}
    for i,(name,_) in enumerate(entries):d.setdefault(group(name),[]).append(i)
    return {k:v for k,v in sorted(d.items()) if any(entries[i][1].numel()>0 for i in v)}

def scaled_direction(gg,indices,global_l2,reference_norm):
    picked=[gg[i] for i in indices];n=norm(picked)
    if n<=1e-20:return None
    # Every compared intervention has EXACTLY the same global L2 update magnitude.
    factor=reference_norm/n
    return {i:gg[i]*factor for i in indices}

def random_direction(entries,ref_l2,seed,indices=None):
    rng_state=torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        idx=range(len(entries)) if indices is None else indices
        z={i:torch.randn_like(entries[i][1]) for i in idx}
    finally:torch.set_rng_state(rng_state)
    n=math.sqrt(sum(float(t.double().square().sum()) for t in z.values()))
    return {i:t*(ref_l2/n) for i,t in z.items()}

def eval_batch(model,batches,device,side,alpha):
    with torch.no_grad():return sum(float(mse(model,b,device,side,alpha)) for b in batches)/len(batches)

def intervention(model,entries,direction,eta,batches,device,side,alpha,baseline=None):
    if direction is None:return None
    # The only changed state is parameter value; buffers stay frozen in eval mode.
    backup={i:entries[i][1].detach().clone() for i in direction}
    try:
        with torch.no_grad():
            for i,d in direction.items(): entries[i][1].add_(d,alpha=-eta)
        after=eval_batch(model,batches,device,side,alpha)
    finally:
        with torch.no_grad():
            for i,old in backup.items():entries[i][1].copy_(old)
    before=baseline if baseline is not None else eval_batch(model,batches,device,side,alpha)
    return {'before':before,'after':after,'improvement':before-after}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',default='.')
    ap.add_argument('--experiment',required=True)
    ap.add_argument('--variant',default='N0_nafnet')
    ap.add_argument('--checkpoint',choices=['best','last'],default='best')
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--crop',type=int,default=128)
    ap.add_argument('--train-images',type=int,default=6)
    ap.add_argument('--pilot-images',type=int,default=6)
    ap.add_argument('--audit-images',type=int,default=6)
    ap.add_argument('--eta',type=float,default=0.003)
    ap.add_argument('--alpha-a',type=float,default=0.5)
    ap.add_argument('--alpha-b',type=float,default=1.0)
    ap.add_argument('--max-groups',type=int,default=8)
    ap.add_argument('--seed',type=int,default=1711)
    ap.add_argument('--trust-own-checkpoints',action='store_true')
    a=ap.parse_args()
    if not a.trust_own_checkpoints:ap.error('Explicit --trust-own-checkpoints required')
    if a.crop<32 or a.crop%16 or min(a.train_images,a.pilot_images,a.audit_images)<2 or a.eta<=0 or a.max_groups<1:ap.error('Invalid experiment dimensions')
    if not (0<a.alpha_a<a.alpha_b<=1):ap.error('Require 0 < alpha-a < alpha-b <= 1')
    repo=Path(a.repo).resolve();exp=Path(a.experiment).resolve();out=Path(a.out).resolve()
    if out.exists():raise FileExistsError(f'Output must not exist: {out}')
    if not exp.is_dir():raise FileNotFoundError(exp)
    if exp==out or exp in out.parents or out in exp.parents:raise ValueError('Output must be outside experiment')
    p0=load_tools(repo)
    cfg=json.loads((exp/'frozen_config.json').read_text())
    if cfg.get('dataset')!='CUHK-CR1-dedup-v1' or (cfg['splits']['train']['count'],cfg['splits']['val']['count'])!=(466,52):raise ValueError('Requires frozen CR1 466/52 split')
    hashes=p0.verify_snapshots(exp,cfg)
    mods,tr=p0.load_snapshot(exp)
    if a.variant not in mods.ANN:raise ValueError('This probe is ANN-only; no unverified SNN reset semantics')
    job=exp/'jobs'/f'{a.variant}_seed0';meta=json.loads((job/'metadata.json').read_text())
    if meta['variant']!=a.variant or meta['seed']!=0 or meta['family']!='ann':raise ValueError('Job metadata mismatch')
    cp,step,state,ref=p0.inspect_checkpoint(job,a.checkpoint,meta,True)
    torch.manual_seed(a.seed);random.seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    model=mods.build_model(a.variant,orbitalif_root=cfg['source_root'],width=cfg['ann']['width'],core_width=cfg['ann']['core_width']).to(device)
    model.load_state_dict(state,strict=True);model.eval()
    entries=params(model)
    ds_train=tr.PairDataset(cfg['splits']['train']['manifest'],patch=cfg['patch'],train=False)
    ds_val=tr.PairDataset(cfg['splits']['val']['manifest'],patch=cfg['patch'],train=False)
    if len(ds_train)!=466 or len(ds_val)!=52:raise RuntimeError('Unexpected dataset size')
    if a.pilot_images+a.audit_images>52 or a.train_images>466:raise ValueError('Requested too many images')
    train_idx=random.Random(a.seed).sample(range(466),a.train_images)
    val_idx=random.Random(a.seed+1).sample(range(52),a.pilot_images+a.audit_images)
    def get_batches(ds,idx):
        return [next(iter(DataLoader(Subset(ds,[i]),batch_size=1,num_workers=0))) for i in idx]
    train_batches=get_batches(ds_train,train_idx)
    pilot=get_batches(ds_val,val_idx[:a.pilot_images]); audit=get_batches(ds_val,val_idx[a.pilot_images:])
    # Freeze BatchNorm statistics, RNG, loss definition, crop coordinates, checkpoint, and data indices.
    ga=grad_for(model,entries,train_batches,device,a.crop,a.alpha_a)
    gb=grad_for(model,entries,train_batches,device,a.crop,a.alpha_b)
    na,nb=norm(ga),norm(gb)
    if min(na,nb)<1e-16:raise RuntimeError('Zero task gradient: cannot perform matched-norm controls')
    groups=group_indices(entries)
    ranked=sorted(groups,key=lambda k:sum(entries[i][1].numel() for i in groups[k]),reverse=True)[:a.max_groups]
    # Normalize global weight displacement to eta * ||g_A||; isolates orientation from norm.
    full=scaled_direction(ga,list(range(len(entries))),1,na)
    ctrl=random_direction(entries,na,a.seed+310)
    ref_a=eval_batch(model,audit,device,a.crop,a.alpha_a)
    ref_b=eval_batch(model,audit,device,a.crop,a.alpha_b)
    checks=[]
    for name,d in [('A_gradient',full),('random_equal_norm',ctrl),('negative_A_gradient',{i:-v for i,v in full.items()})]:
        for condition,alpha,baseline in [('A',a.alpha_a,ref_a),('B',a.alpha_b,ref_b)]:
            result=intervention(model,entries,d,a.eta,audit,device,a.crop,alpha,baseline)
            checks.append({'direction':name,'evaluation':condition,**result})
    # Select a group on pilot ONLY, evaluate selected group against random matched group on audit.
    pilot_base=eval_batch(model,pilot,device,a.crop,a.alpha_b)
    scores=[]
    for name in ranked:
        di=scaled_direction(ga,groups[name],1,na)
        if di is None:continue
        r=intervention(model,entries,di,a.eta,pilot,device,a.crop,a.alpha_b,pilot_base)
        ai=[ga[i] for i in groups[name]];bi=[gb[i] for i in groups[name]]
        c=dot(ai,bi)/(norm(ai)*norm(bi)+1e-30)
        scores.append({'group':name,'pilot_improvement_B':r['improvement'],'gradient_cosine':c,'parameter_count':sum(entries[i][1].numel() for i in groups[name])})
    if len(scores)<2:raise RuntimeError('Need at least two nonzero groups')
    chosen=max(scores,key=lambda z:z['pilot_improvement_B'])['group']
    chosen_cos=max(scores,key=lambda z:z['gradient_cosine'])['group']
    matched_random=random.Random(a.seed+987).choice([g['group'] for g in scores if g['group']!=chosen])
    screened=[]
    for label,gname in [('pilot_selected',chosen),('cosine_selected',chosen_cos),('random_group',matched_random)]:
        di=scaled_direction(ga,groups[gname],1,na)
        screened.append({'policy':label,'group':gname,**intervention(model,entries,di,a.eta,audit,device,a.crop,a.alpha_b,ref_b)})
    # Explicit restore checks after all interventions.
    replay_b=eval_batch(model,audit,device,a.crop,a.alpha_b)
    if abs(replay_b-ref_b)>1e-7:raise RuntimeError('Base checkpoint changed after interventions')
    out.mkdir(parents=True,exist_ok=False)
    result={'status':'completed','protocol':'frozen_one_step_paired_counterfactual_v1','interpretation':'Controlled local SGD-direction perturbations, NOT historical AdamW step or end-to-end training gain. Val was used for original checkpoint selection; audit subset is independent only from the pilot group selection, not an untouched final test.',
     'variant':a.variant,'checkpoint':a.checkpoint,'checkpoint_step':step,'checkpoint_sha256':sha(cp),'snapshot_hashes':hashes,
     'config':vars(a),'train_indices':train_idx,'pilot_val_indices':val_idx[:a.pilot_images],'audit_val_indices':val_idx[a.pilot_images:],
     'conditions':{'A':a.alpha_a,'B':a.alpha_b,'construction':'x_alpha=clip(y+alpha*(x_cloud-y),0,1)','same_base_image_pair':True},
     'grad_A_norm':na,'grad_B_norm':nb,'grad_AB_cosine':dot(ga,gb)/(na*nb),'baseline_audit':{'A_mse':ref_a,'B_mse':ref_b},
     'matched_direction_checks':checks,'group_pilot':scores,'group_audit':screened,'post_restore_error':abs(replay_b-ref_b),
     'real_model':True,'test_read':False,'optimizer_steps':0,'seed':a.seed}
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (out/'group_pilot.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
    with (out/'group_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(screened[0]));w.writeheader();w.writerows(screened)
    print(json.dumps({'status':'completed','out':str(out),'AB_gradient_cosine':result['grad_AB_cosine'],'policies':screened},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()