"""Portable training/evaluation entry point; no automatic GPU job launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from acd.data import datasets_for, seed_worker
from acd.losses import distillation_loss
from acd.models import build, logits


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def load_model(path, model):
    # Only load checkpoints from a trusted source; historical files contain RNG objects.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('student', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=True)
    return ckpt


def evaluate(model, loader, device):
    model.eval(); correct = count = 0
    with torch.inference_mode():
        for x, y in loader:
            z = logits(model, x.to(device)); correct += int((z.argmax(1).cpu()==y).sum()); count += len(y)
    return {'top1': 100*correct/count, 'samples': count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--teacher-checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/run'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,help='Override data-loader workers; use 0 in restricted CPU environments')
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate',type=Path)
    p.add_argument('--download',action='store_true')
    p.add_argument('--method',choices=['ce','kd','angular_kd','standardized_full','conditional','acd'])
    p.add_argument('--etf',action='store_true')
    p.add_argument('--max-steps',type=int,help='Smoke check only; not a complete experiment')
    args=p.parse_args(); cfg=json.loads(args.config.read_text())
    if args.method: cfg['method']=args.method
    if args.etf: cfg['etf']=True
    if args.workers is not None: cfg['workers']=args.workers
    seed=cfg['seed']; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    angular=cfg['method'] not in ('ce','kd')
    student,backbone=build(cfg['student'],cfg['dataset'],angular,cfg['scale'],cfg.get('etf',False))
    student.to(device)
    train,test=datasets_for(cfg,args.data_root,args.download)
    val=DataLoader(test,batch_size=256,num_workers=cfg['workers'])
    if args.evaluate:
        load_model(args.evaluate,student)
        print(json.dumps(evaluate(student,val,device)));return
    teacher=None
    if cfg['method']!='ce':
        if not args.teacher_checkpoint: p.error('--teacher-checkpoint is required for distillation')
        teacher,_=build(cfg['teacher'],cfg['dataset'])
        load_model(args.teacher_checkpoint,teacher)
        teacher.to(device).eval().requires_grad_(False)
    optimizer=torch.optim.SGD(student.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=cfg['lr_decay'])
    start=0
    if args.resume:
        ckpt=load_model(args.resume,student)
        if ckpt['config']!=cfg: raise ValueError('Resume configuration differs from checkpoint')
        if ckpt.get('teacher_sha256') != (sha(args.teacher_checkpoint) if teacher else None): raise ValueError('Resume teacher differs')
        optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler']);start=ckpt['epoch']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    teacher_sha=sha(args.teacher_checkpoint) if teacher else None
    steps=0
    metrics=ckpt['validation'] if args.resume else None
    for epoch in range(start,cfg['epochs']):
        # Epoch-derived RNG and loader seed permit exact epoch-boundary restart.
        value=seed*1000+epoch+1;random.seed(value);np.random.seed(value);torch.manual_seed(value)
        generator=torch.Generator().manual_seed(value)
        loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['workers'],worker_init_fn=seed_worker,generator=generator,pin_memory=device.type=='cuda')
        student.train(); total_loss=count=0
        for x,y in loader:
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            z=logits(student,x)
            with torch.no_grad(): t=logits(teacher,x) if teacher else torch.zeros_like(z)
            loss,diag=distillation_loss(z,t,y,backbone,cfg)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),float('inf'),error_if_nonfinite=True);optimizer.step()
            total_loss+=float(loss.detach())*len(y);count+=len(y);steps+=1
            if args.max_steps and steps>=args.max_steps:
                save(args.output/'smoke.pth',{'student':student.state_dict(),'config':cfg,'steps':steps,'status':'incomplete_smoke'})
                print(json.dumps({'status':'incomplete_smoke','steps':steps,'loss':float(loss.detach())}));return
        scheduler.step()
        metrics=evaluate(student,val,device)
        payload={'student':student.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'config':cfg,'teacher_sha256':teacher_sha,'validation':metrics}
        save(args.output/'latest.pth',payload)
        record={'epoch':epoch+1,'loss':total_loss/count,**metrics}
        with (args.output/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    (args.output/'final.json').write_text(json.dumps({'status':'completed','epoch':cfg['epochs'],'validation':metrics,'checkpoint_sha256':sha(args.output/'latest.pth')},indent=2)+'\n')

if __name__=='__main__':main()
