"""Portable training/evaluation entry point; no automatic GPU job launch."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from acd.data import datasets_for, evaluation_dataset, seed_worker
from acd.checkpoints import load_model, sha256 as sha, validate_resume
from acd.losses import distillation_loss
from acd.models import build, logits
from acd.heads import RotatingETFHead, optimizer_groups


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    os.replace(temporary, path)


def evaluate(model, loader, device):
    model.eval(); correct = top5 = count = 0
    total_ce = 0.
    with torch.inference_mode():
        for x, y in loader:
            y = y.to(device)
            z = logits(model, x.to(device))
            correct += int((z.argmax(1)==y).sum())
            top5 += int(z.topk(min(5, z.shape[1]), dim=1).indices.eq(y[:, None]).any(1).sum())
            total_ce += float(torch.nn.functional.cross_entropy(z, y, reduction='sum'))
            count += len(y)
    if count == 0:
        raise ValueError('Evaluation dataset is empty')
    return {'top1': 100*correct/count, 'top5': 100*top5/count, 'ce': total_ce/count, 'samples': count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--teacher-checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/run'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,help='Override data-loader workers; use 0 in restricted CPU environments')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--resume',type=Path)
    mode.add_argument('--evaluate',type=Path)
    p.add_argument('--evaluate-teacher',action='store_true',help='Use the affine teacher architecture with --evaluate')
    p.add_argument('--eval-batch-size',type=int,default=256)
    p.add_argument('--metrics-json',type=Path,help='Save evaluation metrics and checkpoint SHA256')
    p.add_argument('--allow-legacy-checkpoint',action='store_true',help='Allow Python objects in trusted historical checkpoints')
    p.add_argument('--download',action='store_true')
    p.add_argument('--method',choices=['ce','kd','angular_kd','standardized_full','conditional','acd'])
    p.add_argument('--etf',action='store_true')
    p.add_argument('--rotating-etf',action='store_true')
    p.add_argument('--stop-after-epoch',type=int,help='Save a resumable partial stage without marking the run complete')
    p.add_argument('--max-steps',type=int,help='Smoke check only; not a complete experiment')
    args=p.parse_args(); cfg=json.loads(args.config.read_text())
    if args.evaluate_teacher and not args.evaluate: p.error('--evaluate-teacher requires --evaluate')
    if args.metrics_json and not args.evaluate: p.error('--metrics-json requires --evaluate')
    if args.max_steps is not None and args.max_steps < 1: p.error('--max-steps must be positive')
    if args.eval_batch_size < 1: p.error('--eval-batch-size must be positive')
    if args.method: cfg['method']=args.method
    if args.etf: cfg['etf']=True
    if args.rotating_etf: cfg['rotating_etf']=True
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= cfg['epochs']:
        p.error('--stop-after-epoch must be between 1 and configured epochs')
    if args.workers is not None: cfg['workers']=args.workers
    if cfg['workers'] < 0: p.error('--workers must be nonnegative')
    if not args.evaluate and cfg['method'] != 'ce' and not args.teacher_checkpoint:
        p.error('--teacher-checkpoint is required for distillation')
    seed=cfg['seed']; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; pass --device cpu for CPU execution')
    role='teacher' if args.evaluate_teacher else 'student'
    angular=role == 'student' and cfg['method'] not in ('ce','kd')
    student,backbone=build(cfg[role],cfg['dataset'],angular,cfg['scale'],
                           cfg.get('etf',False) if role == 'student' else False,
                           rotating_etf=cfg.get('rotating_etf',False) if role == 'student' else False,
                           feature_lift=cfg.get('feature_lift',0) if role == 'student' else 0)
    student.to(device)
    if args.evaluate:
        load_model(args.evaluate,student,config=cfg,role=role,allow_legacy=args.allow_legacy_checkpoint)
        test=evaluation_dataset(cfg,args.data_root,args.download)
        val=DataLoader(test,batch_size=args.eval_batch_size,num_workers=cfg['workers'])
        metrics={**evaluate(student,val,device),'checkpoint_sha256':sha(args.evaluate),
                 'dataset':cfg['dataset'],'architecture':cfg[role],'role':role}
        if args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True,exist_ok=True)
            args.metrics_json.write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps(metrics));return
    if not args.resume and any((args.output/name).exists() for name in ('latest.pth','history.jsonl','final.json')):
        p.error('Output already contains a run; use --resume or choose a new --output directory')
    teacher=None
    if cfg['method']!='ce':
        teacher,_=build(cfg['teacher'],cfg['dataset'])
        load_model(args.teacher_checkpoint,teacher,config=cfg,role='teacher',allow_legacy=args.allow_legacy_checkpoint)
        teacher.to(device).eval().requires_grad_(False)
    rotating_heads=[module for module in student.modules() if isinstance(module, RotatingETFHead)]
    optimizer=torch.optim.SGD(optimizer_groups(student),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=cfg['lr_decay'])
    start=0
    if args.resume:
        ckpt=load_model(args.resume,student,config=cfg,allow_legacy=args.allow_legacy_checkpoint)
        validate_resume(ckpt,cfg,sha(args.teacher_checkpoint) if teacher else None)
        optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler']);start=ckpt['epoch']
    train,test=datasets_for(cfg,args.data_root,args.download)
    val=DataLoader(test,batch_size=args.eval_batch_size,num_workers=cfg['workers'])
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume and start >= cfg['epochs'] and not (args.output/'latest.pth').exists():
        save(args.output/'latest.pth',ckpt)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    teacher_sha=sha(args.teacher_checkpoint) if teacher else None
    steps=0
    metrics=ckpt['validation'] if args.resume else None
    stop_epoch=args.stop_after_epoch or cfg['epochs']
    if stop_epoch < start:
        p.error('Checkpoint is already beyond the requested stopping epoch')
    for epoch in range(start,stop_epoch):
        # Epoch-derived RNG and loader seed permit exact epoch-boundary restart.
        value=seed*1000+epoch+1;random.seed(value);np.random.seed(value);torch.manual_seed(value)
        generator=torch.Generator().manual_seed(value)
        loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['workers'],worker_init_fn=seed_worker,generator=generator,pin_memory=device.type=='cuda')
        student.train(); total_loss=count=0; epoch_start=time.monotonic()
        for batch_index,(x,y) in enumerate(loader,1):
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            z=logits(student,x)
            with torch.no_grad(): t=logits(teacher,x) if teacher else torch.zeros_like(z)
            loss,diag=distillation_loss(z,t,y,backbone,cfg)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward()
            for head in rotating_heads: head.project_gradient()
            torch.nn.utils.clip_grad_norm_(student.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step()
            for head in rotating_heads: head.retract(optimizer)
            total_loss+=float(loss.detach())*len(y);count+=len(y);steps+=1
            if batch_index == 1 or batch_index % 100 == 0:
                save_json(args.output/'progress.json',{'status':'training','epoch':epoch+1,
                          'epochs':cfg['epochs'],'stage_stop_epoch':stop_epoch,'batch':batch_index,
                          'batches':len(loader),'loss':total_loss/count,
                          'seconds_this_epoch':time.monotonic()-epoch_start,'updated_at':time.time()})
            if args.max_steps and steps>=args.max_steps:
                save(args.output/'smoke.pth',{'student':student.state_dict(),'config':cfg,'steps':steps,'status':'incomplete_smoke'})
                print(json.dumps({'status':'incomplete_smoke','steps':steps,'loss':float(loss.detach())}));return
        scheduler.step()
        metrics=evaluate(student,val,device)
        payload={'student':student.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'config':cfg,'teacher_sha256':teacher_sha,'validation':metrics}
        save(args.output/'latest.pth',payload)
        geometry=[head.geometry_error() for head in rotating_heads]
        record={'epoch':epoch+1,'loss':total_loss/count,**metrics}
        if geometry: record['etf_gram_max_error']=max(geometry)
        with (args.output/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    complete=stop_epoch == cfg['epochs']
    result={'status':'completed' if complete else 'stage_completed','epoch':stop_epoch,
            'validation':metrics,'checkpoint_sha256':sha(args.output/'latest.pth'),
            'evaluation_split':'training_holdout' if cfg.get('validation_per_class',0) else 'official_test'}
    save_json(args.output/('final.json' if complete else 'stage.json'),result)
    save_json(args.output/'progress.json',{**result,'updated_at':time.time()})

if __name__=='__main__':main()
