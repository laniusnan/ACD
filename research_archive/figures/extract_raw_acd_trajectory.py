#!/usr/bin/env python3
"""Read-only matched raw-KD vs ACD checkpoint-trajectory probe."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision import datasets

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; STUDY=ROOT/'reproductions/acd_cnn_interaction_20261109'; OUT=HERE/'mechanism_figures'
EPOCHS=(1,30,120,240); ARMS={'raw_kd':'linear_kd','acd':'cosine_standardized_full'}; T=4.0
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def norm(z):return (z-z.mean(1,keepdim=True))/z.std(1,keepdim=True).clamp_min(1e-6)
def cond(z,y):
 p=(z/T).softmax(1); mask=torch.ones_like(p,dtype=torch.bool);mask.scatter_(1,y[:,None],False)
 return p[mask].reshape(len(y),-1)/(1-p.gather(1,y[:,None])).clamp_min(1e-12)
def raw_kl(z,q):return float((T*T*F.kl_div((z/T).log_softmax(1),(q/T).softmax(1),reduction='batchmean')).detach())
def tv(z,q,y):return float(.5*(cond(z,y)-cond(q,y)).abs().sum(1).mean())
def main():
 study=module('raw_acd_trajectory_study',STUDY/'study.py'); pair='vgg13_mv1'; seed=2102; job0=study.jobs()[f'{pair}__s{seed}__linear_kd']
 study.core.seed_all(seed); _,teacher,_,_=study.build(job0);teacher.eval()
 data=datasets.CIFAR100(ROOT/'data',train=False,download=False,transform=study.base.clean_transform());x=torch.stack([data[i][0] for i in range(128)]);y=torch.tensor([data[i][1] for i in range(128)])
 with torch.no_grad():q=teacher(x)[0]
 values={};sources={}
 for label,arm in ARMS.items():
  rows=[];initial=[]
  for epoch in EPOCHS:
   job=study.jobs()[f'{pair}__s{seed}__{arm}'];student,_,_,_=study.build(job);p=STUDY/'artifacts'/job['name']/f'epoch_{epoch:03d}.pth';ckpt=torch.load(p,map_location='cpu',weights_only=True);student.load_state_dict(ckpt['student'],strict=True);student.eval()
   with torch.no_grad():z=student(x)[0]
   rows.append({'epoch':epoch,'raw_kl':raw_kl(z,q),'conditional_tv_standardized':tv(norm(z),norm(q),y),'conditional_tv_raw':tv(z,q,y)})
   initial.append(ckpt['initial']);sources[str(p.relative_to(ROOT))]=digest(p)
  if len(set(initial))!=1:raise RuntimeError('Initial backbone changed within arm')
  values[label]={'arm':arm,'rows':rows,'initial':initial[0]}
 if values['raw_kd']['initial']!=values['acd']['initial']:raise RuntimeError('Arms do not share initialization')
 teacher_path=ROOT/'reproductions/head_transfer_20261021/frozen/teachers/best_model_weights_vgg13.pth';sources[str(teacher_path.relative_to(ROOT))]=digest(teacher_path)
 result={'status':'completed','scope':'Read-only matched-checkpoint trajectory probe; no training, endpoint evaluation, or queue mutation.','setup':{'pair':'VGG13→MobileNetV1','seed':seed,'epochs':list(EPOCHS),'probe':'first 128 official CIFAR-100 test images','temperature':T,'raw_arm':'linear KD','acd_arm':'cosine head + standardized full loss','pairing':'same registered interaction study and identical recorded initial backbone hash'},'results':values,'sources_sha256':sources,'limitations':['One pair and one seed; markers are checkpoints, not independent repetitions.','Raw KL and standardized conditional TV measure different coordinate systems and are shown jointly rather than treated as directly interchangeable.','The plot documents trajectories on a fixed probe and does not prove the mechanism causes endpoint accuracy.']}
 OUT.mkdir(exist_ok=True);(OUT/'raw_acd_checkpoint_trajectory_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(OUT/'raw_acd_checkpoint_trajectory_probe.json')
if __name__=='__main__':main()
