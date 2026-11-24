#!/usr/bin/env python3
"""Read-only checkpoint probe for parameterization-to-gradient mechanism figures.

This consumes one completed VGG13→MobileNetV1 linear-KD checkpoint and its
frozen teacher. It performs CPU inference on a fixed first-128 CIFAR-100 test
images, creates no training run, and never modifies a study artifact.
"""
import hashlib, importlib.util, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision import datasets

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OUT=HERE/'mechanism_figures'; OUT.mkdir(exist_ok=True)
STUDY=ROOT/'reproductions/acd_cnn_interaction_20261109'
JOB='vgg13_mv1__s2102__linear_kd'; CHECKPOINT=STUDY/'artifacts'/JOB/'epoch_240.pth'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def norm(z): return (z-z.mean(1,keepdim=True))/z.std(1,keepdim=True).clamp_min(1e-6)
def conditional_probs(z,y,T=4):
    p=(z/T).softmax(1); mask=torch.ones_like(p,dtype=torch.bool);mask.scatter_(1,y[:,None],False)
    return p[mask].reshape(len(y),-1)/(1-p.gather(1,y[:,None])).clamp_min(1e-12)
def grad(logits, target, h):
    loss=16*F.kl_div((logits/4).log_softmax(1),(target.detach()/4).softmax(1),reduction='batchmean')
    return torch.autograd.grad(loss,h)[0]
def cosine(a,b): return F.cosine_similarity(a,b,dim=1)

def main():
    study=module('force_probe_study',STUDY/'study.py'); job=study.jobs()[JOB]
    study.core.seed_all(2102); student,teacher,head,_=study.build(job)
    student.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['student'],strict=True);student.eval();teacher.eval()
    data=datasets.CIFAR100(ROOT/'data',train=False,download=False,transform=study.base.clean_transform())
    x=torch.stack([data[i][0] for i in range(128)]); y=torch.tensor([data[i][1] for i in range(128)])
    with torch.no_grad():
        teacher_logits,_=teacher(x); student_logits, student_feats=student(x); h0=student_feats['pooled_feat']; teacher_h=teacher(x)[1]['pooled_feat']
    W=student.fc.weight.detach(); b=student.fc.bias.detach(); WT=teacher.classifier.weight.detach(); bT=teacher.classifier.bias.detach()
    # The frozen models return logits from their exact linear heads. Assert this
    # before counterfactual reconstruction.
    torch.testing.assert_close(student_logits,F.linear(h0,W,b)); torch.testing.assert_close(teacher_logits,F.linear(teacher_h,WT,bT))
    def stages(h,weight,bias):
        unit=F.normalize(h,dim=1)@F.normalize(weight,dim=1).T
        scale=h.norm(dim=1,keepdim=True)*weight.norm(dim=1)[None,:]*unit
        return unit,scale,scale+bias[None,:]
    unitS,scaleS,fullS=stages(h0,W,b); unitT,scaleT,fullT=stages(teacher_h,WT,bT)
    # Real classifier logits are the endpoint; lambda continuously injects
    # scale/norm and bias into the fixed directional scores.
    target=fullT
    H=h0.detach().clone().requires_grad_(True)
    U,Sc,Fu=stages(H,W,b)
    refs={
        'angular': U,
        'scale_only': Sc,
        'bias_only': U+b[None,:],
        'raw': Fu,
    }
    # A scale-invariant reference uses standardized outputs. This is a local
    # comparison direction, not a claim that it is a semantic ground truth.
    canonical=norm(Fu); g_ref=grad(canonical,norm(target),H)
    gradients={}; stats={}
    for key,z in refs.items():
        H2=h0.detach().clone().requires_grad_(True); u,s,f=stages(H2,W,b); zz={'angular':u,'scale_only':s,'bias_only':u+b[None,:],'raw':f}[key]
        g=grad(zz,target,H2); gradients[key]=g.detach()
        stats[key]={'mean_cosine_to_standardized_reference':float(cosine(g.detach(),g_ref.detach()).mean()),
                    'median_cosine_to_standardized_reference':float(cosine(g.detach(),g_ref.detach()).median()),
                    'mean_norm_ratio_to_reference':float((g.detach().norm(dim=1)/g_ref.detach().norm(dim=1).clamp_min(1e-12)).mean())}
    # Fixed probe selects median raw/reference agreement—not an accuracy-picked
    # visual example. Its probability paths are exact counterfactuals.
    rawcos=cosine(gradients['raw'],g_ref.detach()); idx=int(rawcos.argsort()[len(rawcos)//2])
    lam=torch.linspace(0,1,21)[:,None]
    # alpha=0 is unit angular coordinates; alpha=1 is exact raw scores.
    curve=[]
    for a in lam[:,0]:
        z=(1-a)*unitS[idx]+a*fullS[idx]
        p=(z/4).softmax(0); q=(target[idx]/4).softmax(0)
        curve.append({'lambda':float(a),'student_non_target':conditional_probs(z[None],y[idx:idx+1])[0].tolist(),'teacher_non_target':conditional_probs(target[idx:idx+1],y[idx:idx+1])[0].tolist()})
    # Plot only the four most probable teacher alternatives, while values retain
    # all 99 non-target probabilities for reproduction.
    mask=torch.ones(100,dtype=torch.bool);mask[y[idx]]=False; classes=torch.arange(100)[mask]; top=conditional_probs(target[idx:idx+1],y[idx:idx+1])[0].topk(4).indices
    top_classes=classes[top].tolist()
    # Virtual update plane: orthonormalize reference and raw residual at this
    # real student feature. Coordinates are negative gradients (the SGD force).
    gr=g_ref[idx]; residual=gradients['raw'][idx]-(gradients['raw'][idx]@gr)/(gr@gr).clamp_min(1e-12)*gr; e1=gr/gr.norm().clamp_min(1e-12); e2=residual/residual.norm().clamp_min(1e-12)
    vectors={k:{'reference_axis':float((-g[idx])@e1),'artifact_axis':float((-g[idx])@e2),'norm':float(g[idx].norm()),'cosine_to_reference':float(F.cosine_similarity(g[idx,None],gr[None],dim=1)[0])} for k,g in gradients.items()}
    # Aggregate distribution deformation without selecting an example.
    def delta(a,b): return float((conditional_probs(a,y)-conditional_probs(b,y)).abs().sum(1).mean()/2)
    deformation={'angular_to_scale_tv':delta(unitS,scaleS),'scale_to_raw_bias_tv':delta(scaleS,fullS),'angular_to_raw_tv':delta(unitS,fullS)}
    sources={str(p.relative_to(ROOT)):digest(p) for p in [CHECKPOINT,STUDY/'study.py',ROOT/'reproductions/cvpr_angular_interface_interaction_20261109/losses.py',ROOT/'reproductions/head_transfer_20261021/frozen/teachers/best_model_weights_vgg13.pth']}
    result={'status':'completed','scope':'Read-only local gradient/counterfactual probe; no training, evaluation endpoint, or queue mutation.',
        'setup':{'pair':'VGG13→MobileNetV1','student_arm':'linear KD','seed':2102,'checkpoint_epoch':240,'probe':'first 128 official CIFAR-100 test images','temperature':4,'selected_sample_rule':'median raw-KL versus standardized-reference feature-gradient cosine','selected_index':idx,'selected_label':int(y[idx])},
        'formula':{'logit':'z_k = ||h|| ||w_k|| cos(theta_k) + b_k','raw_gradient':'∇_h L_KD = T Σ_k (p_k - q_k) w_k','reference':'KL between per-sample standardized student and teacher logits'},
        'gradient_statistics':stats,'distribution_deformation':deformation,'selected_probability_curve':curve,'selected_top_teacher_non_target_classes':top_classes,'virtual_update_plane':vectors,'sources_sha256':sources,
        'limitations':['The standardized-logit gradient is a diagnostic reference, not semantic ground truth.','This is one completed pair/seed and a 128-image fixed probe; it is not an accuracy or causal generalization result.','Virtual updates are local gradients, not a substitute for full-training trajectory evidence.','The student is linear-head KD; this figure does not measure ACI/SICD end-to-end behavior.']}
    (OUT/'parameterization_force_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(OUT/'parameterization_force_probe.json')
if __name__=='__main__':main()
