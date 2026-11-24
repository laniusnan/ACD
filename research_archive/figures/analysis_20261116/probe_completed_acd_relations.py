"""Fixed 1024-image CPU diagnostic for four completed ACI+SICD models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from render import plot, np, COLORS, HERE

ROOT = HERE.parents[2]
STUDY = ROOT / 'reproductions/cifar100_acd_rerun_20261116'
PAIRS = [('res56_res20','ResNet56 → ResNet20'),
         ('res110_res32','ResNet110 → ResNet32'),
         ('wrn402_wrn162','WRN40-2 → WRN16-2'),
         ('wrn402_wrn401','WRN40-2 → WRN40-1')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('completed_acd_methods',STUDY/'methods.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.set_num_threads(2)
    config=json.loads((STUDY/'models.json').read_text())
    policy=json.loads((STUDY/'protocol.json').read_text())
    ds=datasets.CIFAR100(STUDY/'data',train=False,download=False,transform=m.clean())
    loader=DataLoader(Subset(ds,list(range(1024))),batch_size=64,shuffle=False,num_workers=0)
    report={'scope':'Fixed-checkpoint CPU forward-only probe; no training',
            'samples':1024,'indices':'First 1024 official CIFAR-100 test images, indices 0–1023',
            'temperature':4,'standardization':'Per-image all-class sample SD, correction=1',
            'seed':3201,'epoch':240,'etf_initialization':False,'models':[],
            'limitations':['No same-recipe baseline or standardized-full control is included.',
                           'Across-pair discrepancies cannot rank method effectiveness: teachers and tasks differ.',
                           'Single trained seed per pair; image variability is not run uncertainty.',
                           'Probe subset, not full-test accuracy evaluation.']}
    arrays={}
    for pair,label in PAIRS:
        run=STUDY/'runs'/f'{pair}__s3201__acd'
        final=json.loads((run/'final.json').read_text())
        assert final['status']=='completed' and final['epoch']==240
        path=run/'latest.pth'
        assert sha(path)==final['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['job']['arm']=='acd' and state['epoch']==240
        identity=final['identity'].get('identity',final['identity'])
        for rel,expected in identity['files'].items():
            if rel.startswith(('frozen/models/','assets/teachers/')) or rel in ('methods.py','models.json','protocol.json','data/cifar-100-python/test'):
                assert sha(STUDY/rel)==expected,rel
        student=m.student(pair,3201,'acd',policy,config)[0].eval().requires_grad_(False)
        student.load_state_dict(state['student'],strict=True)
        teacher_name=config['pairs'][pair]['teacher']
        teacher=m.model(teacher_name,config).eval().requires_grad_(False)
        teacher_path=STUDY/'assets/teachers'/f'{teacher_name}.pth'
        teacher.load_state_dict(torch.load(teacher_path,map_location='cpu',weights_only=True),strict=True)
        before=[m.tensor_hash(net.state_dict()) for net in (student,teacher)]
        tvs=[]; kls=[]; masses=[]; correct_s=[]; correct_t=[]
        with torch.inference_mode():
            for images,ys in loader:
                raw_s=student(images)[0];raw_t=teacher(images)[0]
                zs=m.standardize(raw_s,policy['standardization']['epsilon'])
                zt=m.standardize(raw_t,policy['standardization']['epsilon'])
                mask=torch.ones_like(zs,dtype=torch.bool).scatter_(1,ys[:,None],False)
                log_q=(zs[mask].reshape(len(ys),99)/4).log_softmax(1)
                log_p=(zt[mask].reshape(len(ys),99)/4).log_softmax(1)
                p=log_p.exp();q=log_q.exp()
                mass=(zt/4).softmax(1).masked_fill(~mask,0).sum(1)
                tvs.extend((.5*(p-q).abs().sum(1)).tolist())
                kls.extend((16*mass*(p*(log_p-log_q)).sum(1)).tolist())
                masses.extend(mass.tolist())
                correct_s.extend((raw_s.argmax(1)==ys).tolist())
                correct_t.extend((raw_t.argmax(1)==ys).tolist())
        assert before==[m.tensor_hash(net.state_dict()) for net in (student,teacher)]
        tv=np.asarray(tvs);kl=np.asarray(kls)
        assert len(tv)==1024 and np.isfinite(tv).all() and np.isfinite(kl).all()
        summary={'pair':pair,'label':label,'checkpoint_sha256':sha(path),'teacher_sha256':sha(teacher_path),
                 'tv_mean':float(tv.mean()),'tv_median':float(np.median(tv)),
                 'tv_q25':float(np.quantile(tv,.25)),'tv_q75':float(np.quantile(tv,.75)),
                 'weighted_conditional_kl_mean':float(kl.mean()),
                 'probe_student_top1':float(np.mean(correct_s)*100),
                 'probe_teacher_top1':float(np.mean(correct_t)*100)}
        report['models'].append(summary)
        arrays[pair+'_tv']=tv;arrays[pair+'_weighted_kl']=kl
        print(json.dumps(summary),flush=True)
    stem=HERE/'fig_completed_acd_relations'
    stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(stem.with_suffix('.npz'),**arrays)
    render_saved(report, arrays)


def render_saved(report, arrays):
    stem=HERE/'fig_completed_acd_relations'
    fig,axes=plot.subplots(1,2,figsize=(10.5,3.7))
    labels=[]
    for i,(pair,label) in enumerate(PAIRS):
        tv=arrays[pair+'_tv']; color=COLORS[i]
        axes[0].plot(np.sort(tv),np.arange(1,len(tv)+1)/len(tv),color=color,linewidth=1.8,label=label)
        row=report['models'][i]
        axes[1].plot([row['tv_q25'],row['tv_q75']],[i,i],color=color,linewidth=5,alpha=.35)
        axes[1].scatter(row['tv_median'],i,color=color,marker='o',s=45,zorder=3)
        axes[1].scatter(row['tv_mean'],i,color=color,marker='D',s=35,zorder=3)
        labels.append(label)
        axes[1].annotate(f"({row['tv_median']:.5f}, {row['tv_mean']:.5f})",
                         (row['tv_mean'], i), xytext=(0, 12), textcoords='offset points',
                         ha='center', va='bottom', fontsize=9, color=color)
    axes[0].set_title('(a) Non-target relation discrepancy')
    axes[0].set_xlabel('');axes[0].set_ylabel('Cumulative fraction')
    axes[0].set_xlim(left=0);axes[0].set_ylim(0,1)
    axes[1].set_title('(b) Image IQR; (median, mean)')
    axes[1].set_yticks([]);axes[1].set_ylim(-.65, len(PAIRS)-.5);axes[1].invert_yaxis()
    axes[1].set_xlabel('')
    lower=min(row['tv_q25'] for row in report['models'])
    upper=max(row['tv_q75'] for row in report['models'])
    margin=(upper-lower)*.12
    axes[1].set_xlim(lower-margin,upper+margin)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=9)
    axes[1].scatter([],[],color='#52616B',marker='o',label='Median')
    axes[1].scatter([],[],color='#52616B',marker='D',label='Mean')
    axes[1].legend(loc='upper right',frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.87),w_pad=2)
    for ext in ('png','pdf','svg'):
        fig.savefig(stem.with_suffix('.'+ext),bbox_inches='tight',pad_inches=.04,facecolor='white')
    plot.close(fig)
    print('Done: figure, per-image arrays and provenance summary.',flush=True)


if __name__=='__main__':
    if '--render-only' in sys.argv:
        render_saved(json.loads((HERE/'fig_completed_acd_relations.json').read_text()),
                     np.load(HERE/'fig_completed_acd_relations.npz'))
    else:
        main()
