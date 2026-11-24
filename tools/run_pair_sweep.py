"""Prepare, validate and dispatch the bounded seven-arm CIFAR-100 study.

Each GPU worker holds a filesystem lock, takes one job from shared atomic state,
checks that its GPU is otherwise idle, and runs an immutable source snapshot.
No process termination, dataset downloads or model uploads are performed.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT/'configs/studies/cifar100_seven_arms_20261118.json'
DEFAULT_OUTPUT = ROOT/'outputs/seven_arms_20261118'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temporary, path)


@contextlib.contextmanager
def state_lock(output):
    with (output/'state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def source_files():
    return ([ROOT/'train.py', ROOT/'tools/run_pair_sweep.py', ROOT/'requirements.txt'] +
            sorted((ROOT/'acd').rglob('*.py')) + [ROOT/'acd/models/registry.json'])


def verify_inputs(output):
    manifest = read(output/'manifest.json')
    for entry in manifest['files']:
        path = Path(entry['path'])
        if not path.is_file() or digest(path) != entry['sha256']:
            raise ValueError(f'Frozen input changed or missing: {path}')
    return manifest


def verify_teacher(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Teacher digest differs from release manifest: {path}')


def prepare(output, protocol_path, data_root):
    if (output/'manifest.json').exists():
        verify_inputs(output)
        raise ValueError('Study already prepared; use check/worker/status to reuse it')
    protocol = read(protocol_path)
    output.mkdir(parents=True, exist_ok=True)
    write(output/'protocol.json', protocol)
    data_root = data_root.resolve()
    files = []
    for path in source_files():
        destination = output/'source'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        files.append({'path':str(destination),'sha256':digest(destination),'role':'source'})
    files.append({'path':str(output/'protocol.json'),'sha256':digest(output/'protocol.json'),'role':'protocol'})
    for name in ('train', 'test', 'meta'):
        path = data_root/'cifar-100-python'/name
        files.append({'path':str(path),'sha256':digest(path),'role':'dataset'})
    release_paths = [ROOT/'outputs/release-cifar100-20261116', ROOT/'outputs/release-head-controls-20261031']
    teachers = {}
    for release in release_paths:
        manifest = read(release/'model_manifest.json')
        for model in manifest['models']:
            if model.get('role') != 'teacher':
                continue
            name = model['model_config']['architecture']
            source = release/model['filename']
            verify_teacher(source, model['sha256'])
            destination = output/'teachers'/source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            teachers[name] = str(destination)
            files.append({'path':str(destination),'sha256':digest(destination),'role':'teacher'})
    # The complete split is saved once so future runs need not infer its identity.
    from torchvision.datasets import CIFAR100
    from acd.data import stratified_split
    base = CIFAR100(data_root, train=True)
    train, val = stratified_split(base.targets, protocol['selection']['validation_per_class'],
                                 protocol['selection']['split_seed'])
    write(output/'split.json', {'train_indices':train,'validation_indices':val,
                              'split_seed':protocol['selection']['split_seed']})
    files.append({'path':str(output/'split.json'),'sha256':digest(output/'split.json'),'role':'split'})
    pairs, jobs = {}, []
    for pair in protocol['pairs']:
        base_config = read(ROOT/'configs'/f'{pair}.json')
        cfg = {key:base_config[key] for key in ('pair','teacher','student')}
        cfg.update(protocol['shared'])
        cfg['feature_lift'] = protocol['feature_lift'] if cfg['student'] in protocol['low_dimension_students'] else 0
        pairs[pair] = {'base_config':cfg,'teacher_checkpoint':teachers[cfg['teacher']]}
        for candidate, recipe in protocol['candidates'].items():
            tuning = {**cfg, **recipe, **protocol['arms']['kd'],
                      'seed':protocol['selection']['seed'],
                      'validation_per_class':protocol['selection']['validation_per_class'],
                      'split_seed':protocol['selection']['split_seed']}
            config_path = output/'configs/tuning'/f'{pair}__{candidate}.json'
            write(config_path, tuning)
            files.append({'path':str(config_path),'sha256':digest(config_path),'role':'tuning_config'})
            jobs.append({'id':f'{pair}__screen__{candidate}', 'pair':pair,'stage':'screen',
                         'candidate':candidate, 'config':str(config_path),
                         'output':str(output/'tuning'/pair/candidate), 'status':'pending',
                         'config_sha256':digest(config_path)})
    write(output/'manifest.json', {'created_at':time.time(),'data_root':str(data_root),
                                  'python':sys.executable,'files':files,'pairs':pairs})
    write(output/'state.json', {'created_at':time.time(),'jobs':jobs,'selections':{},'shortlists':{}})
    print(json.dumps({'prepared':str(output),'screening_runs':len(jobs),
                      'promotions':len(pairs)*protocol['selection']['promote'],
                      'final_runs':len(pairs)*len(protocol['arms'])},indent=2))


def score(run_dir, through_epoch):
    records = [json.loads(line) for line in (Path(run_dir)/'history.jsonl').read_text().splitlines()]
    records = [row for row in records if through_epoch-5 < row['epoch'] <= through_epoch]
    if len(records) != 5 or {row['epoch'] for row in records} != set(range(through_epoch-4, through_epoch+1)):
        raise ValueError(f'Expected five distinct validation epochs through {through_epoch}: {run_dir}')
    import math
    if not all(math.isfinite(row['top1']) and math.isfinite(row['ce']) for row in records):
        raise ValueError('Nonfinite validation ranking metric')
    return {'top1':sum(row['top1'] for row in records)/5,
            'ce':sum(row['ce'] for row in records)/5,'epochs':[row['epoch'] for row in records]}


def advance(output, state, manifest, protocol):
    """Called under state_lock. Decisions use holdout scores, never test scores."""
    for pair in protocol['pairs']:
        screening = [j for j in state['jobs'] if j['pair']==pair and j['stage']=='screen']
        if pair not in state['shortlists'] and all(j['status']=='completed' for j in screening):
            ranked = sorted(screening, key=lambda j:(-j['score']['top1'],j['score']['ce'],j['candidate']))
            selected = ranked[:protocol['selection']['promote']]
            state['shortlists'][pair] = {'at':time.time(),'candidates':[j['candidate'] for j in selected],
                                         'screen_scores':{j['candidate']:j['score'] for j in ranked}}
            for previous in selected:
                state['jobs'].append({k:v for k,v in previous.items() if k in
                                      ('pair','candidate','config','config_sha256','output')} |
                                     {'id':f"{pair}__refine__{previous['candidate']}",
                                      'stage':'refine','status':'pending'})
        refined = [j for j in state['jobs'] if j['pair']==pair and j['stage']=='refine']
        if pair not in state['selections'] and len(refined)==protocol['selection']['promote'] and all(j['status']=='completed' for j in refined):
            winner = min(refined, key=lambda j:(-j['score']['top1'],j['score']['ce'],j['candidate']))
            recipe = protocol['candidates'][winner['candidate']]
            common = {**manifest['pairs'][pair]['base_config'], **recipe,
                      'seed':protocol['final_seed'],'validation_per_class':0}
            common_digest = hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest()
            selection = {'pair':pair,'at':time.time(),'candidate':winner['candidate'],
                         'criterion':protocol['selection']['ranking'],
                         'validation_scores':{j['candidate']:j['score'] for j in refined},
                         'common_config':common,'common_config_sha256':common_digest,
                         'scope':'best_of_two_full_length_candidates_from_four_screened',
                         'teacher_checkpoint':manifest['pairs'][pair]['teacher_checkpoint']}
            state['selections'][pair] = selection
            write(output/'selected'/f'{pair}.json',selection)
            for arm in protocol['final_order']:
                cfg = {**common, **protocol['arms'][arm]}
                config_path = output/'configs/final'/f'{pair}__{arm}.json'
                write(config_path,cfg)
                state['jobs'].append({'id':f'{pair}__final__{arm}','pair':pair,'stage':'final','arm':arm,
                                      'config':str(config_path),'config_sha256':digest(config_path),
                                      'output':str(output/'final'/pair/arm),'status':'pending'})


def process_exists(pid):
    if not pid:
        return False
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def gpu_occupants(gpu):
    result = subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,check=True,timeout=45)
    return result.stdout.strip()


def result_for(job, protocol):
    path = Path(job['output'])/('stage.json' if job['stage']=='screen' else 'final.json')
    result = read(path)
    epoch = protocol['selection']['screen_epochs'] if job['stage']=='screen' else protocol['shared']['epochs']
    if result['epoch'] != epoch or result['status'] != ('stage_completed' if job['stage']=='screen' else 'completed'):
        raise ValueError(f'Unexpected completion marker for {job["id"]}')
    if result['checkpoint_sha256'] != digest(Path(job['output'])/'latest.pth'):
        raise ValueError(f'Checkpoint hash mismatch for {job["id"]}')
    expected_split = 'official_test' if job['stage']=='final' else 'training_holdout'
    expected_samples = 10000 if job['stage']=='final' else protocol['selection']['validation_per_class']*100
    if result['evaluation_split'] != expected_split or result['validation']['samples'] != expected_samples:
        raise ValueError(f'Wrong evaluation split for {job["id"]}')
    return result


def worker(output,gpu):
    manifest = verify_inputs(output)
    protocol = read(output/'protocol.json')
    preflight = read(output/'preflight.json')
    if preflight['status']!='passed' or preflight['manifest_sha256']!=digest(output/'manifest.json'):
        raise ValueError('Matching GPU preflight required')
    if gpu not in protocol['gpu_allowlist']:
        raise ValueError('GPU is outside study allowlist')
    # Lock outside the study also coordinates separate workers using this tool.
    with (output.parent/f'acd-gpu-{gpu}.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if (output/'PAUSE').exists():
                write(output/f'worker-{gpu}.json', {'status':'paused_between_jobs','pid':os.getpid(),'at':time.time()})
                time.sleep(20)
                continue
            occupied = gpu_occupants(gpu)
            free = shutil.disk_usage(output).free/(1024**3)
            if occupied or free < protocol['minimum_free_disk_gib']:
                write(output/f'worker-{gpu}.json', {'status':'waiting_for_idle_gpu' if occupied else 'waiting_for_disk',
                      'occupants':occupied,'free_disk_gib':free,'pid':os.getpid(),'at':time.time()})
                time.sleep(20)
                continue
            with state_lock(output):
                state = read(output/'state.json')
                for stale in state['jobs']:
                    if stale['status']=='running' and not process_exists(stale.get('worker_pid')) and not process_exists(stale.get('child_pid')):
                        # Recover a completed marker, or resume an interrupted process.
                        marker = Path(stale['output'])/('stage.json' if stale['stage']=='screen' else 'final.json')
                        if marker.exists():
                            result = result_for(stale,protocol)
                            stale.update(status='completed',result=result)
                            if stale['stage']!='final': stale['score']=score(stale['output'],result['epoch'])
                        else:
                            stale.update(status='pending',recovered_after_interruption_at=time.time())
                advance(output,state,manifest,protocol)
                pending = [j for j in state['jobs'] if j['status']=='pending']
                if not pending:
                    waiting = any(j['status']=='running' for j in state['jobs'])
                    write(output/'state.json',state)
                    write(output/f'worker-{gpu}.json',{'status':'waiting_for_other_worker' if waiting else 'finished',
                           'pid':os.getpid(),'at':time.time()})
                    if not waiting: return
                    job = None
                else:
                    order = {'refine':0,'final':1,'screen':2}
                    job = min(pending,key=lambda j:(order[j['stage']],protocol['pairs'].index(j['pair']),
                              protocol['final_order'].index(j['arm']) if j['stage']=='final' else list(protocol['candidates']).index(j['candidate'])))
                    job.update(status='running',gpu=gpu,worker_pid=os.getpid(),started_at=time.time())
                    write(output/'state.json',state)
            if job is None:
                time.sleep(15)
                continue
            write(output/f'worker-{gpu}.json',{'status':'running','job':job['id'],'pid':os.getpid(),'at':time.time()})
            print(json.dumps({'event':'starting','gpu':gpu,'job':job['id']}),flush=True)
            try:
                if digest(job['config'])!=job['config_sha256']:
                    raise ValueError('Job configuration changed after registration')
                run_dir=Path(job['output']);run_dir.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-B','-u',str(output/'source/train.py'),
                         '--config',job['config'],'--data-root',manifest['data_root'],
                         '--teacher-checkpoint',manifest['pairs'][job['pair']]['teacher_checkpoint'],
                         '--output',str(run_dir),'--device','cuda:0']
                if job['stage']=='screen': command+=['--stop-after-epoch',str(protocol['selection']['screen_epochs'])]
                if (run_dir/'latest.pth').exists(): command+=['--resume',str(run_dir/'latest.pth')]
                environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                             'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                with (run_dir/'console.log').open('a') as console:
                    child=subprocess.Popen(command,cwd=output/'source',env=environment,stdout=console,stderr=subprocess.STDOUT)
                    with state_lock(output):
                        state=read(output/'state.json')
                        current=next(j for j in state['jobs'] if j['id']==job['id'])
                        current['child_pid']=child.pid
                        write(output/'state.json',state)
                    while child.poll() is None:
                        write(output/f'worker-{gpu}.json',{'status':'running','job':job['id'],'pid':os.getpid(),
                              'child_pid':child.pid,'at':time.time()})
                        time.sleep(15)
                    if child.returncode:
                        raise RuntimeError(f'Training exited {child.returncode}; see {run_dir / "console.log"}')
                result=result_for(job,protocol)
                update={'status':'completed','result':result,'finished_at':time.time()}
                if job['stage']!='final': update['score']=score(run_dir,result['epoch'])
            except Exception as error:
                update={'status':'failed','error':repr(error),'finished_at':time.time()}
            with state_lock(output):
                state=read(output/'state.json')
                current=next(j for j in state['jobs'] if j['id']==job['id'])
                current.update(update)
                advance(output,state,manifest,protocol)
                write(output/'state.json',state)
            print(json.dumps({'event':'finished','job':job['id'],**update}),flush=True)


def preflight(output,device):
    manifest=verify_inputs(output)
    protocol=read(output/'protocol.json')
    import torch
    from acd.models import build,logits
    from acd.heads import RotatingETFHead,optimizer_groups
    from acd.checkpoints import load_model
    from acd.losses import distillation_loss
    from acd.data import datasets_for
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rows=[]
    for pair,item in manifest['pairs'].items():
        cfg={**item['base_config'],**protocol['candidates']['lr005_ce05'],
             'validation_per_class':50,'split_seed':protocol['selection']['split_seed']}
        train,_=datasets_for(cfg,manifest['data_root'])
        batch=[train[i] for i in range(8)]
        x=torch.stack([sample[0] for sample in batch]).to(device)
        y=torch.tensor([sample[1] for sample in batch],device=device)
        teacher,_=build(cfg['teacher']);load_model(item['teacher_checkpoint'],teacher,config=cfg|{'method':'kd'},role='teacher')
        teacher.to(device).eval().requires_grad_(False)
        with torch.no_grad(): target=logits(teacher,x)
        del teacher,train,batch
        for arm,fields in protocol['arms'].items():
            torch.manual_seed(protocol['final_seed'])
            current=cfg|fields
            model,backbone=build(cfg['student'],angular=current['method']!='kd',etf=current['etf'],
                                 rotating_etf=current['rotating_etf'],feature_lift=cfg['feature_lift'])
            model.to(device).train()
            optimizer=torch.optim.SGD(optimizer_groups(model),lr=cfg['lr'],momentum=.9,weight_decay=.0005)
            loss,diag=distillation_loss(logits(model,x),target,y,backbone,current)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss: {pair}/{arm}')
            loss.backward()
            heads=[m for m in model.modules() if isinstance(m,RotatingETFHead)]
            for head in heads: head.project_gradient()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step()
            for head in heads: head.retract(optimizer)
            row={'pair':pair,'arm':arm,'loss':float(loss.detach()),'ratio':float(diag['r']),
                 'etf_error':max([h.geometry_error() for h in heads],default=0)}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del model,optimizer,backbone,heads,loss,diag
    write(output/'preflight.json',{'status':'passed','manifest_sha256':digest(output/'manifest.json'),
          'device':device,'torch':str(torch.__version__),'cuda':torch.version.cuda,'at':time.time(),'checks':rows})


def status(output):
    from collections import Counter
    state=read(output/'state.json')
    print(json.dumps({'counts':dict(Counter(j['status'] for j in state['jobs'])),
                      'stages':dict(Counter(j['stage'] for j in state['jobs'])),
                      'selected_pairs':list(state['selections']),
                      'running':[j for j in state['jobs'] if j['status']=='running'],
                      'failed':[j for j in state['jobs'] if j['status']=='failed'],
                      'workers':[read(p) for p in sorted(output.glob('worker-*.json'))]},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','check','preflight','worker','status'])
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    p.add_argument('--data-root',type=Path,default=ROOT.parent/'KD/reproductions/cifar100_acd_rerun_20261116/data')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();output=args.output.resolve()
    if args.action=='prepare': prepare(output,args.protocol,args.data_root)
    elif args.action=='check':
        manifest=verify_inputs(output)
        print(json.dumps({'status':'passed','verified_files':len(manifest['files']),'pairs':len(manifest['pairs'])}))
    elif args.action=='preflight': preflight(output,args.device)
    elif args.action=='worker': worker(output,args.gpu)
    else: status(output)


if __name__=='__main__':
    main()
