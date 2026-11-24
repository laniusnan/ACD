"""Export hash-verified historical KD head controls without their training state.

The selection and restoration receipt are produced by an archive audit. Legacy
checkpoints may contain Python RNG objects; their import requires explicit opt-in.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from acd.checkpoints import load_model, model_config, read_checkpoint, sha256
from acd.models import build

ROOT = Path(__file__).resolve().parents[1]
TAG = 'cifar100-head-controls-20261031'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def student_config(checkpoint, record):
    original = checkpoint['config']
    training = original['training']
    pair = original['pairs'][record['pair']]
    cfg = json.loads((ROOT / 'configs' / f"{record['pair']}.json").read_text())
    cfg.update({key: training[key] for key in ('epochs', 'batch_size', 'workers', 'lr',
                                              'momentum', 'weight_decay', 'ce_weight',
                                              'kd_weight', 'temperature')})
    cfg.update(teacher=pair['teacher'], student=pair['student'], seed=record['seed'],
               method='kd' if record['head'] == 'linear' else 'angular_kd',
               etf=record['head'] == 'cosine_etf', scale=original['heads']['scale'],
               milestones=training['decay_zero_based'], lr_decay=training['decay_rate'],
               augmentation='double_view', evidence='verified_historical_kd_head_control',
               usage='Evaluation and recipe reference; historical training RNG and ETF initialization differ.')
    if training['epochs'] != 240 or training['protocol'] != 'double':
        raise ValueError('Unexpected historical training recipe')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--restoration-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classes-file', type=Path, required=True, help='Official CIFAR-100 fine-label order JSON')
    parser.add_argument('--allow-legacy-checkpoint', action='store_true')
    args = parser.parse_args()
    if not args.allow_legacy_checkpoint:
        parser.error('Import requires --allow-legacy-checkpoint for trusted, hash-verified KD archives')
    torch.set_num_threads(2)
    selections = json.loads(args.selection.read_text())
    receipts = {entry['source_sha256']: entry for entry in json.loads(args.restoration_receipt.read_text())}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'format': 'acd-release-v1', 'tag': TAG, 'source_repository': 'laniusnan/KD',
        'download_base_url': f'https://github.com/laniusnan/ACD/releases/download/{TAG}',
        'selection': 'final_epoch_240_first_seed_per_study_2051_or_2101; ordinary_KD_controls',
        'models': [], 'configs': [], 'teacher_dependencies': [],
    }
    evidence = []
    for selection in selections:
        entry = selection['entry']
        source = Path(selection['local_source'])
        if source.stat().st_size != entry['bytes'] or sha256(source) != entry['sha256']:
            raise ValueError(f'Original checkpoint failed integrity check: {source.name}')
        receipt = receipts[entry['sha256']]
        if (receipt['original_path'] != entry['path'] or receipt['release'] != selection['release']
                or receipt['archive'] != entry['archive'] or receipt['source_bytes'] != entry['bytes']):
            raise ValueError('Restoration receipt does not match the selected source')
        role = selection.get('role', 'student')
        checkpoint = read_checkpoint(source, allow_legacy=role == 'student')
        provenance = {'repository': 'laniusnan/KD', **receipt}
        metadata = {'role': role, 'resumable': False, 'source_sha256': entry['sha256'],
                    'provenance': provenance}
        if role == 'student':
            record = selection['record']
            status_path = Path(selection['status_file'])
            status = json.loads(status_path.read_text())
            status_sha = sha256(status_path)
            if entry.get('status_sha256', status_sha) != status_sha:
                raise ValueError('Status file failed integrity check')
            if (status['status'] != 'completed' or status['epoch'] != 240 or checkpoint['epoch'] != 240
                    or record['checkpoint_sha256'] != entry['sha256']
                    or status['checkpoint_sha256'] != entry['sha256']):
                raise ValueError('Checkpoint is not a verified final completed run')
            for key in ('pair', 'head', 'seed'):
                if status[key] != record[key] or checkpoint['job'][key] != record[key]:
                    raise ValueError(f'Status and checkpoint disagree: {key}')
            metrics = status.get('reloaded_test', status.get('test'))
            if metrics['samples'] != 10000 or abs(metrics['top1'] - record['top1']) > 1e-8:
                raise ValueError('Status and archived score disagree')
            config = student_config(checkpoint, record)
            prefix = {'linear': 'kd', 'cosine': 'ci_kd', 'cosine_etf': 'ci_etf_kd'}[record['head']]
            stem = f"{prefix}_{record['pair']}_s{record['seed']}_e240"
            metadata.update(pair=record['pair'], teacher=config['teacher'], seed=record['seed'],
                            epoch=240, head=record['head'], etf=config['etf'],
                            objective='ordinary_temperature_KL', protocol=record['protocol'],
                            archived_test=metrics, config_file=stem+'.json')
            provenance.update(study=record['study'], status_path=record['source'], status_sha256=status_sha)
            evidence.append({'filename': stem+'.pth', 'provenance': provenance,
                             'archived_test': metrics, 'original_training': checkpoint['config']['training'],
                             'original_heads': checkpoint['config']['heads'], 'initial': checkpoint['initial'],
                             **{key: checkpoint[key] for key in ('registration_sha256', 'split_sha256',
                                 'teacher_state_sha256', 'baseline_checkpoint_sha256') if key in checkpoint}})
            state = checkpoint['student']
        else:
            config = json.loads((ROOT / 'configs/vgg13_vgg8.json').read_text())
            config.update(student=selection['architecture'], teacher=selection['architecture'],
                          method='ce', seed=0, etf=False, pair='vgg13_teacher',
                          evidence='historical_pretrained_teacher', usage='Teacher evaluation only; training recipe unknown.')
            stem = f"teacher_{selection['architecture']}_cifar100"
            metadata.update(config_file=stem+'.json')
            state = checkpoint
        description = model_config(config, role)
        model, _ = build(description['architecture'], description['dataset'], description['angular'],
                         description['scale'] or 30.)
        original_keys = set(state)
        model.load_state_dict(state, strict=True)
        exported_state = model.state_dict()
        for key in original_keys:
            if not torch.equal(state[key], exported_state[key]):
                raise ValueError(f'Import changed tensor {key}')
        metadata['filled_legacy_buffers'] = sorted(set(exported_state) - original_keys)
        payload = {'format': 'acd-weights-v1', 'state_dict': exported_state,
                   'model_config': description, 'metadata': metadata}
        destination = args.output / (stem + '.pth')
        torch.save(payload, destination)
        loaded = load_model(destination, model, config=config, role=role)
        for key, value in exported_state.items():
            if not torch.equal(value, loaded['state_dict'][key]):
                raise ValueError(f'Export changed tensor {key}')
        manifest['models'].append({'filename': destination.name, 'sha256': sha256(destination),
                                   'size_bytes': destination.stat().st_size,
                                   'model_config': description, **metadata})
        config_path = args.output / (stem + '.json')
        write_json(config_path, config)
        manifest['configs'].append({'filename': config_path.name, 'sha256': sha256(config_path),
                                    'size_bytes': config_path.stat().st_size})
        print(json.dumps({'exported': destination.name, 'original_tensors_equal': True}), flush=True)
    original_manifest = json.loads((ROOT / 'docs/model_manifest.json').read_text())
    needed = {entry['teacher'] for entry in manifest['models'] if entry['role'] == 'student'}
    included = {entry['model_config']['architecture'] for entry in manifest['models'] if entry['role'] == 'teacher'}
    for entry in original_manifest['models']:
        if entry['role'] == 'teacher' and entry['model_config']['architecture'] in needed - included:
            manifest['teacher_dependencies'].append({**entry, 'release': original_manifest['tag'],
                'download_base_url': original_manifest['download_base_url']})
    write_json(args.output / 'model_manifest.json', manifest)
    write_json(args.output / 'source_evidence.json', evidence)
    shutil.copyfile(args.restoration_receipt, args.output / 'restoration_receipt.json')
    shutil.copyfile(args.classes_file, args.output / 'cifar100_classes.json')
    print(json.dumps({'models': len(manifest['models']), 'bytes': sum(x['size_bytes'] for x in manifest['models']),
                      'reused_teachers': len(manifest['teacher_dependencies'])}))


if __name__ == '__main__':
    main()
