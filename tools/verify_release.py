"""Evaluate every release weight file on the complete official test split."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from acd.checkpoints import load_model, sha256
from acd.data import evaluation_dataset
from acd.models import build
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.release_dir / 'model_manifest.json').read_text())
    dataset = evaluation_dataset({'dataset': 'cifar100'}, args.data_root)
    loader = DataLoader(dataset, batch_size=128, num_workers=0)
    report = {'status': 'in_progress', 'device': args.device, 'torch': str(torch.__version__), 'split': 'official CIFAR-100 test',
              'selection': manifest['selection'], 'models': []}
    for entry in manifest['models']:
        started = time.monotonic()
        path = args.release_dir / entry['filename']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Asset hash mismatch: {path.name}')
        config = entry['model_config']
        model, _ = build(config['architecture'], config['dataset'], config['angular'], config['scale'] or 30.,
                         etf=config.get('etf', False),
                         rotating_etf=config.get('rotating_etf', False),
                         feature_lift=config.get('feature_lift', 0))
        load_model(path, model)
        model.to(args.device)
        metrics = evaluate(model, loader, torch.device(args.device))
        archived = entry.get('archived_test')
        if metrics['samples'] != 10000:
            raise ValueError('Expected exactly 10000 test images')
        if archived and abs(metrics['top1'] - archived['top1']) > .011:
            raise ValueError(f'Accuracy mismatch: {path.name}: {metrics} vs {archived}')
        row = {'filename': path.name, 'sha256': entry['sha256'], **metrics,
               'seconds': round(time.monotonic() - started, 2),
               'archived_top1_match': abs(metrics['top1'] - archived['top1']) < .011 if archived else None}
        report['models'].append(row)
        (args.release_dir / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    report['status'] = 'completed'
    (args.release_dir / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
