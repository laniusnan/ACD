"""Export the seven archived CIFAR-100 reruns with verified source identities."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from acd.checkpoints import sha256
from tools.export_weights import export_weights

ROOT = Path(__file__).resolve().parents[1]
TAG = 'cifar100-rerun-20261116'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = json.loads((ROOT / 'results/cifar100_rerun.json').read_text())
    records = results['results'] if 'results' in results else results['runs']
    assets = json.loads((args.archive_root / 'artifacts/assets.json').read_text())
    manifest = {'format': 'acd-release-v1', 'tag': TAG, 'source_repository': 'laniusnan/KD',
                'archive': 'reproductions/cifar100_acd_rerun_20261116',
                'download_base_url': f'https://github.com/laniusnan/ACD/releases/download/{TAG}',
                'selection': 'final_epoch_240_single_seed_3201_no_etf', 'models': []}
    args.output.mkdir(parents=True, exist_ok=True)
    teachers = {}
    for record in records:
        pair = record['pair']
        config_path = ROOT / 'configs' / f'{pair}.json'
        config = json.loads(config_path.read_text())
        source = args.archive_root / 'runs' / f'{pair}__s3201__acd' / 'latest.pth'
        if record['status'] != 'completed' or record['epoch'] != 240 or sha256(source) != record['checkpoint_sha256']:
            raise ValueError(f'Unverified student archive: {pair}')
        entry = export_weights(source, args.output / f'acd_{pair}_s3201_e240.pth', config,
                               metadata={'pair': pair, 'teacher': config['teacher'], 'seed': 3201,
                                         'epoch': 240, 'etf': False, 'archived_test': record['test'],
                                         'config_file': f'{pair}.json'})
        manifest['models'].append(entry)
        shutil.copyfile(config_path, args.output / config_path.name)
        teachers[config['teacher']] = config
    for name, config in sorted(teachers.items()):
        source = args.archive_root / 'assets/teachers' / f'{name}.pth'
        if sha256(source) != assets['teachers'][name]['sha256']:
            raise ValueError(f'Unverified teacher archive: {name}')
        entry = export_weights(source, args.output / f'teacher_{name}_cifar100.pth', config, 'teacher',
                               metadata={'provenance': assets['teachers'][name]})
        manifest['models'].append(entry)
    (args.output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'models': len(manifest['models']), 'bytes': sum(item['size_bytes'] for item in manifest['models'])}))


if __name__ == '__main__':
    main()
