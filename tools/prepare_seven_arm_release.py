"""Package one completed seven-arm CIFAR-100 pair as portable release weights.

The script accepts only final epoch-240 records whose checkpoint digest matches
their `final.json`.  It intentionally does not package partially queued pairs.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from acd.checkpoints import sha256
from tools.export_weights import export_weights


ARMS = ('kd', 'cosine_kd', 'cosine_etf_kd', 'rotating_etf_kd',
        'cosine_sicd', 'cosine_etf_sicd', 'rotating_etf_sicd')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--pair', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.study_root
    final_root = root / 'final' / args.pair
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to add files to nonempty {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'format': 'acd-release-v1', 'tag': args.tag,
        'study': 'cifar100_seven_arms_20261118', 'pair': args.pair,
        'download_base_url': f'https://github.com/laniusnan/ACD/releases/download/{args.tag}',
        'selection': 'fixed_pair_recipe_formal_epoch_240_seed_3301', 'models': [],
    }
    teacher_config = None
    teacher_name = None
    for arm in ARMS:
        arm_root = final_root / arm
        config = json.loads((arm_root / 'config.json').read_text())
        final = json.loads((arm_root / 'final.json').read_text())
        source = arm_root / 'latest.pth'
        if final.get('status') != 'completed' or final.get('epoch') != 240:
            raise ValueError(f'{args.pair}/{arm} is not a completed epoch-240 result')
        if sha256(source) != final.get('checkpoint_sha256'):
            raise ValueError(f'{args.pair}/{arm} checkpoint digest does not match final.json')
        if config['pair'] != args.pair or config['dataset'] != 'cifar100':
            raise ValueError(f'Unexpected config identity for {args.pair}/{arm}')
        filename = f'{args.pair}_{arm}_s{config["seed"]}_e{final["epoch"]}.pth'
        entry = export_weights(source, args.output / filename, config,
                               metadata={'pair': args.pair, 'arm': arm,
                                         'seed': config['seed'], 'epoch': final['epoch'],
                                         'archived_test': final['validation'],
                                         'source_checkpoint_sha256': final['checkpoint_sha256'],
                                         'config_file': f'{args.pair}_{arm}.json'})
        manifest['models'].append(entry)
        (args.output / f'{args.pair}_{arm}.json').write_text(json.dumps(config, indent=2) + '\n')
        teacher_config, teacher_name = config, config['teacher']
    teacher_source = root / 'teachers' / f'teacher_{teacher_name}_cifar100.pth'
    if not teacher_source.is_file():
        raise FileNotFoundError(teacher_source)
    entry = export_weights(teacher_source, args.output / f'teacher_{teacher_name}_cifar100.pth',
                           teacher_config, 'teacher',
                           metadata={'pair': args.pair, 'role_detail': 'fixed_study_teacher'})
    manifest['models'].append(entry)
    shutil.copyfile(root / 'selected' / f'{args.pair}.json', args.output / 'selected_recipe.json')
    (args.output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'SHA256SUMS').write_text(''.join(
        f"{item['sha256']}  {item['filename']}\n" for item in manifest['models']))
    print(json.dumps({'models': len(manifest['models']),
                      'bytes': sum(item['size_bytes'] for item in manifest['models'])}, indent=2))


if __name__ == '__main__':
    main()
