"""Export portable evaluation weights, excluding optimizer and training state."""
import argparse
import json
from pathlib import Path

import torch

from acd.checkpoints import load_model, model_config, sha256
from acd.models import build


def export_weights(source, output, config, role='student', allow_legacy=False, metadata=None):
    description = model_config(config, role)
    model, _ = build(description['architecture'], description['dataset'],
                     description['angular'], description['scale'] or 30.,
                     etf=description.get('etf', False),
                     rotating_etf=description.get('rotating_etf', False),
                     feature_lift=description.get('feature_lift', 0))
    load_model(source, model, config=config, role=role, allow_legacy=allow_legacy)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 'acd-weights-v1', 'state_dict': model.state_dict(),
               'model_config': description,
               'metadata': {**(metadata or {}), 'role': role, 'resumable': False,
                            'source_sha256': sha256(source)}}
    torch.save(payload, output)
    load_model(output, model, config=config, role=role)
    return {'filename': output.name, 'sha256': sha256(output), 'size_bytes': output.stat().st_size,
            'model_config': description, **payload['metadata']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=['student', 'teacher'], default='student')
    parser.add_argument('--allow-legacy-checkpoint', action='store_true')
    args = parser.parse_args()
    result = export_weights(args.checkpoint, args.output, json.loads(args.config.read_text()),
                            args.role, args.allow_legacy_checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
