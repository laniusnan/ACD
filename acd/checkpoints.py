"""Checkpoint loading shared by training, evaluation and weight export."""
import hashlib
import pickle
from pathlib import Path

import torch


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_config(config, role='student'):
    angular = role == 'student' and config['method'] not in ('ce', 'kd')
    identity = {'architecture': config[role], 'dataset': config['dataset'],
                'angular': angular, 'scale': config.get('scale', 30.) if angular else None}
    # Keep unchanged identities for existing portable releases.
    if role == 'student':
        if config.get('rotating_etf'):
            identity['rotating_etf'] = True
        if config.get('feature_lift', 0):
            identity['feature_lift'] = config['feature_lift']
    return identity


def read_checkpoint(path, allow_legacy=False):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=not allow_legacy)
    except pickle.UnpicklingError as error:
        raise ValueError('Checkpoint contains legacy Python objects. Use --allow-legacy-checkpoint '
                         'only for a trusted historical file; release weights do not need it.') from error
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a state dictionary or a checkpoint dictionary')
    return checkpoint


def load_model(path, model, *, config=None, role='student', allow_legacy=False):
    checkpoint = read_checkpoint(path, allow_legacy)
    saved_config = checkpoint.get('model_config')
    if saved_config is None and 'config' in checkpoint:
        saved_config = model_config(checkpoint['config'])
    if config is not None and saved_config is not None:
        if saved_config != model_config(config, role):
            raise ValueError('Checkpoint model/dataset/head differs from the selected configuration')
    state = checkpoint.get('student', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state, strict=True)
    return checkpoint


def validate_resume(checkpoint, config, teacher_sha256):
    required = {'config', 'optimizer', 'scheduler', 'epoch', 'validation', 'teacher_sha256'}
    if checkpoint.get('status') == 'incomplete_smoke' or not required.issubset(checkpoint):
        raise ValueError('Resume requires an epoch-boundary latest.pth from this trainer. '
                         'Exported weights, archived reruns and smoke checkpoints are for evaluation only.')
    if checkpoint['config'] != config:
        raise ValueError('Resume configuration differs from checkpoint')
    if checkpoint['teacher_sha256'] != teacher_sha256:
        raise ValueError('Resume teacher differs')
