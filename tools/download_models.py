"""Download release weights and verify SHA256 before making them available."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(entry, directory, base_url):
    filename = entry['filename']
    if Path(filename).name != filename or filename in ('.', '..'):
        raise ValueError('Manifest contains an invalid asset filename')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.exists():
        if checksum(destination) != entry['sha256']:
            raise ValueError(f'Existing file has a different SHA256: {destination}')
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            asset_base = entry.get('download_base_url', base_url).rstrip('/')
            with urlopen(f'{asset_base}/{filename}', timeout=60) as response:
                shutil.copyfileobj(response, stream)
        if checksum(temporary) != entry['sha256']:
            raise ValueError(f'Download SHA256 mismatch: {filename}')
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def select_models(manifest, *, pair=None, filename=None, student_only=False):
    """Resolve selected variants and their teachers, including reused releases."""
    available = manifest['models'] + manifest.get('teacher_dependencies', [])
    if len({entry['filename'] for entry in available}) != len(available):
        raise ValueError('Manifest contains duplicate model filenames')
    if pair is not None:
        selected = [entry for entry in available
                    if entry['role'] == 'student' and entry.get('pair') == pair]
    elif filename is not None:
        selected = [entry for entry in available if entry['filename'] == filename]
    else:
        selected = list(available)
    if not selected:
        raise ValueError('No matching model; use --list to see available filenames and pairs')
    if student_only:
        selected = [entry for entry in selected if entry['role'] == 'student']
        if not selected:
            raise ValueError('The selection contains no student model')
    else:
        needed = {entry['teacher'] for entry in selected if entry['role'] == 'student'}
        teachers = [entry for entry in available if entry['role'] == 'teacher'
                    and entry['model_config']['architecture'] in needed]
        if needed - {entry['model_config']['architecture'] for entry in teachers}:
            raise ValueError('Manifest is missing a teacher; use --student-only for evaluation')
        selected += [entry for entry in teachers if entry not in selected]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'docs/model_manifest.json')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--pair', help='Download every released variant of a pair and its teacher')
    selection.add_argument('--model', help='Download an exact model filename and its teacher, if applicable')
    selection.add_argument('--all', action='store_true', help='Download all models in the selected manifest')
    selection.add_argument('--list', action='store_true', help='List assets without network access')
    parser.add_argument('--student-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.list:
        for entry in manifest['models'] + manifest.get('teacher_dependencies', []):
            print(f"{entry['role']:7} {entry['filename']:52} {entry['size_bytes']/1e6:6.1f} MB"
                  f"  pair={entry.get('pair', '-')}  release={entry.get('release', manifest['tag'])}")
        return
    try:
        entries = select_models(manifest, pair=args.pair, filename=args.model, student_only=args.student_only)
    except ValueError as error:
        parser.error(str(error))
    if 'configs' in manifest:
        configs = {entry['filename']: entry for entry in manifest['configs']}
        required = {entry['config_file'] for entry in entries if 'config_file' in entry}
        if required - configs.keys():
            parser.error('Manifest is missing a referenced evaluation configuration')
        entries += [configs[name] for name in sorted(required)]
    for entry in entries:
        print(download(entry, args.output, manifest['download_base_url']), flush=True)


if __name__ == '__main__':
    main()
