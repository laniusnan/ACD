"""Plot this trainer's history.jsonl as PNG, SVG and PDF."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/training'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error('History contains no completed epochs')
    epochs = [row['epoch'] for row in rows]
    if epochs != sorted(set(epochs)):
        parser.error('Epochs must be unique and increasing; do not combine separate runs')
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(epochs, [row['loss'] for row in rows], color='#3366a3')
    axes[0].set_ylabel('Training objective')
    for key in ['top1', 'top5']:
        if all(key in row for row in rows):
            axes[1].plot(epochs, [row[key] for row in rows], label=key.replace('top', 'Top-'))
    axes[1].set_ylabel('Evaluation accuracy (%)');axes[1].legend()
    for axis in axes:
        axis.set_xlabel('Epoch');axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg', 'pdf']:
        destination = args.output.with_suffix('.'+extension)
        figure.savefig(destination, dpi=180, bbox_inches='tight')
        print(destination)
    plt.close(figure)


if __name__ == '__main__':
    main()
