"""Regenerate the README's workflow, recorded result plot and training curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'res110_res32': 'ResNet110 → ResNet32', 'res32x4_sv1': 'ResNet32×4 → ShuffleNetV1',
          'res32x4_sv2': 'ResNet32×4 → ShuffleNetV2', 'res50_mv1': 'ResNet50 → MobileNetV1',
          'res56_res20': 'ResNet56 → ResNet20', 'wrn402_wrn162': 'WRN40-2 → WRN16-2',
          'wrn402_wrn401': 'WRN40-2 → WRN40-1'}
COLORS = ['#3366a3', '#ce7737', '#3c8c75', '#ae5675', '#8465ad', '#6c863e', '#687c8b']


def save(figure, name):
    for suffix in ('png', 'svg', 'pdf'):
        figure.savefig(ROOT / 'figs' / f'{name}.{suffix}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    figure, axis = plt.subplots(figsize=(12, 2.5))
    axis.set(xlim=(0, 12), ylim=(0, 2.5)); axis.axis('off')
    boxes = [(0.05, '1  Download', 'Pair + teacher weights\nSHA256 verified'),
             (3.10, '2  Train', 'Config + frozen teacher\nlatest.pth + history.jsonl'),
             (6.15, '3  Evaluate', 'Student or teacher\nTop-1 / Top-5 / CE'),
             (9.20, '4  Share', 'Export model weights\nModel guide + manifest')]
    for index, (left, title, detail) in enumerate(boxes):
        axis.add_patch(FancyBboxPatch((left, .45), 2.65, 1.65, boxstyle='round,pad=.035,rounding_size=.10',
                                     facecolor='#f3f7fb', edgecolor=COLORS[0], linewidth=1.3))
        axis.text(left+.16, 1.62, title, weight='bold', fontsize=13, color='#243c58')
        axis.text(left+.16, 1.0, detail, fontsize=10.5, linespacing=1.7, va='center')
        if index < 3:
            axis.annotate('', xy=(left+3., 1.3), xytext=(left+2.72, 1.3),
                          arrowprops={'arrowstyle': '->', 'color': '#687c8b', 'lw': 1.5})
    save(figure, 'workflow')
    results = json.loads((ROOT / 'results/cifar100_rerun.json').read_text())['results']
    figure, axis = plt.subplots(figsize=(9.5, 4.2))
    values = [row['test']['top1'] for row in results]
    bars = axis.barh([LABELS[row['pair']] for row in results], values, color=COLORS, height=.65)
    axis.invert_yaxis(); axis.set_xlim(0, 88)
    axis.bar_label(bars, labels=[f'{value:.2f}%' for value in values], padding=5, fontsize=10)
    axis.set_xlabel('CIFAR-100 test Top-1 accuracy (%)')
    axis.set_title('Recorded final checkpoints · seed 3201 · 240 epochs · no ETF', fontsize=12, pad=14)
    axis.set_axisbelow(True);axis.grid(axis='x', color='#e5e9ef')
    save(figure, 'cifar100_results')
    curves = json.loads((ROOT / 'results/training_curves.json').read_text())['runs']
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for run, color in zip(curves, COLORS):
        history = run['history']; epochs = [row['epoch'] for row in history]
        for axis, key in zip(axes, ['loss', 'train_top1']):
            axis.plot(epochs, [row[key] for row in history], color=color, lw=1.4, label=LABELS[run['pair']])
    for axis, label in zip(axes, ['Training objective', 'Training Top-1 accuracy (%)']):
        axis.set(xlabel='Epoch', ylabel=label);axis.grid(alpha=.2)
    figure.suptitle('Training on 100,000 augmented views per epoch · seed 3201', fontsize=12)
    figure.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=9, frameon=False)
    figure.tight_layout(rect=(0, .18, 1, .95))
    save(figure, 'training_curves')


if __name__ == '__main__':
    main()
