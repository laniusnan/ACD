"""Verified ImageNet endpoints; baseline preview, not a five-arm ablation."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / 'reproductions/imagenet1k_acd_20261113'
DATA = Path('/fastdata/users/iaii2026/datasets/imagenet1k/val')

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def cam(net, x, target):
    captured = []
    # Last convolutional block, before global spatial pooling.
    hook = net.features[18].register_forward_hook(lambda m, a, out: captured.append(out))
    try:
        logits = net(x.requires_grad_(True))
        a = captured[0]
        assert tuple(a.shape) == (1, 1280, 7, 7), a.shape
        g = torch.autograd.grad(logits[0, target], a)[0]
        denominator = 2*g.square() + a.sum((2, 3), keepdim=True)*g.pow(3)
        alpha = torch.where(g != 0, g.square()/(denominator+1e-7), torch.zeros_like(g))
        w = (alpha*g.relu()).sum((2, 3), keepdim=True)
        native = (w*a).sum(1, keepdim=True).relu()
        assert torch.isfinite(native).all() and native.max() > 0
        heat = F.interpolate(native, (224, 224), mode='bilinear', align_corners=False)[0, 0]
        heat = (heat-heat.min())/(heat.max()-heat.min()).clamp_min(1e-12)
        return heat.detach().numpy(), native.detach().numpy(), int(logits.argmax(1))
    finally:
        hook.remove()

def main():
    torch.set_num_threads(2)
    entries, nets = [], []
    for arm in ['I0', 'I2']:
        directory = STUDY / 'runs' / f'{arm}_seed3101_baseline_20261114'
        path = directory / 'latest.pth'
        final = json.loads((directory/'final.json').read_text())
        sha = digest(path)
        assert sha == final['reloaded_checkpoint_sha256']
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        assert ckpt['epoch'] == 100
        net = models.mobilenet_v2(weights=None).eval().requires_grad_(False)
        net.load_state_dict(ckpt['student'], strict=True)
        nets.append(net)
        entries.append({'arm': arm, 'checkpoint': str(path.relative_to(ROOT)),
                        'sha256': sha, 'epoch': 100, 'validation': final['validation']})
    index = json.loads((STUDY/'artifacts/imagenet_class_index.json').read_text())
    names = [index[str(i)][0] for i in range(1000)]
    prep = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(224)])
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (.485, .456, .406), (.229, .224, .225))])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    rows, arrays = [], {}
    # Fixed first filename in bird and dog classes, no prediction-based selection.
    for row, wnid in enumerate(['n01514668', 'n02099601']):
        path = sorted((DATA/wnid).glob('*.JPEG'))[0]
        label = names.index(wnid)
        image = prep(Image.open(path).convert('RGB'))
        rgb = np.asarray(image)/255.
        axes[row, 0].imshow(rgb)
        item = {'image': str(path), 'image_sha256': digest(path), 'target': label,
                'class': index[str(label)][1], 'predictions': []}
        for col, net in enumerate(nets, 1):
            heat, native, prediction = cam(net, normalize(image).unsqueeze(0), label)
            axes[row, col].imshow(np.clip(.55*rgb+.45*plt.get_cmap('jet')(heat)[..., :3], 0, 1))
            item['predictions'].append(prediction)
            arrays[f'row{row}_method{col}_native'] = native
            arrays[f'row{row}_method{col}_heat'] = heat
        rows.append(item)
    for col, name in enumerate(['Input', 'SKD', 'Ours']):
        axes[0, col].set_title(name, fontsize=12, fontweight='bold', pad=6)
    for ax in axes.flat:
        ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=.94, wspace=.025, hspace=.025)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT/f'fig_imagenet_baseline_gradcampp.{ext}', dpi=300, bbox_inches='tight', pad_inches=.015)
    np.savez_compressed(OUT/'fig_imagenet_baseline_gradcampp.npz', **arrays)
    metadata = {'status': 'layout_placeholder_not_skd_or_acd_results', 'models': entries,
                'display_labels': ['Input', 'SKD', 'Ours'],
                'actual_methods': ['Input', 'CE baseline', 'Ordinary KD'],
                'caption_disclosure_required': True,
                'images': rows, 'target_layer': 'features.18 (last conv block before global pooling)',
                'activation_shape': [1, 1280, 7, 7], 'target': 'same ground-truth class for all methods',
                'cam': 'Grad-CAM++ conventional gradient-power implementation',
                'normalization': 'per-image per-method min-max', 'palette': 'jet',
                'overlay_alpha': .45, 'native_resolution': [7, 7],
                'limitation': 'No complete ImageNet cosine/ETF/full-standardized/SICD ablation endpoints found.'}
    (OUT/'fig_imagenet_baseline_gradcampp.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (OUT/'overleaf.tex').write_text(r'''% Layout placeholder: retain the provenance disclosure in the caption.
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{fig_imagenet_baseline_gradcampp.pdf}
    \caption{\textbf{Layout preview for an SKD--ACD comparison.}
    The columns labeled SKD and Ours currently contain placeholder
    Grad-CAM++ maps from CE-trained and ordinary-KD-trained MobileNetV2
    models, respectively, rather than results from SKD or ACD.
    Maps use identical ImageNet inputs and ground-truth target classes
    at the last convolutional block before global pooling.}
    \label{fig:gradcampp_placeholder}
\end{figure}
''')
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
