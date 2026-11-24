import importlib
import json
from pathlib import Path
from torchvision import models as tv
from ..heads import CosineHead, LiftedClassifier, RotatingETFHead

REGISTRY = json.loads(Path(__file__).with_name('registry.json').read_text())

def build(name, dataset='cifar100', angular=False, scale=30., etf=False,
          rotating_etf=False, feature_lift=0):
    if (etf or rotating_etf) and not angular:
        raise ValueError('ETF initialization requires an angular student head')
    if etf and rotating_etf:
        raise ValueError('Select ETF initialization or rotating ETF, not both')
    if not isinstance(feature_lift, int) or feature_lift < 0:
        raise ValueError('feature_lift must be a nonnegative integer')
    if dataset == 'cifar100':
        entry = REGISTRY['models'][name]
        result = getattr(importlib.import_module('.'+entry['module'], __name__), entry['constructor'])(num_classes=100)
        net = result[0] if isinstance(result, tuple) else result
        head_name = entry['head']
    elif dataset == 'imagenet':
        net = getattr(tv, name)(weights=None)
        head_name = 'classifier.1' if name.startswith('mobilenet') else 'fc'
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    head = net.get_submodule(head_name)
    if feature_lift:
        lift = LiftedClassifier(head.in_features, feature_lift, head.out_features,
                                bias=head.bias is not None)
        parent, _, key = head_name.rpartition('.')
        target = net.get_submodule(parent) if parent else net
        setattr(target, key, lift)
        head_name += '.classifier'
        head = lift.classifier
    if angular:
        replacement = (RotatingETFHead(head.weight, scale) if rotating_etf
                       else CosineHead(head.weight, scale, etf))
        parent, _, key = head_name.rpartition('.')
        target = net.get_submodule(parent) if parent else net
        setattr(target, key, replacement)
    backbone = [p for n, p in net.named_parameters() if not n.startswith(head_name+'.') and p.requires_grad]
    return net, backbone

def logits(net, x):
    out = net(x)
    return out[0] if isinstance(out, tuple) else out
