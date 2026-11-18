import importlib
import json
from pathlib import Path
from torchvision import models as tv
from ..heads import CosineHead

REGISTRY = json.loads(Path(__file__).with_name('registry.json').read_text())

def build(name, dataset='cifar100', angular=False, scale=30., etf=False):
    if dataset == 'cifar100':
        entry = REGISTRY['models'][name]
        result = getattr(importlib.import_module('.'+entry['module'], __name__), entry['constructor'])(num_classes=100)
        net = result[0] if isinstance(result, tuple) else result
        head_name = entry['head']
    else:
        net = getattr(tv, name)(weights=None)
        head_name = 'classifier.1' if name.startswith('mobilenet') else 'fc'
    head = net.get_submodule(head_name)
    if angular:
        replacement = CosineHead(head.weight, scale, etf)
        parent, _, key = head_name.rpartition('.')
        target = net.get_submodule(parent) if parent else net
        setattr(target, key, replacement)
    backbone = [p for n, p in net.named_parameters() if not n.startswith(head_name+'.') and p.requires_grad]
    return net, backbone

def logits(net, x):
    out = net(x)
    return out[0] if isinstance(out, tuple) else out
