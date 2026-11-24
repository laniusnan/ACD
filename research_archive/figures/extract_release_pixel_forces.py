"""CPU, fixed-checkpoint pixel-force probe from verified release weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / 'mechanism_figures'
MODELS = OUT / 'recovered_force_models'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name):
    path = ROOT / 'CIFAR-100' / 'Models' / (name + '.py')
    spec = importlib.util.spec_from_file_location('force_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2051)
    assert digest(MODELS / 'student.pth') == 'b77595011d4f175664ab3afd45ce5f8098a246d7435728c24294fffdd6fd3963'
    assert digest(MODELS / 'teacher.pth') == '91afc5b40a57dd293778a98502fe6a1132ce033ac284ff89382d733b322d8046'
    checkpoint = torch.load(MODELS / 'student.pth', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 240 and checkpoint['job']['head'] == 'linear'
    mobile, mobile_path = load_module('mobilenetv1')
    vgg, vgg_path = load_module('vgg')
    for path in (mobile_path, vgg_path):
        assert digest(path) == checkpoint['identity']['sources']['frozen/models/' + path.name]
    student = mobile.mobilenetv1()[0].eval()
    teacher = vgg.vgg13(num_classes=100)[0].eval()
    student.load_state_dict(checkpoint['student'], strict=True)
    teacher.load_state_dict(torch.load(MODELS / 'teacher.pth', map_location='cpu', weights_only=True), strict=True)
    for model in (student, teacher):
        model.requires_grad_(False)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    ds = datasets.CIFAR100(ROOT / 'reproductions/cifar100_acd_rerun_20261116/data',
                          train=False, download=False,
                          transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))]))
    examples = []
    for index in (0, 1, 2, 3):
        x0 = ds[index][0].unsqueeze(0)
        with torch.no_grad():
            target = teacher(x0)[0]
        maps = {}
        for name in ('angular','scale_norm','raw'):
            x = x0.clone().requires_grad_(True)
            raw, features = student(x)
            h, w, b = features['pooled_feat'], student.fc.weight, student.fc.bias
            angular = F.normalize(h, dim=1) @ F.normalize(w, dim=1).T
            scaled = h.norm(dim=1,keepdim=True)*w.norm(dim=1)[None,:]*angular
            assert torch.allclose(scaled+b, raw, atol=2e-5, rtol=2e-5)
            logits = {'angular':angular, 'scale_norm':scaled, 'raw':raw}[name]
            loss = 16*F.kl_div((logits/4).log_softmax(1),(target/4).softmax(1),reduction='batchmean')
            force = -torch.autograd.grad(loss,x)[0][0]
            assert torch.isfinite(force).all() and force.norm() > 0
            maps[name] = force.tolist()
        examples.append({'index':index, 'class_name':ds.classes[ds.targets[index]],
                         'rgb':ds.data[index].tolist(), 'force_maps':maps})
    assert all(torch.equal(before[k], v) for k,v in student.state_dict().items())
    result = {'status':'completed', 'scope':'CPU local pixel-gradient probe; no training or optimizer updates',
              'setup':{'pair':'VGG13→MobileNetV1','student_arm':'linear KD','seed':2051,'epoch':240,
                       'selection_rule':'First four official CIFAR-100 test images, indices 0,1,2,3; fixed before computing forces',
                       'temperature':4,'angular_scale':1,
                       'force':'Negative KD gradient with respect to normalized input pixels',
                       'release':'model-backup-20261027'},
              'examples':examples,
              'sources_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
                               (MODELS/'student.pth',MODELS/'teacher.pth',mobile_path,vgg_path)},
              'limitations':['Post-hoc score substitutions on one affine-trained checkpoint, not trained ACI endpoints.',
                             'Angular reference uses scale 1, retaining the original visualization definition; magnitude and parameterization interventions are not isolated.',
                             'Colors show per-pixel RGB-gradient 1-cosine relative to angular; not semantic attention or a measure of accuracy.',
                             'No claims of downstream improvement or training-time conflict frequency.']}
    (OUT/'same_input_pixel_force_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Computed four samples; weight/source hashes, affine reconstruction and unchanged model state verified.')


if __name__ == '__main__':
    main()
