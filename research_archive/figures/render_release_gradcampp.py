"""Two-column Grad-CAM++ comparison of two release-trained MobileNetV1s."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from extract_release_pixel_forces import digest, load_module, ROOT, OUT, MODELS
import sys
sys.path.insert(0, str(Path(__file__).parent / 'analysis_20261116'))
from render import plot, np, save
from matplotlib.colors import LinearSegmentedColormap


class CosineHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(100, 512))
        self.register_buffer('scale', torch.tensor(30.))

    def forward(self, h):
        return self.scale * F.linear(F.normalize(h,dim=1), F.normalize(self.weight,dim=1))


def model(name, expected):
    path = MODELS / name
    assert digest(path) == expected
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 240 and checkpoint['job']['seed'] == 2051
    module, source = load_module('mobilenetv1')
    assert digest(source) == checkpoint['identity']['sources']['frozen/models/mobilenetv1.py']
    net = module.mobilenetv1()[0]
    if checkpoint['job']['head'] == 'cosine':
        net.fc = CosineHead()
    net.load_state_dict(checkpoint['student'], strict=True)
    return net.eval().requires_grad_(False), checkpoint


def cam(net, x, target):
    # Same pointwise convolution in both models. Its 4x4 map retains more
    # spatial detail than the final 2x2 stage at the native CIFAR resolution.
    captured = []
    hook = net.model[5][3].register_forward_hook(lambda module, args, out: captured.append(out))
    before = {k: v.clone() for k,v in net.state_dict().items()}
    try:
        logits = net(x.clone().requires_grad_(True))[0]
        a = captured[0]
        assert a.shape[-2:] == (4,4)
        g = torch.autograd.grad(logits[0,target], a)[0]
        # Conventional Grad-CAM++ gradient-power implementation.
        g2, g3 = g.square(), g.pow(3)
        denom = 2*g2 + a.sum((2,3),keepdim=True)*g3
        alpha = torch.where(g != 0, g2/(denom+1e-7), torch.zeros_like(g))
        weights = (alpha*g.relu()).sum((2,3),keepdim=True)
        native = (weights*a).sum(1,keepdim=True).relu()
        assert torch.isfinite(native).all()
        up = F.interpolate(native, size=(32,32),mode='bilinear',align_corners=False)[0,0]
        low, high = up.min(), up.max()
        heat = (up-low)/(high-low) if high>low else torch.zeros_like(up)
        assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
        return heat.detach().numpy(), native.detach().flatten().tolist(), int(logits.argmax(1))
    finally:
        hook.remove()


def main():
    torch.set_num_threads(2)
    nets = [model('student.pth','b77595011d4f175664ab3afd45ce5f8098a246d7435728c24294fffdd6fd3963'),
            model('cosine_student.pth','8223fd2c071b9e6043201cedba46cdfe5d3e5142034a69ef617ca205da5d5947')]
    assert nets[0][1]['initial']['backbone'] == nets[1][1]['initial']['backbone']
    ds = datasets.CIFAR100(ROOT/'reproductions/cifar100_acd_rerun_20261116/data',
                          train=False, download=False,
                          transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))]))
    cmap = LinearSegmentedColormap.from_list('muted_cam', ['#344D69','#769EAB','#ADC4AC','#D7C69A','#B77670'])
    fig, axes = plot.subplots(4,2,figsize=(5.0,8.8))
    records = []
    for row,index in enumerate((0,1,2,3)):
        x,y = ds[index]
        rgb = ds.data[index]/255.
        for col,(net,checkpoint) in enumerate(nets):
            heat, native, pred = cam(net,x[None],y)
            overlay = .55*rgb+.45*cmap(heat)[...,:3]
            ax=axes[row,col]
            ax.imshow(overlay,interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for spine in ax.spines.values(): spine.set_visible(False)
            if row==0: ax.set_title(['Ordinary KD','Cosine-head KD'][col],fontsize=12)
            if col==0: ax.set_ylabel(ds.classes[y]+f'\nindex {index}',fontsize=9)
            records.append({'index':index,'class':ds.classes[y],'target_class':y,
                            'method':checkpoint['job']['head'],'predicted_class':ds.classes[pred],
                            'native_cam':native,'normalized_cam':heat.tolist()})
    fig.subplots_adjust(left=.18,right=.97,top=.94,bottom=.12,wspace=.02,hspace=.02)
    bar_ax=fig.add_axes([.28,.065,.50,.012])
    fig.colorbar(plot.cm.ScalarMappable(norm=plot.Normalize(0,1),cmap=cmap),cax=bar_ax,
                 orientation='horizontal',ticks=[0,.5,1])
    bar_ax.tick_params(labelsize=8)
    fig.text(.53,.026,'Grad-CAM++ · within-map normalized activation',ha='center',fontsize=8)
    stem=OUT/'mobilenetv1_kd_cosine_gradcampp'
    for ext in ('png','pdf','svg'):
        fig.savefig(stem.with_suffix('.'+ext),bbox_inches='tight',facecolor='white',dpi=220)
    plot.close(fig)
    stem.with_suffix('.json').write_text(json.dumps({
        'kind':'Grad-CAM++ gradient-power implementation','dataset':'CIFAR-100 official test',
        'selection':'Indices 0,1,2,3 fixed before attribution computation',
        'pair':'VGG13→MobileNetV1','seed':2051,'epoch':240,
        'target':'Same ground-truth class for both methods, differentiated pre-softmax score',
        'target_layer':'model[5][3], pointwise convolution, native 4x4',
        'normalization':'Per-map min-max after bilinear interpolation to 32x32; zero maps remain zero',
        'blend':{'image':.55,'heatmap':.45},
        'student_sha256':[digest(MODELS/n) for n in ('student.pth','cosine_student.pth')],
        'records':records,
        'limitations':['CIFAR-100 experiment, not the ImageNet setting of SKD Figure 3.',
                       'Two separately trained models; cosine-head KD is not ACI+SICD.',
                       'Intermediate-layer 4x4 attribution; upsampling adds no spatial information.',
                       'Per-map normalization supports spatial comparison, not absolute attribution magnitude comparison.',
                       'Qualitative examples do not establish accuracy or localization superiority.']},indent=2)+'\n')
    print('Generated PNG/PDF/SVG and attribution records; verified paired initialization and unchanged model states.')


if __name__=='__main__':
    main()
