"""L_ACD = CE + lambda * r * SICD, with optional legacy CE weighting."""
import torch
from torch.nn import functional as F

def standardize(z, eps=1e-6):
    z = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
    return (z-z.mean(1, keepdim=True))/z.std(1, keepdim=True, correction=1).clamp_min(eps)

def full_kl(z, t, tau):
    return tau**2 * F.kl_div((z/tau).log_softmax(1), (t.detach()/tau).softmax(1), reduction='batchmean')

def conditional_kl(z, t, y, tau):
    t = t.detach()
    mask = torch.ones_like(z, dtype=torch.bool).scatter_(1, y[:, None], False)
    q = (z[mask].reshape(len(y), -1)/tau).log_softmax(1)
    p = (t[mask].reshape(len(y), -1)/tau).softmax(1)
    mass = (t/tau).softmax(1).masked_fill(~mask, 0).sum(1)
    return tau**2 * (mass*F.kl_div(q, p, reduction='none').sum(1)).mean()

def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=False)
    return torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt()

def distillation_loss(z, teacher, labels, backbone, config):
    tau = config.get('temperature', 4.)
    method = config.get('method', 'acd')
    if tau <= 0 or z.shape != teacher.shape or z.shape[1] < 2:
        raise ValueError('Positive temperature and matching class dimensions >=2 required')
    ce = F.cross_entropy(z, labels)
    ratio = z.new_tensor(1.)
    if method == 'ce':
        kd = z.new_tensor(0.)
    elif method in ('kd', 'angular_kd'):
        kd = full_kl(z, teacher, tau)
    elif method in ('standardized_full', 'acd', 'conditional'):
        zs = standardize(z, config.get('standardization_eps', 1e-6))
        ts = standardize(teacher.detach(), config.get('standardization_eps', 1e-6))
        kd = full_kl(zs, ts, tau) if method == 'standardized_full' else conditional_kl(zs, ts, labels, tau)
        if method == 'acd':
            parameters = tuple(backbone)
            if not parameters:
                raise ValueError('ACD gradient matching requires nonempty backbone parameters')
            full = full_kl(zs, ts, tau)
            ratio = (gradient_norm(full, parameters)/gradient_norm(kd, parameters).clamp_min(config.get('gradient_eps', 1e-12))).detach()
    else:
        raise ValueError(f'Unknown method: {method}')
    loss = config.get('ce_weight', 1.)*ce + config.get('kd_weight', 1.)*ratio*kd
    return loss, {'ce': ce.detach(), 'sicd_or_kd': kd.detach(), 'r': ratio.detach()}
