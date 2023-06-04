import torch
import torch.nn.functional as F


def interpolate_mask(mask, shape, allow_scale_mask=False, mask_scale_mode='nearest'):
    assert mask is not None
    assert allow_scale_mask or shape == mask.shape[-2:]
    if shape != mask.shape[-2:] and allow_scale_mask:
        if mask_scale_mode == 'maxpool':
            mask = F.adaptive_max_pool2d(mask, shape)
        else:
            mask = F.interpolate(mask, size=shape, mode=mask_scale_mode)
    return mask


def generator_loss(discr_fake_pred: torch.Tensor, mask=None, args=None):
    fake_loss = F.softplus(-discr_fake_pred)
    # == if masked region should be treated differently
    if (args['mask_as_fake_target'] and args['extra_mask_weight_for_gen'] > 0) or not args['use_unmasked_for_gen']:
        mask = interpolate_mask(mask, discr_fake_pred.shape[-2:], args['allow_scale_mask'], args['mask_scale_mode'])
        if not args['use_unmasked_for_gen']:
            fake_loss = fake_loss * mask
        else:
            pixel_weights = 1 + mask * args['extra_mask_weight_for_gen']
            fake_loss = fake_loss * pixel_weights

    return fake_loss.mean() * args['weight']


def feature_matching_loss(fake_features, target_features, mask=None):
    if mask is None:
        res = torch.stack([F.mse_loss(fake_feat, target_feat)
                           for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
    else:
        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, target_features):
            cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
            error_weights = 1 - cur_mask
            cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
    return res


def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty


def discriminator_real_loss(real_batch, discr_real_pred, gp_coef, do_GP=True):
    real_loss = F.softplus(-discr_real_pred).mean()
    if do_GP:
        grad_penalty = (make_r1_gp(discr_real_pred, real_batch) * gp_coef).mean()
        # grad_penalty = torch.tensor(0.0)
    else:
        grad_penalty = 0

    return real_loss, grad_penalty

def discriminator_fake_loss(discr_fake_pred: torch.Tensor, mask=None, args=None):

    fake_loss = F.softplus(discr_fake_pred)

    if not args['use_unmasked_for_discr'] or args['mask_as_fake_target']:
        # == if masked region should be treated differently
        mask = interpolate_mask(mask, discr_fake_pred.shape[-2:], args['allow_scale_mask'], args['mask_scale_mode'])
        # use_unmasked_for_discr=False only makes sense for fakes;
        # for reals there is no difference beetween two regions
        fake_loss = fake_loss * mask
        if args['mask_as_fake_target']:
            fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

    sum_discr_loss = fake_loss
    return sum_discr_loss.mean()
