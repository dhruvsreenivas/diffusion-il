import torch
from torch import autograd


def compute_gradient_penalty(discriminator, expert_obs, expert_actions, obs, actions):
    """Compute gradient penalty."""

    alpha = torch.rand(expert_actions.size(0), 1)

    # interpolate for observations.
    mixup_obs = dict()
    for k, expert_v in expert_obs.items():
        policy_v = obs[k]

        alpha_v = alpha.expand_as(expert_v).to(expert_v.device)
        mixup_obs_k = alpha_v * expert_v + (1 - alpha_v) * policy_v
        mixup_obs_k.requires_grad = True

        mixup_obs[k] = mixup_obs_k

    # interpolate for actions.
    alpha_actions = alpha.expand_as(expert_actions).to(expert_actions.device)
    mixup_actions = alpha_actions * expert_actions + (1 - alpha_actions) * actions
    mixup_actions.requires_grad = True

    disc = discriminator(mixup_obs, mixup_actions)
    ones = torch.ones(disc.size()).to(disc.device)
    grad = autograd.grad(
        outputs=disc,
        inputs=(mixup_obs, mixup_actions),
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_pen = (grad.norm(2, dim=1) - 1).pow(2).sum()
    return grad_pen
