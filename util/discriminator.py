import torch
from torch import autograd


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )

    return x[(...,) + (None,) * dims_to_append]


def compute_gradient_penalty(discriminator, expert_obs, expert_actions, obs, actions):
    """Compute gradient penalty."""

    assert {k: v.size() for k, v in expert_obs.items()} == {
        k: v.size() for k, v in obs.items()
    }
    assert expert_actions.size() == actions.size()

    alpha = torch.rand(expert_actions.size(0), 1)

    # interpolate for observations.
    mixup_obs = dict()
    for k, expert_v in expert_obs.items():
        policy_v = obs[k]

        alpha_v = append_dims(alpha, expert_v.ndim)
        alpha_v = alpha_v.expand_as(expert_v).to(expert_v.device)
        mixup_obs_k = alpha_v * expert_v + (1 - alpha_v) * policy_v
        mixup_obs_k.requires_grad = True

        mixup_obs[k] = mixup_obs_k

    # interpolate for actions.
    alpha_actions = append_dims(alpha, expert_actions.ndim)
    alpha_actions = alpha_actions.expand_as(expert_actions).to(expert_actions.device)
    mixup_actions = alpha_actions * expert_actions + (1 - alpha_actions) * actions
    mixup_actions.requires_grad = True

    disc = discriminator(mixup_obs, mixup_actions)
    ones = torch.ones(disc.size(), device=disc.device)
    grads = autograd.grad(
        outputs=disc,
        inputs=(*mixup_obs.values(), mixup_actions),
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )
    grad = torch.cat([grad.flatten(start_dim=1) for grad in grads], dim=-1)

    grad_pen = (grad.norm(2, dim=1) - 1).pow(2).sum()
    return grad_pen
