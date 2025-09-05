"""
DGAIL: Diffusion Generative Adversarial Imitation Learning.

To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
from model.diffusion.diffusion_ppo import PPODiffusion
from util.discriminator import compute_gradient_penalty


class GAILDiffusion(PPODiffusion):
    def __init__(
        self,
        discriminator,
        divergence="wass",
        max_discriminator_diff=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Discriminator module.
        self.discriminator = discriminator.to(self.device)

        # Discriminator loss type.
        self.divergence = divergence

        # Max discriminator difference when computing loss.
        self.max_discriminator_diff = max_discriminator_diff

    def discriminator_loss(
        self,
        obs,
        actions,
        expert_obs,
        expert_actions,
    ):
        """
        GAIL discriminator loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (pB, To, Do)
            rgb: (pB, To, C, H, W)
        actions: (pB, Ta, Da)
        expert_obs: dict with key state/rgb; more recent obs at the end
            state: (eB, To, Do)
            rgb: (eB, To, C, H, W)
        expert_actions: (eB, Ta, Da)
        """

        eB = expert_actions.size(0)
        pB = actions.size(0)

        # Get discriminator outputs for expert and online data.
        policy_d = self.discriminator(obs, actions)  # [pB, 1]
        expert_d = self.discriminator(expert_obs, expert_actions)  # [eB, 1]

        # Compute loss depending on type.
        if self.divergence == "js":
            # Classic binary cross entropy loss.
            ones = torch.ones(eB, device=self.device)
            zeros = torch.zeros(pB, device=self.device)
            disc_label = torch.cat([ones, zeros], dim=0).unsqueeze(-1)  # [eB + pB, 1]

            disc_output = torch.cat([expert_d, policy_d], dim=0)
            disc_loss = F.binary_cross_entropy_with_logits(
                disc_output, disc_label, reduction="sum"
            )
            disc_loss /= eB + pB // 2
        elif self.divergence == "rkl":
            # Reverse KL divergence loss.
            disc_loss = torch.mean(torch.exp(-expert_d)) + torch.mean(policy_d)
        elif self.divergence == "wass":
            # Wasserstein-1 distance loss.
            disc_loss = torch.mean(policy_d) - torch.mean(expert_d)
            if self.max_discriminator_diff is not None:
                disc_loss = torch.clamp(
                    disc_loss, -self.max_discriminator_diff, self.max_discriminator_diff
                )
        elif self.divergence == "least_squares":
            # Least squares loss from https://arxiv.org/pdf/2104.02180.
            # DeepMimic code: https://github.com/xbpeng/DeepMimic/blob/master/learning/amp_agent.py#L251-L257
            expert_loss = 0.5 * (expert_d - 1) ** 2
            policy_loss = 0.5 * (policy_d + 1) ** 2
            disc_loss = 0.5 * (expert_loss.mean() + policy_loss.mean())
        else:
            raise ValueError(f"Invalid divergence {self.divergence}.")

        if eB < pB:
            # choose random policy samples to do gradient penalty over.
            policy_inds = torch.randint(0, pB, (eB,))

            grad_expert_obs = expert_obs
            grad_expert_actions = expert_actions
            grad_obs = {k: v[policy_inds] for k, v in obs.items()}
            grad_actions = actions[policy_inds]
        elif eB > pB:
            # choose random expert samples to do gradient penalty over.
            expert_inds = torch.randint(0, eB, (pB,))

            grad_expert_obs = {k: v[expert_inds] for k, v in expert_obs.items()}
            grad_expert_actions = expert_actions[expert_inds]
            grad_obs = obs
            grad_actions = actions
        else:
            grad_expert_obs = expert_obs
            grad_expert_actions = expert_actions
            grad_obs = obs
            grad_actions = actions

        # Compute gradient penalty.
        grad_pen = compute_gradient_penalty(
            self.discriminator,
            expert_obs=grad_expert_obs,
            expert_actions=grad_expert_actions,
            obs=grad_obs,
            actions=grad_actions,
        )
        grad_pen /= min(eB, pB)

        return disc_loss, grad_pen

    @torch.no_grad()
    def get_reward(self, obs, actions):
        """
        Computes reward.

        obs: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        actions: (B, Ta, Da)
        """

        d = self.discriminator(obs, actions).view(-1)

        if self.divergence == "js":
            s = torch.sigmoid(d)
            rewards = s.log() - (1 - s).log()
        elif self.divergence == "rkl":
            rewards = torch.sigmoid(d)
        elif self.divergence == "wass":
            rewards = d
        elif self.divergence == "least_squares":
            rewards = F.relu(1.0 - 0.25 * (d - 1.0) ** 2)
        else:
            raise ValueError(f"Invalid divergence {self.divergence}.")

        return rewards

    @torch.no_grad()
    def get_divergence(self, obs, actions, expert_obs, expert_actions):
        """Computes the divergence for logging.

        obs: dict with key state/rgb; more recent obs at the end
            state: (pB, To, Do)
            rgb: (pB, To, C, H, W)
        actions: (pB, Ta, Da)
        expert_obs: dict with key state/rgb; more recent obs at the end
            state: (eB, To, Do)
            rgb: (eB, To, C, H, W)
        expert_actions: (eB, Ta, Da)
        """

        # Get discriminator outputs for expert and online data.
        policy_d = self.discriminator(obs, actions)  # [pB, 1]
        expert_d = self.discriminator(expert_obs, expert_actions)  # [eB, 1]

        if self.divergence == "js":
            # Not implemented.
            divergence = None
        elif self.divergence == "rkl":
            disc_out = torch.exp(-expert_d).mean()
            policy_out = policy_d.mean()
            divergence = disc_out + policy_out
        elif self.divergence == "wass":
            divergence = expert_d.mean() - policy_d.mean()
        elif self.divergence == "least_squares":
            divergence = (policy_d < 0).float().mean() + (expert_d > 0).float().mean()
        else:
            raise ValueError(f"Invalid divergence {self.divergence}.")

        return divergence
