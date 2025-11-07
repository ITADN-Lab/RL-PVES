import copy

import gymnasium as gym
import numpy as np
import torch

from .agent import DrlAgent, TargetNetMixin
from .ddpg import Ddpg
from .networks import SACActorNet, DDPGCriticNet
from .shared_code.processing import batch_to_tensors
from .shared_code.memory import (ReplayMemory, PrioritizedReplayMemory)
# TODO: Add Prio replay buffer to SAC as optional
from .shared_code.exploration import GaussianNoise


class Sac(Ddpg):
    def __init__(self, env, memory_size=500000,
                 gamma=0.99, batch_size=256, tau=0.001, start_train=2000,
                 actor_fc_dims=(256, 256, 256), critic_fc_dims=(256, 256, 256),
                 actor_learning_rate=0.0001, critic_learning_rate=0.0005,
                 entropy_learning_rate=0.0001,
                 train_interval=1, train_steps=1,
                 optimizer='Adam', fixed_alpha=None, target_entropy=None,
                 grad_clip=None, layer_norm=True, *args, **kwargs):
        self.start_train = max(start_train, batch_size)
        actor_fc_dims = list(actor_fc_dims)
        critic_fc_dims = list(critic_fc_dims)

        # TODO: Discrete actions?!
        assert isinstance(env.action_space, gym.spaces.Box)
        # Actor only outputs tanh action space [-1, 1] -> rescale
        env = gym.wrappers.RescaleAction(env, -1, 1)
        # The wrapper destroys seeding of the action space -> re-seed
        env.action_space.seed(kwargs['seed'])

        super(Ddpg, self).__init__(env, gamma, *args, **kwargs)

        self.tau = tau
        self.update_counter = 0
        self.batch_size = batch_size  # Move to superclass?
        self.batch_idxs = np.arange(
            self.batch_size, dtype=np.int32)   # Move to superclass?
        self.grad_clip = grad_clip
        self.train_interval = train_interval
        self.train_steps = train_steps

        try:
            self.n_rewards = len(env.reward_space.low)
        except AttributeError:
            self.n_rewards = 1

        self._init_networks(list(actor_fc_dims), actor_learning_rate,
                            list(critic_fc_dims), critic_learning_rate,
                            optimizer, layer_norm)

        self.device = self.actor.device  # TODO: How to do for multiple nets?
        self._init_memory(memory_size)

        self.fixed_alpha = fixed_alpha
        if self.fixed_alpha:
            self.alpha = torch.tensor(fixed_alpha)
        else:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                 # Heuristic from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=entropy_learning_rate)
            self.alpha = self.log_alpha.exp()

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, layer_norm):
        self.actor = SACActorNet(
            self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate,
            optimizer=optimizer, output_activation='tanh', layer_norm=layer_norm)
        # target comes from the current policy instead of a target policy
        # TODO: variable optimizer for Critics as well
        self.critic1 = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic2_target = copy.deepcopy(self.critic2)

    @torch.no_grad()
    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train:
            return self.env.action_space.sample()

        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        return np.clip(self.actor(obs, act_only=True).cpu().numpy(), -1, 1)

    @torch.no_grad()
    def test_act(self, obs, deterministic=True):
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        # Return only the mean, deterministic action for testing
        return np.clip(
            self.actor(obs, act_only=True, deterministic=deterministic)
            .cpu().numpy(), -1, 1)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        self._train_critic(obss, acts, rewards, next_obss, dones)

        self._train_actor(obss, acts, rewards, next_obss, dones)

        if not self.fixed_alpha:
            self._update_alpha(obss)

        self._soft_target_update(self.critic1, self.critic1_target, self.tau)
        self._soft_target_update(self.critic2, self.critic2_target, self.tau)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        targets = self._compute_targets(next_obss, dones, rewards)
        for critic in (self.critic1, self.critic2):
            critic.optimizer.zero_grad()
            q_values = critic(obss, acts)
            critic_loss = critic.loss(targets, q_values)
            critic_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), self.grad_clip)
            critic.optimizer.step()

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        self.actor.optimizer.zero_grad()
        entropy, acts = self.actor.forward(obss)
        q_values = torch.minimum(self.critic1(obss, acts).sum(axis=1),
                                 self.critic2(obss, acts).sum(axis=1))

        actor_loss = -(q_values + self.alpha.detach() * entropy).mean()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip)
        self.actor.optimizer.step()

    def _update_alpha(self, obss):
        entropy, _ = self.actor.forward(obss)
        self.alpha_optimizer.zero_grad()

        alpha_loss = -((self.target_entropy - entropy.detach()) 
                       * self.log_alpha).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        next_entropy, next_acts = self.actor.forward(next_obss)
        target_values1 = self.critic1_target(next_obss, next_acts)
        target_values2 = self.critic2_target(next_obss, next_acts)
        target_values = torch.minimum(target_values1, target_values2)

        target_values += (self.alpha.item() * next_entropy).reshape(-1, 1)

        target_values[dones == 1.0] = 0.0

        return rewards + self.gamma * target_values

    def store_model(self):
        torch.save(self.actor.state_dict(), self.path + 'actor.pth')
        torch.save(self.critic1.state_dict(), self.path + 'critic1.pth')
        torch.save(self.critic2.state_dict(), self.path + 'critic2.pth')

    def load_model(self):
        actor_weight_dict = torch.load(
            self.path + 'actor.pth', map_location=torch.device(self.device))
        self.actor.load_state_dict(actor_weight_dict)
        critic1_weight_dict = torch.load(
            self.path + 'critic.pth', map_location=torch.device(self.device))
        self.critic1.load_state_dict(critic1_weight_dict)
        self.critic1_target.load_state_dict(critic1_weight_dict)
        critic2_weight_dict = torch.load(
            self.path + 'critic.pth', map_location=torch.device(self.device))
        self.critic2.load_state_dict(critic2_weight_dict)
        self.critic2_target.load_state_dict(critic2_weight_dict)
