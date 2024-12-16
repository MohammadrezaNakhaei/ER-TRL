import os
from turtle import position
import torch
import torch.optim as optim
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import product
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.torch.brac import divergences
from rlkit.torch.brac import utils
import pdb


class ERTRL(OfflineMetaRLAlgorithm): # entropy regularized task representtaion learning
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            extreme_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            extreme_tasks=extreme_tasks,
            goal_radius=goal_radius,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        self.recurrent                      = kwargs['recurrent']
        self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.use_brac                       = kwargs['use_brac']
        self.use_value_penalty              = kwargs['use_value_penalty']
        self.alpha_max                      = kwargs['alpha_max']
        self._c_iter                        = kwargs['c_iter']
        self.train_alpha                    = kwargs['train_alpha']
        self._target_divergence             = kwargs['target_divergence']
        self.alpha_init                     = kwargs['alpha_init']
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']
        self.z_loss_weight                  = kwargs['z_loss_weight']
        self.max_entropy                    = kwargs['max_entropy']
        self.allow_backward_z               = kwargs['allow_backward_z']
        self.generator_dim                  = kwargs['generator_dim']
        self.mi_mir_loss_weight             = kwargs['mir_loss_weight']
        self.n_update_gan                   = kwargs['n_update_gan']
        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()
        self.cross_entropy_loss             = nn.CrossEntropyLoss()

        self.qf1, self.qf2, self.vf, self.c, self.generator, self.discriminator = nets[1:]
        self.target_vf                      = self.vf.copy()

        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
        self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.context_optimizer              = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.generator_optimizer            = optimizer_class(self.generator.parameters(), lr=0.0002)
        self.discriminator_optimizer        = optimizer_class(self.discriminator.parameters(), lr=0.0002)
        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf, self.c, self.generator, self.discriminator]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)

        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def FOCAL_z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b # index in task * batch dim
            for j in range(i+1, len(indices)):
                idx_j = j * b # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)

    def _discriminator_loss(self, t, b, real_actions, fake_actions, obs, task_z):
        """Loss computed to train the GAN discriminator.
        """
        # YOUR CODE HERE
        criterion1 = nn.BCELoss(reduction='mean')
        criterion2 = nn.BCELoss(reduction='mean')
        out_real = self.discriminator(t, b, real_actions, obs, task_z)
        out_fake = self.discriminator(t, b, fake_actions, obs, task_z)
        target_real = torch.ones((t*b, 1), device=ptu.device)
        target_fake = torch.zeros((t*b, 1), device=ptu.device)
        d_loss_real = criterion1(out_real, target_real)
        d_loss_fake = criterion2(out_fake, target_fake)
        return d_loss_real, d_loss_fake

    def _generator_loss(self, t, b, fake_actions, obs, task_z):
        criterion = nn.BCELoss(reduction='mean')
        out = self.discriminator(t, b, fake_actions, obs, task_z)
        targets = torch.ones((t*b, 1), device=ptu.device)
        return criterion(out, targets)


    def _take_step(self, indices, context):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)

        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()

        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        for _ in range(self.n_update_gan):
            eps = torch.randn((t*b, self.generator_dim), device=ptu.device)
            fake_actions = self.generator(t, b, obs, task_z, eps)
            d_loss_real, d_loss_fake = self._discriminator_loss(t, b, actions, fake_actions, obs, task_z)
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = d_loss_fake + d_loss_real
            discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()
            generator_loss =  self._generator_loss(t, b, fake_actions, obs, task_z)
            generator_loss.backward(retain_graph=True)
            self.generator_optimizer.step()

            self.loss["generator_loss"] = generator_loss.item()
            self.loss['discriminator_loss'] = discriminator_loss.item()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()

        # entropy maximization
        eps = torch.randn((t*b, self.generator_dim), device=ptu.device)
        #_, task_z, task_z_vars= self.agent(obs.view(t, b, -1), context, task_indices=indices)
        fake_actions = self.generator(t, b, obs, task_z, eps)
        fake_actions = fake_actions.reshape(t, b, -1)
        mu = fake_actions.mean(1, keepdim=True)
        diff = fake_actions-mu
        cov = 1/(b-1) * torch.transpose(diff, 1, 2) @ diff
        dist = torch.distributions.MultivariateNormal(mu.view(t,-1), cov+1e-5*torch.eye(action_dim, device=ptu.device))
        entorpy = dist.entropy()
        MI_loss = -self.mi_mir_loss_weight*torch.mean(entorpy)
        MI_loss.backward(retain_graph=True)
        self.loss["MI_loss"] = MI_loss.item()

        if self.use_FOCAL_cl:
            z_loss = self.z_loss_weight * self.FOCAL_z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                        self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()

        for i in range(len(self.agent.z_means[0])):
            z_mean = ptu.get_numpy(self.agent.z_means[0][i])
            name = f'train/z_mean_{i}'
            self.eval_statistics[name] = z_mean

        z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))

        self.eval_statistics['train/Z_variance'] = z_sig

        self.eval_statistics['train/loss_generator'] = ptu.get_numpy(generator_loss)
        self.eval_statistics['train/loss_dicriminator'] = ptu.get_numpy(discriminator_loss)
        self.eval_statistics['train/loss_MI'] = ptu.get_numpy(MI_loss)
        if self.use_FOCAL_cl:
            self.eval_statistics['train/loss_focal'] = ptu.get_numpy(z_loss)

        self.eval_statistics['train/loss_qf'] = np.mean(ptu.get_numpy(qf_loss))
        self.eval_statistics['train/loss_vf'] = np.mean(ptu.get_numpy(vf_loss))
        self.eval_statistics['train/loss_policy'] = np.mean(ptu.get_numpy(
            policy_loss
        ))
        if self.use_brac:
            self.eval_statistics['train/loss_dual_brack'] = np.mean(ptu.get_numpy(c_loss))
        self.eval_statistics['train/avg_q_values'] =  ptu.get_numpy(q1_pred).mean()
        self.eval_statistics['train/avg_v_values'] =  ptu.get_numpy(v_pred).mean()
        self.eval_statistics['train/log_policy'] =  ptu.get_numpy(log_pi).mean()
        self.eval_statistics['train/alpha'] =  ptu.get_numpy((self._alpha_var).reshape(-1)).mean()
        self.eval_statistics['train/div_estimate'] =  ptu.get_numpy(div_estimate).mean()
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            generator=self.generator.state_dict(),
            discriminator=self.discriminator.state_dict(),
            c=self.c.state_dict(),
            )
        return snapshot
