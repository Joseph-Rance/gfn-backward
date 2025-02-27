import torch
from torch_scatter import scatter

from back_gfn.tasks.util import TrajectoryBalanceBase


def huber(x, beta=1, i_delta=4):
    ax = torch.abs(x)
    return torch.where(ax <= beta, 0.5 * x * x, beta * (ax - beta / 2)) * i_delta

# unfortunately, torch-discounted-cumsum no longer works :(
def discounted_cumsum_left(l, gamma):
    l = l.clone()
    for i in range(1, len(l)):
        l[i] += l[i-1] * gamma
    return l


class TrajectoryBalancePrefAC(TrajectoryBalanceBase):
    # Actor Critic backward loss

    def __init__(self, ctx, sampler, device, preference_strength=0, gamma=1, entropy_loss_multiplier=1, **kwargs):

        super().__init__(ctx, sampler, device, preference_strength, **kwargs)

        self.gamma = gamma
        self.entropy_loss_multiplier = entropy_loss_multiplier

    def compute_batch_losses(self, model, batch, _target_model, loss=huber):

        forward_log_Z = model.logZ(batch.cond_info[batch.from_p_b.logical_not()])[:, 0]
        forward_clipped_log_R = torch.maximum(batch.log_rewards[batch.from_p_b.logical_not()], torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        p_b_mask = batch.from_p_b.to(self.device).repeat_interleave(batch.traj_lens)
        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)[p_b_mask.logical_not()]
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0) - 1
        first_graph_idxs = torch.roll(final_graph_idxs, 1, dims=0)
        first_graph_idxs[0] = 0

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        c = batch.traj_lens.sum()
        rewards = - torch.concatenate(batch.bbs_costs[batch.from_p_b])

        state_vals = per_graph_out[:, 1].detach()  # 0 is for reward pred (unused)
        prev_state_vals = torch.roll(per_graph_out[:, 1], -1, 0)  # prev state when going backwards

        advantage = (rewards + self.gamma * state_vals - prev_state_vals)
        critic_loss = - loss(advantage)
        advantage = advantage.detach()

        p_B_loss = - (advantage * log_p_B[batch.from_p_b.repeat_interleave(batch.traj_lens)]).mean() \
                   + self.entropy_loss_multiplier * sum([(i * i.exp()).sum(1) for i in log_p_B[p_b_mask]])

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (forward_log_Z + traj_log_p_F[batch.from_p_b.logical_not()]) - (forward_clipped_log_R + traj_log_p_B[batch.from_p_b.logical_not()])
        tb_loss = loss(traj_diffs).mean()  # train p_F with p_B from prev. iteration
                                           # (slightly different from algorithm 1 in the paper)

        loss = tb_loss + p_B_loss + critic_loss

        info = {
            "log_z": forward_log_Z.mean().item(),
            "log_p_f": traj_log_p_F[batch.from_p_b.logical_not()].mean().item(),
            "log_p_b": traj_log_p_B[batch.from_p_b.logical_not()].mean().item(),
            "log_r": forward_clipped_log_R.mean().item(),
            "tb_loss": tb_loss.item(),
            "p_b_loss": p_B_loss.item(),
            "critic_loss": critic_loss.item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalancePrefDQN(TrajectoryBalanceBase):
    # DQN backward loss

    def __init__(self, ctx, sampler, device, preference_strength=0, gamma=1, entropy_loss_multiplier=1, **kwargs):

        super().__init__(ctx, sampler, device, preference_strength, **kwargs)

        self.gamma = gamma
        self.entropy_loss_multiplier = entropy_loss_multiplier

    def compute_batch_losses(self, model, batch, target_model, loss=huber):

        forward_log_Z = model.logZ(batch.cond_info[batch.from_p_b.logical_not()])[:, 0]
        forward_clipped_log_R = torch.maximum(batch.log_rewards[batch.from_p_b.logical_not()], torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        p_b_mask = batch.from_p_b.to(self.device).repeat_interleave(batch.traj_lens)
        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)[p_b_mask.logical_not()]
        log_q_B = bck_cat.log_prob(batch.bck_actions, softmax=False)  # represents states values
        log_p_B = bck_cat.log_prob(batch.bck_actions)                 # represents probability distribution

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0) - 1

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0
        
        log_q_B = torch.roll(log_q_B, -1, 0)
        log_q_B[batch.is_sink] = 0

        with torch.no_grad():
            _target_fwd_cat, target_bck_cat, _target_per_graph_out = target_model(batch, batch.cond_info[batch_idx])
        max_target_q_B = target_bck_cat._compute_batchwise_max()[0]  # docstring not right for this fn?

        target = -torch.concatenate(batch.bbs_costs[batch.from_p_b]) + self.gamma * max_target_q_B[batch.from_p_b]
        q_pred = log_q_B[batch.from_p_b.repeat_interleave(batch.traj_lens)].exp()
        p_B_loss = loss(target - q_pred).mean() \
                 + self.entropy_loss_multiplier * sum([(i * i.exp()).sum(1) for i in log_p_B[p_b_mask]])

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (forward_log_Z + traj_log_p_F[batch.from_p_b.logical_not()]) - (forward_clipped_log_R + traj_log_p_B[batch.from_p_b.logical_not()])
        tb_loss = loss(traj_diffs).mean()  # train p_F with p_B from prev. iteration
                                           # (slightly different from algorithm 1 in the paper)

        loss = tb_loss + p_B_loss

        info = {
            "log_z": forward_log_Z.mean().item(),
            "log_p_f": traj_log_p_F[batch.from_p_b.logical_not()].mean().item(),
            "log_p_b": traj_log_p_B[batch.from_p_b.logical_not()].mean().item(),
            "log_r": forward_clipped_log_R.mean().item(),
            "tb_loss": tb_loss.item(),
            "p_b_loss": p_B_loss.item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalancePrefPPO(TrajectoryBalanceBase):
    # PPO (with baseline) backward loss

    def __init__(self, ctx, sampler, device, preference_strength=0, gamma=1, eps=0.2, entropy_loss_multiplier=1, **kwargs):

        super().__init__(ctx, sampler, device, preference_strength, **kwargs)

        self.gamma = gamma
        self.entropy_loss_multiplier = entropy_loss_multiplier
        self.eps = eps

    def compute_batch_losses(self, model, batch, target_model, loss=huber):

        forward_log_Z = model.logZ(batch.cond_info[batch.from_p_b.logical_not()])[:, 0]
        forward_clipped_log_R = torch.maximum(batch.log_rewards[batch.from_p_b.logical_not()], torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        p_b_mask = batch.from_p_b.to(self.device).repeat_interleave(batch.traj_lens)
        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)[p_b_mask.logical_not()]
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        with torch.no_grad():
            _target_fwd_cat, target_bck_cat, _target_per_graph_out = target_model(batch, batch.cond_info[batch_idx])
        target_log_p_B = target_bck_cat.log_prob(batch.bck_actions)

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0) - 1
        first_graph_idxs = torch.roll(final_graph_idxs, 1, dims=0)
        first_graph_idxs[0] = 0

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        target_log_p_B = torch.roll(target_log_p_B, -1, 0)
        target_log_p_B[batch.is_sink] = 0

        c = batch.traj_lens.sum()
        rewards = - torch.concatenate(batch.bbs_costs[batch.from_p_b])
        G = torch.zeros(rewards.shape)
        for l in batch.traj_lens[::-1]:
            G[c-l:c] = discounted_cumsum_left(rewards[c-l:c], self.gamma)
            c -= l

        advantage = (G - per_graph_out[:, 1])  # 0 is for reward pred (unused)
        baseline_loss = - loss(advantage)
        advantage = advantage.detach()

        idxs = batch.from_p_b.repeat_interleave(batch.traj_lens)
        ratio = (log_p_B[idxs] - target_log_p_B[idxs]).exp()

        p_B_loss = - torch.minimum(ratio * advantage, torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage).mean() \
                   + self.entropy_loss_multiplier * sum([(i * i.exp()).sum(1) for i in log_p_B[p_b_mask]])

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (forward_log_Z + traj_log_p_F[batch.from_p_b.logical_not()]) - (forward_clipped_log_R + traj_log_p_B[batch.from_p_b.logical_not()])
        tb_loss = loss(traj_diffs).mean()  # train p_F with p_B from prev. iteration
                                           # (slightly different from algorithm 1 in the paper)

        loss = tb_loss + p_B_loss + baseline_loss

        info = {
            "log_z": forward_log_Z.mean().item(),
            "log_p_f": traj_log_p_F[batch.from_p_b.logical_not()].mean().item(),
            "log_p_b": traj_log_p_B[batch.from_p_b.logical_not()].mean().item(),
            "log_r": forward_clipped_log_R.mean().item(),
            "tb_loss": tb_loss.item(),
            "p_b_loss": p_B_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalancePrefREINFORCE(TrajectoryBalanceBase):
    # REINFORCE (with baseline) backward loss

    def __init__(self, ctx, sampler, device, preference_strength=0, gamma=1, entropy_loss_multiplier=1, **kwargs):

        super().__init__(ctx, sampler, device, preference_strength, **kwargs)

        self.gamma = gamma
        self.entropy_loss_multiplier = entropy_loss_multiplier

    def compute_batch_losses(self, model, batch, _target_model, loss=huber):

        forward_log_Z = model.logZ(batch.cond_info[batch.from_p_b.logical_not()])[:, 0]
        forward_clipped_log_R = torch.maximum(batch.log_rewards[batch.from_p_b.logical_not()], torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        p_b_mask = batch.from_p_b.to(self.device).repeat_interleave(batch.traj_lens)
        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)[p_b_mask.logical_not()]
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0) - 1
        first_graph_idxs = torch.roll(final_graph_idxs, 1, dims=0)
        first_graph_idxs[0] = 0

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        c = batch.traj_lens.sum()
        rewards = - torch.concatenate(batch.bbs_costs[batch.from_p_b])
        G = torch.zeros(rewards.shape)
        for l in batch.traj_lens[::-1]:
            G[c-l:c] = discounted_cumsum_left(rewards[c-l:c], self.gamma)
            c -= l

        advantage = (G - per_graph_out[:, 1])  # 0 is for reward pred (unused)
        baseline_loss = - loss(advantage)
        advantage = advantage.detach()

        p_B_loss = - (advantage * log_p_B[batch.from_p_b.repeat_interleave(batch.traj_lens)]).mean() \
                   + self.entropy_loss_multiplier * sum([(i * i.exp()).sum(1) for i in log_p_B[p_b_mask]])

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (forward_log_Z + traj_log_p_F[batch.from_p_b.logical_not()]) - (forward_clipped_log_R + traj_log_p_B[batch.from_p_b.logical_not()])
        tb_loss = loss(traj_diffs).mean()  # train p_F with p_B from prev. iteration
                                           # (slightly different from algorithm 1 in the paper)

        loss = tb_loss + p_B_loss + baseline_loss

        info = {
            "log_z": forward_log_Z.mean().item(),
            "log_p_f": traj_log_p_F[batch.from_p_b.logical_not()].mean().item(),
            "log_p_b": traj_log_p_B[batch.from_p_b.logical_not()].mean().item(),
            "log_r": forward_clipped_log_R.mean().item(),
            "tb_loss": tb_loss.item(),
            "p_b_loss": p_B_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalanceUniform(TrajectoryBalanceBase):

    def compute_batch_losses(self, model, batch, _target_model, loss=huber):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = batch.log_p_B

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0) - 1

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        loss = loss(traj_diffs).mean()

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalanceTLM(TrajectoryBalanceBase):

    def compute_batch_losses(self, model, batch, _target_model, loss=huber):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0) - 1

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        back_loss = traj_log_p_B.mean()
        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        tb_loss = loss(traj_diffs).mean()  # train p_F with p_B from prev. iteration
                                                    # (slightly different from algorithm 1 in the paper)

        loss = tb_loss + back_loss

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "tb_loss": tb_loss.item(),
            "back_loss": back_loss.item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalanceMaxEnt(TrajectoryBalanceBase):

    def compute_batch_losses(self, model, batch, _target_model, loss=huber):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=self.device)).float()

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0)
        first_graph_idxs = torch.roll(final_graph_idxs, 1, dims=0)
        first_graph_idxs[0] = 0

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)

        fwd_cat, bck_cat, per_graph_out = model(batch, batch.cond_info[batch_idx])

        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs - 1] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        log_n_preds = per_graph_out[:, 1]  # 0 is for reward pred (unused)
        log_n_preds[first_graph_idxs] = 0

        # we want to minimise (for all i):
        #     l(s_i) - l(s_{i+d}) - sum_{t=i}^{i+d-1}[ log(q(s_t|s_{t+1})) ]
        # where l and log.q are learnt and we let l(s_0) = 0. This allows us to learn
        #      l(s_0) = 0
        #       l(s') = log(sum_{s in parents(s')}[ exp(l(s)) ])
        #     log(q(s|s')) = l(s) - l(s')
        # For a non-rigorous, intuitive explanation, consider the case where d=1. Then we are trying
        # to minimise:
        #     l(s_i) - l(s_{i+1}) - log(q(s_i|s_{i+1}))
        # so we will find something that looks like
        #     l(s_{i+1}) = l(s_i) - log(q(s_i|s_{i+1}))
        # apply exp:
        #     exp(l(s_{i+1})) = exp(l(s_i))/q(s_i|s_{i+1})
        # And now consider the expected value over s_i of the RHS:
        #     E[ exp(l(s_i))/q(s_i|s_{i+1}) ]
        #     = sum_{s in parents(s_{i+1})}[ exp(l(s)) ]
        # so if we learn an accurate q, it makes sense that we will find an accurate l. The reverse
        # also trivially holds, since we want q(s|s') = n(s)/n(s')
        # note: q(.|s) MUST be normalised, otherwise we can just learn l(.)=0 and q(.|.)=1
        # we let d = 1. Then, since l(s_0) = 0, the loss function becomes:
        #     l(s_F) + sum[ log(q(s_t|s_{t+1})) ]
        # where the sum is over the full trajectory and s_F is the last state

        traj_pred_l = log_n_preds[torch.maximum(final_graph_idxs - 2, first_graph_idxs)]  # kind of wasteful
        n_loss = loss(traj_log_p_B + traj_pred_l).mean()
        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        tb_loss = loss(traj_diffs).mean()

        loss = tb_loss + n_loss

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "tb_loss": tb_loss.item(),
            "n_loss": n_loss.item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalanceFree(TrajectoryBalanceBase):

    def compute_batch_losses(self, model, batch, _target_model, loss=huber):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        final_graph_idxs = torch.cumsum(batch.traj_lens, 0) - 1

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idxs] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        # note: nan loss here may require updating action masks to have only -1_000 penalty
        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        loss = loss(traj_diffs).mean()

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "loss": loss.item()
        }

        return loss, info
