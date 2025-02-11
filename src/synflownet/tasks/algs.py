import torch
from torch_scatter import scatter

from synflownet.tasks.util import TrajectoryBalanceBase

class TrajectoryBalancePref(TrajectoryBalanceBase):

    def compute_batch_losses(self, model, batch):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idx] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        p_B_loss = 0  # TODO: apply REINFORCE for batch.log_bbs_cost (also test other algorithms / without log)

        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        loss = (traj_diffs * traj_diffs).mean()

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalanceUniform(TrajectoryBalanceBase):

    def compute_batch_losses(self, model, batch):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = batch.log_p_B

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        loss = (traj_diffs * traj_diffs).mean()

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalanceTLM(TrajectoryBalanceBase):

    def compute_batch_losses(self, model, batch):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=self.device)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=self.device).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idx] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        back_loss = traj_log_p_B.mean()
        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        td_loss = (traj_diffs * traj_diffs).mean()  # train p_F with p_B from prev. iteration
                                                    # (slightly different from algorithm 1 in the paper)

        loss = td_loss + back_loss

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "td_loss": td_loss.item(),
            "back_loss": back_loss.item(),
            "loss": loss.item()
        }

        return loss, info


class TrajectoryBalanceMaxEnt(TrajectoryBalanceBase):

    def huber(_self, x, beta=1, i_delta=4):
        ax = torch.abs(x)
        return torch.where(ax <= beta, 0.5 * x * x, beta * (ax - beta / 2)) * i_delta

    def compute_batch_losses(self, model, batch):

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
        n_loss = self.huber(traj_log_p_B + traj_pred_l).mean()
        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        tb_loss = self.huber(traj_diffs).mean()

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

    def compute_batch_losses(self, model, batch):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=DEVICE)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=DEVICE).repeat_interleave(batch.traj_lens)
        fwd_cat, bck_cat, _per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        log_p_B = bck_cat.log_prob(batch.bck_actions)

        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # don't count padding states on forward (kind of wasteful to compute these)
        log_p_F[final_graph_idx] = 0

        # don't count starting states or consecutive sinks (due to padding) on backward
        log_p_B = torch.roll(log_p_B, -1, 0)
        log_p_B[batch.is_sink] = 0

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        loss = (traj_diffs * traj_diffs).mean()

        info = {
            "log_z": log_Z.mean().item(),
            "log_p_f": traj_log_p_F.mean().item(),
            "log_p_b": traj_log_p_B.mean().item(),
            "log_r": clipped_log_R.mean().item(),
            "loss": loss.item()
        }

        return loss, info