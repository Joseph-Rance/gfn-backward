"""Run with: python -m synflownet.tasks.preference_reward"""
# NOTE: algorithm implementation not here because will be merged with another file

from itertools import cycle
import pickle
import os
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset
import torch_geometric.data as gd
from torch_scatter import scatter
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from synflownet import ObjectProperties, LogScalar
from synflownet.envs.synthesis_building_env import ReactionTemplateEnvContext
from synflownet.envs.graph_building_env import Graph, GraphActionType
from synflownet.data.replay_buffer import detach_and_cpu
from synflownet.models import bengio2021flow
from synflownet.algo.graph_sampling import Sampler

from synflownet.tasks.n_model import GraphTransformerSynGFN

DEVICE = "cuda"
SEED = 0
PRINT_EVERY = 1
REWARD_THRESH = 0.9
PREFERENCE_STRENGTH = 0.5


class ReactionTask:

    def __init__(self):
        self.model = bengio2021flow.load_original_model()
        self.model = self.model.to(DEVICE)
        self.num_cond_dim = 32

    def cond_info_to_logreward(self, cond_info, flat_reward):
        return LogScalar(flat_reward.squeeze().clamp(min=1e-30).log() * cond_info["beta"])

    def compute_obj_properties(self, mols):
        graphs = [bengio2021flow.mol2graph(m) for m in mols]
        batch = gd.Batch.from_data_list(graphs)
        batch.to(DEVICE)
        preds = self.model(batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        preds = preds.clip(1e-4, 100).reshape((-1, 1))
        return ObjectProperties(preds)


class ReactionTemplateEnv:

    def __init__(self, ctx):
        self.ctx = ctx

    def empty_graph(self):
        return Graph()

    def step(self, smi, action):

        mol = self.ctx.get_mol(smi)

        if action.action is GraphActionType.Stop:
            return mol

        elif action.action is GraphActionType.AddReactant \
          or action.action is GraphActionType.AddFirstReactant:
            return self.ctx.get_mol(self.ctx.building_blocks[action.bb])

        elif action.action is GraphActionType.ReactUni:
            return self.ctx.unimolecular_reactions[action.rxn].run_reactants((mol,))

        else:
            reaction = self.ctx.bimolecular_reactions[action.rxn]
            reactant2 = self.ctx.get_mol(self.ctx.building_blocks[action.bb])
            return reaction.run_reactants((mol, reactant2))

    def count_backward_transitions(self, g):
        parents_count = 0

        gd = self.ctx.graph_to_Data(g, traj_len=4)
        for _, atype in enumerate(self.ctx.bck_action_type_order):
            nza = getattr(gd, atype.mask_name)[0].nonzero()
            parents_count += len(nza)

        return parents_count


class SynthesisSampler(Sampler):

    def __init__(self, ctx, env, max_len=None):

        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 5

    def sample_from_model(self, model, n, cond_info, random_action_prob=0.0):

        data = [
            {"traj": [], "is_valid": True, "bbs": []}
            for _ in range(n)
        ]

        bck_logprob = [[] for _ in range(n)]

        graphs = [self.env.empty_graph() for _ in range(n)]
        done = [False] * n

        for t in range(self.max_len):

            torch_graphs = [self.ctx.graph_to_Data(g, traj_len=t) for i, g in enumerate(graphs) if not done[i]]
            nx_graphs = [g for i, g in enumerate(graphs) if not done[i]]
            not_done_mask = torch.tensor(done, device=DEVICE).logical_not()

            fwd_cat, *_ = model(self.ctx.collate(torch_graphs).to(DEVICE), cond_info[not_done_mask])
            actions = fwd_cat.sample(nx_graphs=nx_graphs, model=model, random_action_prob=random_action_prob)
            graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a, fwd=True) for g, a in zip(torch_graphs, actions)]

            for i, j in zip((k for k in range(n) if not done[k]), range(n)):

                data[i]["traj"].append((graphs[i], graph_actions[j]))

                if graph_actions[j].action is GraphActionType.Stop:

                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=DEVICE).log())

                else:
                    
                    if graph_actions[j].action in [GraphActionType.AddFirstReactant,
                                                   GraphActionType.ReactBi]:
                        data[i]["bbs"].append(graph_actions[j].bb)

                    gp = self.env.step(graphs[i], graph_actions[j])

                    try:
                        Chem.SanitizeMol(gp)
                    except Exception as e:
                        data[i]["is_valid"] = False
                        done[i] = True
                        bck_logprob[i].append(torch.tensor([1.0], device=DEVICE).log())
                        continue

                    g = self.ctx.obj_to_graph(gp)

                    n_back = self.env.count_backward_transitions(g)
                    if n_back > 0:
                        bck_logprob[i].append(torch.tensor([1 / n_back], device=DEVICE).log())
                    else:
                        bck_logprob[i].append(torch.tensor([0.001], device=DEVICE).log())

                    if t == self.max_len - 1:
                        done[i] = True
                        continue

                    graphs[i] = g

                if done[i] and len(data[i]["traj"]) < 2:
                    data[i]["is_valid"] = False

            if all(done):
                break

        for i in range(n):
            data[i]["result"] = graphs[i]
            if bck_logprob[i]:
                data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)

        return data


class TrajectoryBalance:

    def __init__(self, ctx, sampler):
        self.ctx = ctx
        self.sampler = sampler

    def create_training_data_from_own_samples(self, model, n, cond_info, random_action_prob=0.0):

        cond_info = cond_info.to(DEVICE)
        data = self.sampler.sample_from_model(model, n, cond_info, random_action_prob)
        return data

    def construct_batch(self, trajs):

        torch_graphs = [
            self.ctx.graph_to_Data(i[0], traj_len=k)
            for tj in trajs
            for k, i in enumerate(tj["traj"])
        ]

        batch = self.ctx.collate(torch_graphs)
        batch.actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a, fwd=True)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.nx_graphs = [i[0] for tj in trajs for i in tj["traj"]]
        batch.log_rewards = torch.stack([t["log_reward"] for t in trajs])
        batch.cond_info = torch.stack([t["cond_info"]["encoding"] for t in trajs])
        batch.cond_info_beta = torch.stack([t["cond_info"]["beta"] for t in trajs])
        batch.secondary_masks = self.ctx.precompute_secondary_masks(batch.actions, batch.nx_graphs)
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.log_bbs_cost = torch.log(torch.stack([t["bbs_cost"] for t in trajs]))

        #batch.log_rewards -= torch.max(batch.log_rewards)
        #batch.bbs_cost -= torch.max(batch.bbs_cost) - torch.log(PREFERENCE_STRENGTH)

        # TODOJ: consider better methods of combining the reward
        batch.log_rewards -= batch.log_bbs_cost * PREFERENCE_STRENGTH

        return batch

    def compute_batch_losses(self, model, batch):

        log_Z = model.logZ(batch.cond_info)[:, 0]
        clipped_log_R = torch.maximum(batch.log_rewards, torch.tensor(-75, device=DEVICE)).float()

        batch_idx = torch.arange(batch.traj_lens.shape[0], device=DEVICE).repeat_interleave(batch.traj_lens)
        fwd_cat, per_graph_out = model(batch, batch.cond_info[batch_idx])
        for atype, (idcs, mask) in batch.secondary_masks.items():
            fwd_cat.set_secondary_masks(atype, idcs, mask)


        final_graph_idxs = torch.cumsum(batch.traj_lens, 0)
        first_graph_idxs = torch.roll(final_graph_idxs, 1, dims=0)
        first_graph_idxs[0] = 0
        log_n_preds = per_graph_out[:, 1]  # 0 is for reward pred (unused)


        # to want to minimise (for all i):
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

        neg_log_traj_n = scatter(
            batch.log_p_B, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum"
        )
        n_diff = neg_log_traj_n + log_n_preds[torch.maximum(final_graph_idxs - 2, first_graph_idxs)]
        n_loss = (n_diff * n_diff).mean()

        log_p_F = fwd_cat.log_prob(batch.actions, batch.nx_graphs, model)
        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=batch.traj_lens.shape[0], reduce="sum")

        traj_log_p_B = log_n_preds[torch.maximum(final_graph_idxs - 2, first_graph_idxs)]
        traj_log_p_B.detach()

        traj_diffs = (log_Z + traj_log_p_F) - (clipped_log_R + traj_log_p_B)
        tb_loss = (traj_diffs * traj_diffs).mean()

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


class DataSource(IterableDataset):
    def __init__(self, ctx, algo, task):

        self.iterators = []
        self.ctx = ctx
        self.algo = algo
        self.task = task

    def __iter__(self):

        its = [i() for i in self.iterators]

        while True:

            iterator_outputs = [next(i, None) for i in its]

            if any(i is None for i in iterator_outputs):
                break

            return self.algo.construct_batch(detach_and_cpu(sum(iterator_outputs, [])))

    def do_sample_model(self, model, num_samples):

        def iterator():
            while True:

                cond_info = {
                    "beta": torch.tensor(np.array(32).repeat(num_samples).astype(np.float32)),
                    "encoding": torch.zeros((num_samples, 32))
                }

                trajs = self.algo.create_training_data_from_own_samples(model, num_samples, cond_info["encoding"], 0)

                for i in range(len(trajs)):
                    trajs[i]["cond_info"] = {k: cond_info[k][i] for k in cond_info}

                self.compute_properties(trajs)
                self.compute_log_rewards(trajs)

                yield trajs[:num_samples]

        self.iterators.append(iterator)

    def do_sample_model_n_times(self, model, num_samples_per_batch, num_total):

        def iterator():
            num_so_far = 0

            while True:

                n_this_time = min(num_total - num_so_far, num_samples_per_batch)

                if n_this_time == 0:
                    break

                num_so_far += n_this_time

                cond_info = {
                    "beta": torch.tensor(np.array(32).repeat(n_this_time).astype(np.float32)),
                    "encoding": torch.zeros((num_samples_per_batch, 32))
                }

                trajs = self.algo.create_training_data_from_own_samples(model, n_this_time, cond_info["encoding"], 0)

                for i in range(len(trajs)):
                    trajs[i]["cond_info"] = {k: cond_info[k][i] for k in cond_info}
                    trajs[i]["bbs_cost"] = sum(self.ctx.bbs_costs[bb] for bb in trajs[i]["bbs"])

                self.compute_properties(trajs)
                self.compute_log_rewards(trajs)

                yield trajs[:n_this_time]

        self.iterators.append(iterator)

    def compute_properties(self, trajs):

        valid_idcs = torch.tensor([i for i in range(len(trajs)) if trajs[i].get("is_valid", True)]).long()
        objs = [self.ctx.graph_to_obj(trajs[i]["result"]) for i in valid_idcs]
        obj_props = self.task.compute_obj_properties(objs)

        all_fr = torch.zeros((len(trajs), obj_props.shape[1]))
        all_fr[valid_idcs] = obj_props

        for i in range(len(trajs)):
            trajs[i]["obj_props"] = all_fr[i]

    def compute_log_rewards(self, trajs):

        obj_props = torch.stack([t["obj_props"] for t in trajs])
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}

        log_rewards = self.task.cond_info_to_logreward(cond_info, obj_props)
        min_r = torch.as_tensor(-75).float()

        for i in range(len(trajs)):
            trajs[i]["log_reward"] = log_rewards[i] if trajs[i].get("is_valid", True) else min_r


if __name__ == "__main__":

    rel_path = "/".join(os.path.abspath(__file__).split("/")[:-2])

    with open(rel_path + "/data/building_blocks/enamine_bbs.txt", "r") as file:
        building_blocks = file.read().splitlines()

    with open(rel_path + "/data/templates/hb.txt", "r") as file:
        reaction_templates = file.read().splitlines()

    with open(rel_path + "/data/building_blocks/precomputed_bb_masks_enamine_bbs.pkl", "rb") as f:
        precomputed_bb_masks = pickle.load(f)

    task = ReactionTask()  # for reward
    ctx = ReactionTemplateEnvContext(  # deals with molecules
        num_cond_dim=task.num_cond_dim,
        building_blocks=building_blocks,
        reaction_templates=reaction_templates,
        precomputed_bb_masks=precomputed_bb_masks,
        fp_type="morgan_1024",
        fp_path=None,
        strict_bck_masking=False
    )
    env = ReactionTemplateEnv(ctx)  # for actions
    sampler = SynthesisSampler(ctx, env)  # for sampling policies
    algo = TrajectoryBalance(ctx, sampler)  # for computing loss / helps making batches
    model = GraphTransformerSynGFN(ctx, outs=2)

    Z_params = list(model._logZ.parameters())
    non_Z_params = [i for i in model.parameters() if all(id(i) != id(j) for j in Z_params)]

    opt = torch.optim.Adam(non_Z_params, 1e-4, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)
    opt_Z = torch.optim.Adam(Z_params, 1e-3, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda steps: 2 ** -(steps/2_000))
    lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(opt_Z, lambda steps: 2 ** -(steps/50_000))

    model.to(DEVICE)

    with torch.no_grad():
        train_src = DataSource(ctx, algo, task)  # gets training data inc rewards
        train_src.do_sample_model(model, 64)
        train_dl = torch.utils.data.DataLoader(train_src, batch_size=None)

    unique_scaffolds = set()
    num_unique_scaffolds = [0]
    num_mols_tested = [0]

    full_results = [[] for __ in range(6)]

    start_time = time.time()

    for it, batch in zip(range(1, 5001), cycle(train_dl)):

        batch = batch.to(DEVICE)
        
        model.train()

        loss, info = algo.compute_batch_losses(model, batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        opt.step()
        opt.zero_grad()
        opt_Z.step()
        opt_Z.zero_grad()

        lr_sched.step()
        lr_sched_Z.step()

        # test number of unique high reward scaffolds we found *during training*
        with torch.no_grad():

            mols = [ctx.graph_to_obj(batch.nx_graphs[i]) for i in (torch.cumsum(batch.traj_lens, 0) - 1)]
            rewards = torch.exp(batch.log_rewards / batch.cond_info_beta)

            murcko_scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]

            scaffolds_above_thresh = [smi for smi, r in zip(murcko_scaffolds, rewards) if r > REWARD_THRESH]
            unique_scaffolds.update(scaffolds_above_thresh)

            num_mols_tested.append(num_mols_tested[-1] + len(mols))
            num_unique_scaffolds.append(len(unique_scaffolds))

            np.save("unique_scaffolds.npy", list(zip(num_mols_tested, num_unique_scaffolds)))

            total_time = time.time() - start_time
            start_time = time.time()

            if it % PRINT_EVERY == 0:
                print(f"iteration {it} : loss:{info['loss']:7.3f} " \
                    f"sampled_reward_avg:{rewards.mean().item():6.4f} " \
                    f"time_spent:{total_time:4.2f} " \
                    f"logZ:{info['log_z']:7.4f} " \
                    f"gen scaffolds: {len(scaffolds_above_thresh)} " \
                    f"unique scaffolds: {len(unique_scaffolds)}")

            full_results[0].append(info["loss"])
            full_results[1].append(rewards.mean().item())
            full_results[2].append(total_time)
            full_results[3].append(info["log_z"])
            full_results[4].append(len(scaffolds_above_thresh))
            full_results[5].append(len(unique_scaffolds))
            np.save("full_results.npy", full_results)
