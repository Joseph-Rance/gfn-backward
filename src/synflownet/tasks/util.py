import numpy as np
import torch
from torch.utils.data import IterableDataset
import torch_geometric.data as gd
from rdkit import Chem

from synflownet import ObjectProperties, LogScalar
from synflownet.data.replay_buffer import detach_and_cpu
from synflownet.models import bengio2021flow
from synflownet.envs.graph_building_env import Graph, GraphAction, GraphActionType
from synflownet.algo.graph_sampling import Sampler


class ReactionTask:

    def __init__(self, device):
        self.model = bengio2021flow.load_original_model()
        self.model = self.model.to(device)
        self.num_cond_dim = 32
        self.device = device

    def cond_info_to_logreward(self, cond_info, flat_reward):
        return LogScalar(flat_reward.squeeze().clamp(min=1e-30).log() * cond_info["beta"])

    def compute_obj_properties(self, mols):
        graphs = [bengio2021flow.mol2graph(m) for m in mols]
        batch = gd.Batch.from_data_list(graphs)
        batch.to(self.device)
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

    def reverse(self, g, action):
        if action.action is GraphActionType.AddFirstReactant:
            return GraphAction(GraphActionType.BckRemoveFirstReactant)
        elif action.action is GraphActionType.ReactUni:
            return GraphAction(GraphActionType.BckReactUni, rxn=action.rxn)
        elif action.action is GraphActionType.ReactBi:

            bck_a = GraphAction(GraphActionType.BckReactBi, rxn=action.rxn, bb=0)

            mol = self.ctx.get_mol(g)
            reaction = self.ctx.bimolecular_reactions[bck_a.rxn]
            products = reaction.run_reverse_reactants((mol,))
            products_smi = [Chem.MolToSmiles(p) for p in products]

            all_bbs = self.ctx.building_blocks
            if (products_smi[0] in all_bbs) and (products_smi[1] in all_bbs):
                return GraphAction(GraphActionType.BckReactBi, rxn=action.rxn, bb=1)
            else:
                return GraphAction(GraphActionType.BckReactBi, rxn=action.rxn, bb=0)

        elif action.action is GraphActionType.BckRemoveFirstReactant:
            return GraphAction(GraphActionType.AddFirstReactant)
        elif action.action is GraphActionType.BckReactUni:
            return GraphAction(GraphActionType.ReactUni, rxn=action.rxn)
        elif action.action is GraphActionType.BckReactBi:
            return GraphAction(GraphActionType.ReactBi, rxn=action.rxn, bb=action.bb)


class SynthesisSampler(Sampler):

    def __init__(self, ctx, env, device, max_len=None):

        self.ctx = ctx
        self.env = env
        self.device = device
        self.max_len = max_len if max_len is not None else 5

    def sample_from_model(self, model, n, cond_info, random_action_prob=0):

        data = [
            {"traj": [], "is_valid": True, "bck_a": [GraphAction(GraphActionType.Stop)], "is_sink": [], "bck_logprobs": []}
            for _ in range(n)
        ]

        graphs = [self.env.empty_graph() for _ in range(n)]
        done = [False] * n

        for t in range(self.max_len):

            torch_graphs = [self.ctx.graph_to_Data(g, traj_len=t) for i, g in enumerate(graphs) if not done[i]]
            nx_graphs = [g for i, g in enumerate(graphs) if not done[i]]
            not_done_mask = torch.tensor(done, device=self.device).logical_not()

            fwd_cat, *_ = model(self.ctx.collate(torch_graphs).to(self.device), cond_info[not_done_mask])
            actions = fwd_cat.sample(nx_graphs=nx_graphs, model=model, random_action_prob=random_action_prob)
            graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a, fwd=True) for g, a in zip(torch_graphs, actions)]

            for i, j in zip((k for k in range(n) if not done[k]), range(n)):

                data[i]["traj"].append((graphs[i], graph_actions[j]))

                if graph_actions[j].action is GraphActionType.Stop:

                    data[i]["bck_a"].append(GraphAction(GraphActionType.Stop))
                    data[i]["bck_logprobs"].append(torch.tensor([1.0], device=self.device).log())
                    data[i]["is_sink"].append(True)
                    done[i] = True

                else:

                    gp = self.env.step(graphs[i], graph_actions[j])

                    data[i]["bck_a"].append(self.env.reverse(gp, graph_actions[j]))  # TODO: time and put behind if statement if too slow

                    Chem.SanitizeMol(gp)
                    g = self.ctx.obj_to_graph(gp)

                    n_back = self.env.count_backward_transitions(g)  # TODO: time and put behind if statement if too slow
                    data[i]["bck_logprobs"].append(torch.tensor([1 / n_back] if n_back > 0 else [0.001],
                                                                device=self.device).log())

                    # in original implementation this is set to True for t = max_len-1
                    data[i]["is_sink"].append(False)
                    
                    if graph_actions[j].action in [GraphActionType.AddFirstReactant,
                                                   GraphActionType.ReactBi]:
                        data[i]["bbs"].append(graph_actions[j].bb)

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

            data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
            data[i]["is_sink"].append(True)
            data[i]["bck_logprobs"] = torch.stack(data[i]["bck_logprobs"]).reshape(-1)

        return data


class TrajectoryBalanceBase:

    def __init__(self, ctx, sampler, device, preference_strength=0):
        self.ctx = ctx
        self.sampler = sampler
        self.device = device
        self.preference_strength = preference_strength

    def create_training_data_from_own_samples(self, model, n, cond_info, random_action_prob=0.0):

        cond_info = cond_info.to(self.device)
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
        batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
        batch.log_bbs_cost = torch.log(torch.stack([t["bbs_cost"] for t in trajs]))
        batch.bck_actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a, fwd=False)
            for g, a in zip(torch_graphs, [i for tj in trajs for i in tj["bck_a"]])
        ]

        # TODO: try better methods of combining the reward
        batch.log_rewards -= batch.log_bbs_cost * self.preference_strength

        return batch

    def compute_batch_losses(self, _model, _batch):
        raise NotImplementedError()

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

            yield self.algo.construct_batch(detach_and_cpu(sum(iterator_outputs, [])))

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
                    trajs[i]["bbs_cost"] = sum(self.ctx.bbs_costs[bb] for bb in trajs[i]["bbs"])

                self.compute_properties(trajs)
                self.compute_log_rewards(trajs)

                yield trajs[:num_samples]

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