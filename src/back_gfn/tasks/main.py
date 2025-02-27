import copy
from itertools import cycle
import pickle
import os
from shutil import rmtree
import time
import numpy as np
from scipy.spatial.distance import pdist
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

from synflownet.envs.synthesis_building_env import ReactionTemplateEnvContext

from back_gfn.tasks.n_model import GraphTransformerSynGFN
from back_gfn.tasks.algs import (TrajectoryBalanceUniform,
                                 TrajectoryBalanceFree,
                                 TrajectoryBalanceTLM,
                                 TrajectoryBalanceMaxEnt,
                                 TrajectoryBalancePrefDQN,
                                 TrajectoryBalancePrefREINFORCE,
                                 TrajectoryBalancePrefAC,
                                 TrajectoryBalancePrefPPO)
from back_gfn.tasks.util import ReactionTask, ReactionTemplateEnv, SynthesisSampler, DataSource


DEVICE = "cuda"
PRINT_EVERY = 1
CHECKPOINT_EVERY = 5
REWARD_THRESH = 0.9

# UNIFORM BACKWARD POLICY
config = {
    "preference_strength": 0,  # cost preference weighting in reward; TODO: test >0
    "algo": TrajectoryBalanceUniform,  # algorithm to determine policy loss
    "parameterise_p_b": False,  # whether to learn P_b
    "sample_backward": False,  # whether to sample trajectories using the backward policy
    "target_model": False,  # whether to maintain a backward model
    "gamma": None,  # discount factor on rewards for backward policy
    "entropy_loss_multiplier": None,  # weighting of entropy term in backward policy loss
    "target_update_freq": None,  # frequency to update target model with main weights
    "epsilon": None,  # clip proportion for PPO
    "outs": 1  # number of per graph outputs
}

# UNCONSTRAINED BACKWARD POLICY (TODO)
config["algo"] = TrajectoryBalanceFree
config["parameterise_p_b"] = True

# TRAJECTORY LIKELIHOOD MAXIMISATION BACKWARD POLICY (TODO)
#config["algo"] = TrajectoryBalanceTLM
#config["parameterise_p_b"] = True

# MAXIMUM ENTROPY BACKWARD POLICY (TODO)
#config["algo"] = TrajectoryBalanceMaxEnt
#config["parameterise_p_b"] = True
#config["outs"] = 2

# PREFERENCE BACKWARD POLICY WITH DQN (TODO)
#config["algo"] = TrajectoryBalancePrefDQN
#config["parameterise_p_b"] = config["sample_backward"] = config["target_model"] = True
#config["gamma"] = 1  # TODO: test <1
#config["entropy_loss_multiplier"] = 0  # TODO: test >0
#config["target_update_freq"] = 5  # TODO: ablation

# PREFERENCE BACKWARD POLICY WITH REINFORCE (TODO)
#config["algo"] = TrajectoryBalancePrefREINFORCE
#config["parameterise_p_b"] = config["sample_backward"] = True
#config["gamma"] = 1  # TODO: test <1
#config["entropy_loss_multiplier"] = 0  # TODO: test >0
#config["outs"] = 2

# PREFERENCE BACKWARD POLICY WITH ACTOR CRITIC (TODO)
#config["algo"] = TrajectoryBalancePrefAC
#config["parameterise_p_b"] = config["sample_backward"] = True
#config["gamma"] = 1  # TODO: test <1
#config["entropy_loss_multiplier"] = 0  # TODO: test >0
#config["outs"] = 2

# PREFERENCE BACKWARD POLICY WITH PPO (TODO)
#config["algo"] = TrajectoryBalancePrefPPO
#config["parameterise_p_b"] = config["sample_backward"] = config["target_model"] = True
#config["gamma"] = 1  # TODO: test <1
#config["entropy_loss_multiplier"] = 0  # TODO: test >0
#config["target_update_freq"] = 1  # TODO: ablation
#config["epsilon"] = 0.2  # TODO: ablation
#config["outs"] = 2


if __name__ == "__main__":

    rmtree("back_gfn/results", ignore_errors=True)
    os.mkdir("back_gfn/results")
    os.mkdir("back_gfn/results/models")
    os.mkdir("back_gfn/results/batches")

    rel_path = "/".join(os.path.abspath(__file__).split("/")[:-2])

    with open(rel_path + "/data/building_blocks/enamine_bbs.txt", "r") as file:
        building_blocks = file.read().splitlines()

    with open(rel_path + "/data/templates/hb.txt", "r") as file:
        reaction_templates = file.read().splitlines()

    with open(rel_path + "/data/building_blocks/precomputed_bb_masks_enamine_bbs.pkl", "rb") as f:
        precomputed_bb_masks = pickle.load(f)

    task = ReactionTask(DEVICE)  # for reward
    ctx = ReactionTemplateEnvContext(  # deals with molecules  # TODO: change line 55 to actually work
        num_cond_dim=task.num_cond_dim,
        building_blocks=building_blocks,
        reaction_templates=reaction_templates,
        precomputed_bb_masks=precomputed_bb_masks,
        fp_type="morgan_1024",
        fp_path=None,
        strict_bck_masking=False
    )
    env = ReactionTemplateEnv(ctx)  # for actions
    sampler = SynthesisSampler(ctx, env, DEVICE, config["parameterise_p_b"], not config["parameterise_p_b"])  # for sampling policies
    algo = config["algo"](ctx, sampler, DEVICE, config["preference_strength"],
                gamma=config["gamma"], entropy_loss_multiplier=config["entropy_loss_multiplier"], eps=config["epsilon"])  # for computing loss / helps making batches
    model = GraphTransformerSynGFN(ctx, do_bck=config["parameterise_p_b"], outs=config["outs"])
    target_model = GraphTransformerSynGFN(ctx, do_bck=config["parameterise_p_b"], outs=config["outs"]) if config["target_model"] else None

    Z_params = list(model._logZ.parameters())
    non_Z_params = [i for i in model.parameters() if all(id(i) != id(j) for j in Z_params)]

    opt = torch.optim.Adam(non_Z_params, 1e-4, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)
    opt_Z = torch.optim.Adam(Z_params, 1e-3, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda steps: 2 ** -(steps/200))
    lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(opt_Z, lambda steps: 2 ** -(steps/5_000))

    model.to(DEVICE)

    with torch.no_grad():
        train_src = DataSource(ctx, algo, task, DEVICE, use_replay=config["sample_backward"])  # gets training data inc rewards

        train_src.do_sample_model(model, 64)
        if config["sample_backward"]:
            train_src.do_sample_backward(model, 64)

        train_dl = torch.utils.data.DataLoader(train_src, batch_size=None)

    unique_scaffolds = set()
    num_unique_scaffolds = [0]
    num_mols_tested = [0]

    mean_tanimoto_distances = []

    full_results = [[] for __ in range(7)]

    start_time = time.time()

    for it, batch in zip(range(1, 5001), cycle(train_dl)):

        batch = batch.to(DEVICE)

        if config["target_model"] and it % config["target_update_freq"] == 0:
            # copy probably redundant here
            target_model.load_state_dict(copy.deepcopy(model.state_dict()))

        model.train()

        loss, info = algo.compute_batch_losses(model, batch, target_model)

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

            np.save("back_gfn/results/unique_scaffolds.npy", list(zip(num_mols_tested, num_unique_scaffolds)))

            fps = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2000)) for mol in mols])
            mean_tanimoto_dist = np.mean(pdist(fps, metric="jaccard"))
            mean_tanimoto_distances.append(mean_tanimoto_dist)

            np.save("back_gfn/results/tanimoto_distances.npy", mean_tanimoto_distances)

            total_time = time.time() - start_time
            start_time = time.time()

            if it % PRINT_EVERY == 0:
                print(f"iteration {it} : loss:{info['loss']:7.3f} " \
                    f"sampled_reward_avg:{rewards.mean().item():6.4f} " \
                    f"time_spent:{total_time:4.2f} " \
                    f"logZ:{info['log_z']:7.4f} " \
                    f"gen scaffolds: {len(scaffolds_above_thresh)} " \
                    f"unique scaffolds: {len(unique_scaffolds)} " \
                    f"Tanimoto: {mean_tanimoto_dist} " \
                    f"mean synth. cost: {sum([i.sum().item() for i in batch.bbs_costs])/len(batch.traj_lens):4.2f}")

            full_results[0].append(info["loss"])
            full_results[1].append(rewards.mean().item())
            full_results[2].append(total_time)
            full_results[3].append(info["log_z"])
            full_results[4].append(len(scaffolds_above_thresh))
            full_results[5].append(len(unique_scaffolds))
            full_results[6].append(sum([i.sum().item() for i in batch.bbs_costs])/len(batch.traj_lens))
            np.save("back_gfn/results/results.npy", full_results)

            if it % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(), f"back_gfn/results/models/{it}.pt")
                np.save(f"back_gfn/results/batches/{it}.npy", np.array([batch.cpu()], dtype=object), allow_pickle=True)
