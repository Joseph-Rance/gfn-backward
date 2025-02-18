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
from back_gfn.tasks.algs import TrajectoryBalanceUniform
from back_gfn.tasks.util import ReactionTask, ReactionTemplateEnv, SynthesisSampler, DataSource


DEVICE = "cuda"
PRINT_EVERY = 1
REWARD_THRESH = 0.9
PREFERENCE_STRENGTH = 0  # TODO: test with >0
ALGO = TrajectoryBalanceUniform  # TODO: test the other algorithms
PARAMETERISE_P_B = False
OUTS = 1  # 2 for MaxEnt
CHECKPOINT_EVERY = 5
SAMPLE_BACKWARD = False  # True for REINFORCE

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
    sampler = SynthesisSampler(ctx, env, DEVICE)  # for sampling policies
    algo = ALGO(ctx, sampler, DEVICE, PREFERENCE_STRENGTH)  # for computing loss / helps making batches
    model = GraphTransformerSynGFN(ctx, do_bck=PARAMETERISE_P_B, outs=OUTS)

    Z_params = list(model._logZ.parameters())
    non_Z_params = [i for i in model.parameters() if all(id(i) != id(j) for j in Z_params)]

    opt = torch.optim.Adam(non_Z_params, 1e-4, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)
    opt_Z = torch.optim.Adam(Z_params, 1e-3, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda steps: 2 ** -(steps/200))
    lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(opt_Z, lambda steps: 2 ** -(steps/5_000))

    model.to(DEVICE)

    with torch.no_grad():
        train_src = DataSource(ctx, algo, task, DEVICE, use_replay=SAMPLE_BACKWARD)  # gets training data inc rewards

        train_src.do_sample_model(model, 64)
        if SAMPLE_BACKWARD:
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
                    f"mean synth. cost: {batch.log_bbs_cost.exp().mean().item()}")

            full_results[0].append(info["loss"])
            full_results[1].append(rewards.mean().item())
            full_results[2].append(total_time)
            full_results[3].append(info["log_z"])
            full_results[4].append(len(scaffolds_above_thresh))
            full_results[5].append(len(unique_scaffolds))
            full_results[6].append(batch.log_bbs_cost.exp().mean().item())
            np.save("back_gfn/results/results.npy", full_results)

            if it % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(), f"back_gfn/results/models/{it}.pt")
                np.save(f"back_gfn/results/batches/{it}.npy", np.array([batch], dtype=object), allow_pickle=True)

    # TODO: does this fix memory leak?
    del batch
