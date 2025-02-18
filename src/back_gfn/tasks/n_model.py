from itertools import chain
import torch
import torch.nn as nn

from synflownet.envs.graph_building_env import GraphActionType, action_type_to_mask
from synflownet.envs.synthesis_building_env import ActionCategorical
from synflownet.models.graph_transformer import GraphTransformer

def mlp(n_in, n_hid, n_out, n_layer):
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), nn.LeakyReLU()] for i in range(n_layer + 1)], [])[:-1])

DEVICE = "cuda"

class GraphTransformerSynGFN(nn.Module):

    def __init__(self, env_ctx, do_bck=False, outs=1):
        super().__init__()

        num_emb = 128

        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=4,
            num_heads=2
        )

        self._action_type_to_num_inputs_outputs = {
            GraphActionType.Stop: (num_emb*2, 1),
            GraphActionType.AddNode: (num_emb, env_ctx.num_new_node_values),
            GraphActionType.SetNodeAttr: (num_emb, env_ctx.num_node_attr_logits),
            GraphActionType.AddEdge: (num_emb*2, 1),
            GraphActionType.SetEdgeAttr: (num_emb*2, env_ctx.num_edge_attr_logits),
            GraphActionType.RemoveNode: (num_emb, 1),
            GraphActionType.RemoveNodeAttr: (num_emb, env_ctx.num_node_attrs - 1),
            GraphActionType.RemoveEdge: (num_emb*2, 1),
            GraphActionType.RemoveEdgeAttr: (num_emb*2, env_ctx.num_edge_attrs),
            GraphActionType.AddFirstReactant: (num_emb*2, env_ctx.building_blocks_dim),
            GraphActionType.AddReactant: (num_emb*2 + env_ctx.num_bimolecular_rxns, env_ctx.building_blocks_dim),
            GraphActionType.ReactUni: (num_emb*2, env_ctx.num_unimolecular_rxns),
            GraphActionType.ReactBi: (num_emb*2, env_ctx.num_bimolecular_rxns),
            GraphActionType.BckReactUni: (num_emb*2, env_ctx.num_unimolecular_rxns),
            GraphActionType.BckReactBi: (num_emb*2, env_ctx.num_bimolecular_rxns),
            GraphActionType.BckRemoveFirstReactant: (num_emb*2, 1)
        }

        self._action_type_to_graph_part = {
            GraphActionType.Stop: "graph",
            GraphActionType.AddNode: "node",
            GraphActionType.SetNodeAttr: "node",
            GraphActionType.AddEdge: "non_edge",
            GraphActionType.SetEdgeAttr: "edge",
            GraphActionType.RemoveNode: "node",
            GraphActionType.RemoveNodeAttr: "node",
            GraphActionType.RemoveEdge: "edge",
            GraphActionType.RemoveEdgeAttr: "edge",
            GraphActionType.AddFirstReactant: "graph",
            GraphActionType.ReactUni: "graph",
            GraphActionType.ReactBi: "graph",
            GraphActionType.AddReactant: "graph",
            GraphActionType.BckReactUni: "graph",
            GraphActionType.BckReactBi: "graph",
            GraphActionType.BckRemoveFirstReactant: "graph",
        }

        self._graph_part_to_key = {
            "graph": None,
            "node": "x",
            "non_edge": "non_edge_index",
            "edge": "edge_index",
        }

        self._action_type_to_key = {
            at: self._graph_part_to_key[gp] for at, gp in self._action_type_to_graph_part.items()
        }

        mlps = {}
        atypes = chain(env_ctx.action_type_order, env_ctx.bck_action_type_order) if do_bck \
            else env_ctx.action_type_order
        for atype in atypes:
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, 0)
        self.mlps = nn.ModuleDict(mlps)

        self.do_bck = do_bck

        self.action_type_order = env_ctx.action_type_order
        self.bck_action_type_order = env_ctx.bck_action_type_order

        self.emb2graph_out = mlp(num_emb*2, num_emb, outs, 0)
        self._logZ = mlp(max(1, env_ctx.num_cond_dim), num_emb*2, 1, 2)

        self._ae = [env_ctx.building_blocks_embs.to(DEVICE)]

    def _make_cat(self, g, emb, action_types, fwd=True):

        raw_logits = []
        for t in action_types:

            mlp_outputs = self.mlps[t.cname](emb[self._action_type_to_graph_part[t]])
            if t in [GraphActionType.AddFirstReactant, GraphActionType.AddReactant]:
                mlp_outputs = torch.matmul(mlp_outputs, self._ae[0].T)

            raw_logits.append(mlp_outputs)

        return ActionCategorical(
            g,
            emb["graph"],
            raw_logits=raw_logits,
            keys=[self._action_type_to_key[t] for t in action_types if self._action_type_to_key[t]],
            action_masks=[action_type_to_mask(t, g) for t in action_types],
            types=action_types,
            fwd=fwd,
        )

    def forward(self, g, cond):

        node_embeddings, graph_embeddings = self.transf(g, cond)

        if hasattr(g, "non_edge_index"):
            ne_row, ne_col = g.non_edge_index
            non_edge_embeddings = torch.cat([node_embeddings[ne_row], node_embeddings[ne_col]], 1)
        else:
            non_edge_embeddings = None

        e_row, e_col = g.edge_index[:, ::2]
        edge_embeddings = torch.cat([node_embeddings[e_row], node_embeddings[e_col]], 1)

        emb = {
            "graph": graph_embeddings,
            "node": node_embeddings,
            "edge": edge_embeddings,
            "non_edge": non_edge_embeddings,
        }

        graph_out = self.emb2graph_out(graph_embeddings)

        action_type_order = [x for x in self.action_type_order if x != GraphActionType.AddReactant]
        bck_action_type_order = self.bck_action_type_order

        fwd_cat = self._make_cat(g, emb, action_type_order, fwd=True)

        if self.do_bck:
            bck_cat = self._make_cat(g, emb, bck_action_type_order, fwd=False)
            return fwd_cat, bck_cat, graph_out

        return fwd_cat, graph_out

    def logZ(self, cond_info):
        return self._logZ(cond_info)

    def call_mlp(self, mlp_name, input):

        mlp_outputs = self.mlps[mlp_name](input)
        if mlp_name == GraphActionType.AddReactant.cname:
            mlp_outputs = torch.matmul(mlp_outputs, self._ae[0].T)

        return mlp_outputs
