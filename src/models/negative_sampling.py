import numpy as np
import torch


class FastNegativeSampling:
    def __init__(self, nb_nodes):
        self.nb_nodes = nb_nodes

    def __call__(self, edge_index, oversample=1.25):
        batch_size = edge_index.shape[1]
        el_hash = lambda x: x[0, :] + x[1, :] * self.nb_nodes

        el1d = el_hash(edge_index).cpu().numpy()
        neg = np.array([[], []])

        while neg.shape[1] < batch_size:
            maybe_neg = np.random.randint(
                0, self.nb_nodes, (2, int(batch_size * oversample))
            )  # generates a 2d matrix
            neg_hash = el_hash(maybe_neg)

            neg = np.concatenate([neg, maybe_neg[:, ~np.isin(neg_hash, el1d)]], axis=1)
        neg = torch.tensor(neg[:, :batch_size]).long()
        pos = edge_index
        return pos, neg

class SameSrcRandomDstNegativeSampling:
    def __init__(self, nb_nodes, device):
        self.nb_nodes = nb_nodes
        self.device = device

    def __call__(self, edge_index):
        neg_dst_nodes = torch.tensor(
            np.random.randint(
                0,
                self.nb_nodes,
                (
                    len(
                        edge_index[0],
                    )
                ),
            ),
            device=self.device,
        )
        neg_edge_index = torch.stack([edge_index[0], neg_dst_nodes], dim=0)
        return edge_index, neg_edge_index
