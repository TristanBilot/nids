import numpy as np
import torch
import torch.nn as nn
from libauc.losses import APLoss

from .decoders import InnerProductDecoder
from .encoders import (
    EULER_GCN,
    VAE,
    Autoencoder,
    GINENet,
    FocusEncoder,
    Argus_LANL,
    Argus_OPTC,
    GraphAttentionEmbedding,
    GraphSAGE,
    GIN,
)
from .recurrent import GRU, LSTM


class Model(nn.Module):
    def __init__(
        self,
        x_dim,
        nb_nodes,
        nb_edge_feats,
        device,
        gnn_flow,
        node_dropout,
        hid_node_emb,
        out_node_emb,
        use_recurrent,
        use_vae,
        use_skip_connection,
        use_recon_loss,
        use_contra_loss,
        lstm_or_gru,
        encoder,
        use_only_autoencoder,
        dataset,
        use_fast_neg_sampling,
        use_without_tanh,
        use_with_epsilon,
        use_ap_loss,
    ):
        super().__init__()

        self.nb_nodes = nb_nodes
        self.nb_edge_feats = nb_edge_feats
        self.device = device
        self.use_recurrent = use_recurrent
        self.use_skip_connection = use_skip_connection
        self._encoder = encoder
        self.use_recon_loss = use_recon_loss
        self.use_contra_loss = use_contra_loss
        self.lstm_or_gru = lstm_or_gru
        self.use_only_autoencoder = use_only_autoencoder
        self.dataset = dataset
        self.use_fast_neg_sampling = use_fast_neg_sampling
        self.use_ap_loss = use_ap_loss

        hid = hid_node_emb
        out = out_node_emb

        # Encoder
        if encoder == "FOCUS":
            self.encoder = FocusEncoder(
                x_dim=x_dim,
                h_dim=hid,
                z_dim=out,
                edge_dim=nb_edge_feats,
                node_dropout=node_dropout,
                flow=gnn_flow,
                without_tanh=use_without_tanh,
                with_epsilon=use_with_epsilon,
            )
        elif encoder == "GINE":
            self.encoder = GINENet(
                x_dim, hid, out, nb_edge_feats, node_dropout=node_dropout
            )
        elif encoder == "EULER":
            self.encoder = EULER_GCN(x_dim, hid, out, node_dropout=node_dropout)
        elif encoder == "SAGE":
            self.encoder = GraphSAGE(x_dim, hid, out, node_dropout=node_dropout)
        elif encoder == "Argus_LANL":
            self.encoder = Argus_LANL(x_dim, hid, out, nb_edge_feats, device)
        elif encoder == "Argus_OPTC":
            self.encoder = Argus_OPTC(x_dim, hid, out, nb_edge_feats, device)
        elif encoder == "Transformer":
            self.encoder = GraphAttentionEmbedding(x_dim, hid, out, nb_edge_feats, node_dropout=node_dropout)
        elif encoder == "GIN":
            self.encoder = GIN(x_dim, hid, out, node_dropout=node_dropout)
        else:
            raise ValueError(f"Invalid encoder {encoder}")

        # Temporal encoder (recurrent)
        if self.lstm_or_gru == "lstm":
            self.recurrent = LSTM(out, hid, out, device)
        else:
            self.recurrent = GRU(out, hid, out, device)

        # Decoder
        self.inner_product_decoder = InnerProductDecoder(node_dropout)

        ae_hidden = out // 2
        self.ae = (
            VAE(input_dim=out, hidden_dim=ae_hidden, latent_dim=out)
            if use_vae
            else Autoencoder(input_dim=out, hidden_dim=ae_hidden)
        )

        skip_in_shape = 2 * out + nb_edge_feats if use_skip_connection else 2 * out
        self.skip_linear = nn.Sequential(
            nn.Linear(skip_in_shape, hid),
            nn.ReLU(),
            nn.Linear(hid, out),
        )

        if self.use_skip_connection:
            self.skip_gate = nn.Sequential(
                nn.Linear(skip_in_shape, nb_edge_feats), nn.Sigmoid()
            )

        if self.use_only_autoencoder:
            self.ae_only_edge_features = Autoencoder(
                input_dim=nb_edge_feats, hidden_dim=nb_edge_feats // 2
            )
            self.skip_gate_only_edge_features = nn.Sequential(
                nn.Linear(nb_edge_feats, nb_edge_feats), nn.Sigmoid()
            )

        self.mse_mean = nn.MSELoss(reduction="mean")
        self.mse_scores = nn.MSELoss(reduction="none")
        
        if self.use_ap_loss:
            if dataset == 'OPTC':
                self.ap_loss = APLoss(pos_len=30900604, margin=0.8, gamma=0.1, surrogate_loss='squared', device=device)
            elif dataset == 'LANL':
                self.ap_loss = APLoss(pos_len=70519, margin=0.8, gamma=0.01, surrogate_loss='squared', device=device)


    def forward(self, data, h0=None, aug_coef=1):
        if self.use_only_autoencoder:
            gate = self.skip_gate_only_edge_features(data.edge_attr)
            e_feats_hat = self.ae_only_edge_features(gate * data.edge_attr)
            return self.mse_mean(e_feats_hat, data.edge_attr)

        # Encoding
        h_n = self._embed_nodes(data.x, data.edge_index, data.edge_attr)
        h = self._embed_edges(h_n, data.edge_index, data.edge_attr)

        # Decoding
        loss = torch.tensor(0.0, requires_grad=True)
        if self.use_recon_loss:
            h_ae = self.ae.encoder(h)
            h_hat = self.ae.decoder(h_ae)
            loss = loss + self.mse_mean(h_hat, h)

        if self.use_contra_loss:
            neg_edge_index = self._get_negative_samples(data.edge_index)
            pos_scores = self.inner_product_decoder(
                src=data.edge_index[0, :], dst=data.edge_index[1, :], h=h_n
            )
            neg_scores = self.inner_product_decoder(
                src=neg_edge_index[0, :], dst=neg_edge_index[1, :], h=h_n
            )

            contra_loss = self.bce(pos_scores, neg_scores)
            loss = loss + contra_loss
            
        if self.use_ap_loss:
            neg_edge_index = self._get_negative_samples(data.edge_index)
            pos_scores = self.inner_product_decoder(
                src=data.edge_index[0, :], dst=data.edge_index[1, :], h=h_n
            )
            neg_scores = self.inner_product_decoder(
                src=neg_edge_index[0, :], dst=neg_edge_index[1, :], h=h_n
            )
            t_index = torch.arange(0, pos_scores.shape[0], dtype=torch.int64).to(self.device).detach()

            ap_loss = self.ap_loss(
                torch.cat((pos_scores, neg_scores), 0),
                torch.cat((torch.ones(pos_scores.shape[0]),torch.zeros(neg_scores.shape[0])), 0).to(self.device).detach(),
                t_index)
            loss = loss + ap_loss

        return loss

    def _embed_nodes(self, x, edge_index, edge_attr):
        h = self.encoder(x, edge_index, edge_attr)

        if self.use_recurrent:
            h = self.recurrent(h)

        return h

    def _embed_edges(self, h, edge_index, edge_attr):
        # edge embeddings
        h = torch.cat([h[edge_index[0, :]], h[edge_index[1, :]]], dim=1)

        # add skip connections
        if self.use_skip_connection:
            concat = torch.cat([h, edge_attr], dim=1)
            gate_values = self.skip_gate(concat)
            h = torch.cat([h, gate_values * edge_attr], dim=1)

        h = self.skip_linear(h)
        return h

    def inference(self, data):
        with torch.no_grad():
            if self.use_only_autoencoder:
                gate = self.skip_gate_only_edge_features(data.edge_attr)
                e_feats_hat = self.ae_only_edge_features(gate * data.edge_attr)
                scores = self.mse_scores(e_feats_hat, data.edge_attr)
                scores = torch.sum(scores, dim=1)
                return scores

            h_n = self._embed_nodes(data.x, data.edge_index, data.edge_attr)
            h = self._embed_edges(h_n, data.edge_index, data.edge_attr)

            scores = torch.zeros_like(data.y, dtype=torch.float)

            if self.use_contra_loss:
                contra_scores = self.inner_product_decoder(
                    src=data.edge_index[0, :], dst=data.edge_index[1, :], h=h_n
                )
                scores += 1 - contra_scores

            if self.use_recon_loss:  # The higher, the more anomalous
                h_hat = self.ae(h)
                recon_scores = self.mse_scores(h_hat, h)
                recon_scores = torch.sum(recon_scores, dim=1)
                scores += recon_scores
                
            if self.use_ap_loss:
                pos_scores = self.inner_product_decoder(
                    src=data.edge_index[0, :], dst=data.edge_index[1, :], h=h_n
                )
                scores += 1 - pos_scores

            return scores

    def bce(self, positive, negative):
        EPS = 1e-6
        pos_loss = -torch.log(positive + EPS).mean()
        neg_loss = -torch.log(1 - negative + EPS).mean()
        loss = (pos_loss + neg_loss) * 0.5
        return loss

    def reset(self):
        self.recurrent.reset_state()

    def detach(self):
        self.recurrent.detach_state()

    def _fast_negative_sampling(self, edge_index, aug_coef=1.3):
        el_hash = lambda x: x[0, :] + x[1, :] * self.nb_nodes
        el1d = el_hash(edge_index).detach().cpu().numpy()

        maybe_neg = np.random.randint(
            0, self.nb_nodes, (2, int(len(edge_index[0]) * aug_coef))
        )
        maybe_neg = maybe_neg[:, maybe_neg[0] != maybe_neg[1]]  # remove self-loops
        neg_hash = el_hash(maybe_neg)

        neg_samples = maybe_neg[:, ~np.in1d(neg_hash, el1d)]
        return torch.tensor(neg_samples).to(self.device)

    def _get_negative_samples(self, edge_index):
        if self.use_fast_neg_sampling:
            neg_edge_index = self._fast_negative_sampling(edge_index, aug_coef=1.0)
        else:
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

        return neg_edge_index
