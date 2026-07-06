"""Adapter around PANCDR Encoder / GCN / ADV."""

import torch

from ModelTraining.model import ADV, Encoder, GCN


class PANCDRModelAdapter(object):
    def __init__(self, n_gene_features, atom_feature_dim, params, device):
        self.n_gene_features = n_gene_features
        self.atom_feature_dim = atom_feature_dim
        self.params = params
        self.device = torch.device(
            device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        )
        self.nz = params["nz"]
        self.d_dim = params["d_dim"]
        self.encoder = None
        self.gcn = None
        self.adv = None

    def build_models(self):
        self.encoder = Encoder(self.n_gene_features, self.nz, self.device)
        self.gcn = GCN(
            self.atom_feature_dim,
            [256, 256, 256],
            h_dims=[self.d_dim, self.nz + self.d_dim],
            use_dropout=False,
        )
        self.adv = ADV(self.nz)
        self.encoder.to(self.device)
        self.gcn.to(self.device)
        self.adv.to(self.device)
        return self

    def save_checkpoint(self, path, metadata):
        payload = {
            "EN_model": self.encoder.state_dict(),
            "GCN_model": self.gcn.state_dict(),
            "ADV_model": self.adv.state_dict(),
            "params": self.params,
            "n_gene_features": self.n_gene_features,
            "atom_feature_dim": self.atom_feature_dim,
        }
        payload.update(metadata)
        torch.save(payload, str(path))

    def load_checkpoint(self, path):
        ckpt = torch.load(str(path), map_location=self.device)
        self.params = ckpt["params"]
        self.nz = self.params["nz"]
        self.d_dim = self.params["d_dim"]
        self.build_models()
        self.encoder.load_state_dict(ckpt["EN_model"])
        self.gcn.load_state_dict(ckpt["GCN_model"])
        self.adv.load_state_dict(ckpt["ADV_model"])
        return ckpt

    def predict(self, drug_feat, drug_adj, gexpr):
        self.encoder.eval()
        self.gcn.eval()
        with torch.no_grad():
            z, mu, logvar = self.encoder(gexpr)
            scores = self.gcn(drug_feat, drug_adj, z)
        return scores.view(-1)

    def extract_latent(self, gexpr, use_mu=True):
        self.encoder.eval()
        with torch.no_grad():
            z, mu, logvar = self.encoder(gexpr)
        primary = mu if use_mu else z
        return {
            "encoder_z": z.cpu().numpy(),
            "encoder_mu": mu.cpu().numpy(),
            "encoder_logvar": logvar.cpu().numpy(),
            "primary": primary.cpu().numpy(),
        }
