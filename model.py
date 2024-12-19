import torch
import torch.nn as nn
import torch.nn.functional as F

class NETWORK(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim, 
        hidden_dim,
        nhead=1,
        num_encoder_layers=1,
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        )
            

        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )

        self.derivatives_dim = input_dim
        self.probabilities_dim = 4*input_dim//2

        self.derivative_decoder = nn.Sequential(
            nn.Linear(hidden_dim, self.derivatives_dim), #3 output kinetic parameters for each gene
        )

        self.probabilities_decoder = nn.Linear(hidden_dim, self.probabilities_dim) #4 output probabilities for each gene

        self.v_u = None
        self.v_s = None
        self.v_u_pos = None
        self.v_s_pos = None
        self.pp = None
        self.nn = None
        self.pn = None
        self.np = None

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a sequence dimension
        z = self.encoder(x)
        z = z.squeeze(1)  # Remove the sequence dimension
        z_shared = self.shared_decoder(z)
        self.derivatives = self.derivative_decoder(z_shared)
        self.v_u_pos, self.v_s_pos = torch.split(self.derivatives, self.derivatives_dim // 2, dim=1)        
        v_u_neg = -1 * self.v_u_pos
        v_s_neg = -1 * self.v_s_pos
        p_sign = self.probabilities_decoder(z_shared)
        p_sign = p_sign.view(-1, self.probabilities_dim//4, 4)
        p_sign = F.softmax(p_sign, dim=-1)
        self.pp = p_sign[:,:,0]
        self.nn = p_sign[:,:,1]
        self.pn = p_sign[:,:,2]
        self.np = p_sign[:,:,3]
        unspliced, spliced = torch.split(x.squeeze(1), x.size(2) // 2, dim=1)

        self.v_u = self.v_u_pos * self.pp + v_u_neg * self.nn + self.v_u_pos * self.pn + v_u_neg * self.np
        self.v_s = self.v_s_pos * self.pp + v_s_neg * self.nn + v_s_neg * self.pn + self.v_s_pos * self.np

        unspliced_pred = unspliced + self.v_u
        spliced_pred = spliced + self.v_s

        self.prediction = torch.cat([unspliced_pred, spliced_pred], dim=1)

        self.out_dic = {
            "x" : x.squeeze(1),
            "pred" : self.prediction,
            "v_u" : self.v_u,
            "v_s" : self.v_s,
            "v_u_pos" : self.v_u_pos,
            "v_s_pos" : self.v_s_pos,
            "pp" : self.pp,
            "nn" : self.nn,
            "pn" : self.pn,
            "np" : self.np

        }

        return self.out_dic

    def heuristic_loss(
            self,
            adata, 
            x, 
            batch_indices,
            lambda1,
            lambda2,
            out_dic,
            device,
            K):

            x = out_dic["x"]
            prediction_nn = out_dic["pred"]

            reference_data = x #fetch the GE data of the samples in the batch 
            neighbor_indices = adata.uns["indices"][batch_indices,1:K] #fetch the nearest neighbors   
            neighbor_data_u = torch.from_numpy(adata.layers["Mu"][neighbor_indices]).to(device) 
            neighbor_data_s = torch.from_numpy(adata.layers["Ms"][neighbor_indices]).to(device)
            neighbor_data = torch.cat([neighbor_data_u, neighbor_data_s], dim=2) #fetch the GE data of the neighbors for each sample in the batch

            model_prediction_vector = prediction_nn - reference_data #compute the difference vector of the model prediction vs the input samples
            neighbor_prediction_vectors = neighbor_data - reference_data.unsqueeze(1) #compute the difference vector of the neighbor data vs the input samples

            # Normalize the vectors cell-wise
            model_prediction_vector_normalized = F.normalize(model_prediction_vector, p=2, dim=1)
            neighbor_prediction_vectors_normalized = F.normalize(neighbor_prediction_vectors, p=2, dim=2)

            # Calculate the norms of the normalized vectors
            model_prediction_vector_norms = torch.norm(model_prediction_vector_normalized, p=2, dim=1)
            neighbor_prediction_vectors_norms = torch.norm(neighbor_prediction_vectors_normalized, p=2, dim=2)
            
            # Assertions to ensure each vector is a unit vector, considering a small tolerance
            tolerance = 1e-4  # Adjust the tolerance if needed
            #assert torch.allclose(model_prediction_vector_norms, torch.ones_like(model_prediction_vector_norms), atol=tolerance), "Model prediction vectors are not properly normalized"
            #assert torch.allclose(neighbor_prediction_vectors_norms, torch.ones_like(neighbor_prediction_vectors_norms), atol=tolerance), "Neighbor prediction vectors are not properly normalized"

            cos_sim = F.cosine_similarity(neighbor_prediction_vectors_normalized, model_prediction_vector_normalized.unsqueeze(1), dim=-1)

            aggr, _ = cos_sim.max(dim=1)
            cell_loss = 1 - aggr 
            heuristic_loss = torch.mean(cell_loss) # compute the batch loss
            discrepancy_loss = 0 
            for p in ["pp", "nn", "pn", "np"]:
                discrepancy_loss += (torch.tensor(0.25, device=device).expand_as(out_dic[p]) - out_dic[p]) ** 2
            discrepancy_loss = (discrepancy_loss / 4).mean()

            weighted_heuristic_loss = lambda1 * heuristic_loss
            weighted_discrepancy_loss = lambda2 * discrepancy_loss

            total_loss = weighted_heuristic_loss + discrepancy_loss

            losses_dic = {
                 "heuristic_loss" : heuristic_loss,
                 "heuristic_loss_weighted" : weighted_heuristic_loss,
                 "cell_loss" : cell_loss,
                 "uniform_p_loss" : discrepancy_loss,
                 "uniform_p_loss_weighted" : weighted_discrepancy_loss,
                 "total_loss" : total_loss,
                 "batch_indices" : batch_indices,
            }

            return losses_dic
