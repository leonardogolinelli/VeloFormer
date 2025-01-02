import torch
import torch.nn as nn
import torch.nn.functional as F

class NETWORK(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim, 
        hidden_dim,
        emb_dim,
        num_genes,
        num_bins,
        nhead=1,
        num_encoder_layers=1,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(2 * num_genes, emb_dim)  # Move embedding here
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.input_dim = input_dim
        self.derivatives_dim = input_dim
        self.probabilities_dim = 4*input_dim//2

        self.shared_decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Softplus(),
            nn.Linear(emb_dim, 1),
            nn.Softplus(),
        )

        self.derivative_decoder = nn.Sequential(
            nn.Linear(input_dim,input_dim)
        )

        self.probabilities_decoder = nn.Sequential(
            nn.Linear(input_dim, 4*input_dim//2)
        )

    def forward(self, binned_indices, data):
        tokens = self.embeddings(binned_indices)  # Use embeddings within the model
        gene_embeddings = self.encoder(tokens)
        cell_embeddings = self.shared_decoder(gene_embeddings).squeeze(-1)
        derivatives = self.derivative_decoder(cell_embeddings)
        v_u_pos, v_s_pos = torch.split(derivatives, self.derivatives_dim // 2, dim=1)   
        v_u_neg = -1 * v_u_pos
        v_s_neg = -1 * v_s_pos
        batch_size = cell_embeddings.shape[0]
        p_sign = self.probabilities_decoder(cell_embeddings).reshape(batch_size, self.input_dim//2, 4)
        p_sign = F.softmax(p_sign, dim=-1)
        self.pp = p_sign[:,:,0]
        self.nn = p_sign[:,:,1]
        self.pn = p_sign[:,:,2]
        self.np = p_sign[:,:,3]
        v_u = v_u_pos * self.pp + v_u_neg * self.nn + v_u_pos * self.pn + v_u_neg * self.np
        v_s = v_s_pos * self.pp + v_s_neg * self.nn + v_s_neg * self.pn + v_s_pos * self.np

        unspliced, spliced = torch.split(data, data.size(1) // 2, dim=1)
        unspliced_pred = unspliced + v_u
        spliced_pred = spliced + v_s

        prediction = torch.cat([unspliced_pred, spliced_pred], dim=1)

        with torch.no_grad():
            v_u_max = torch.where(
                self.pp >= torch.max(torch.max(self.nn, self.pn), self.np), v_u_pos,
                torch.where(torch.max(self.nn, self.pn) >= self.np, v_u_neg,
                torch.where(self.pn >= self.np, v_u_pos, v_u_neg))
            )

            v_s_max = torch.where(
                self.pp >= torch.max(torch.max(self.nn, self.pn), self.np), v_s_pos,
                torch.where(torch.max(self.nn, self.pn) >= self.np, v_s_neg,
                torch.where(self.pn >= self.np, v_s_neg, v_s_pos))
            )

            # If positive probability is equal to negative probability, use expected value
            v_u_max = torch.where(self.pp == self.nn, v_u, v_u_max)
            v_s_max = torch.where(self.pp == self.nn, v_s, v_s_max)

        self.out_dic = {
            "tokens": tokens,
            "data" : data.squeeze(1),
            "pred" : prediction,
            "v_u" : v_u,
            "v_s" : v_s,
            "v_u_pos" : v_u_pos,
            "v_s_pos" : v_s_pos,
            "v_u_max" : v_u_max,
            "v_s_max" : v_s_max,
            "pp" : self.pp,
            "nn" : self.nn,
            "pn" : self.pn,
            "np" : self.np,
            "gene_embeddings" : gene_embeddings,
            "cell_embeddings" : cell_embeddings,
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

            x = out_dic["data"]
            #prediction_nn = out_dic["pred"]

            reference_data = x #fetch the GE data of the samples in the batch 
            neighbor_indices = adata.uns["indices"][batch_indices,1:K] #fetch the nearest neighbors   
            neighbor_data_u = torch.from_numpy(adata.layers["Mu"][neighbor_indices]).to(device) 
            neighbor_data_s = torch.from_numpy(adata.layers["Ms"][neighbor_indices]).to(device)
            neighbor_data = torch.cat([neighbor_data_u, neighbor_data_s], dim=2) #fetch the GE data of the neighbors for each sample in the batch

            #model_prediction_vector = prediction_nn - reference_data #compute the difference vector of the model prediction vs the input samples
            model_prediction_vector = torch.cat([out_dic["v_u"], out_dic["v_s"]], dim=1)
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
