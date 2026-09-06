# multimodal_model_with_ppi.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np


import torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class EdgeWeightedGATv2Layer(nn.Module):
    """
    [Fixed version] GATv2 layer with edge weights.
    Fix: Expand the alpha dimensions to resolve the broadcasting error.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads=4,
        edge_dim=1,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        concat=True,
        allow_zero_in_degree=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.concat = concat
        self.allow_zero_in_degree = allow_zero_in_degree

        self.fc_src = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.fc_dst = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        
        if edge_dim > 0:
            self.fc_edge = nn.Linear(edge_dim, num_heads * out_dim, bias=False)
        else:
            self.fc_edge = None
        
        self.attn = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))

        # Residual connection
        total_out_dim = out_dim * num_heads if concat else out_dim
        self.res_fc = None
        if in_dim != total_out_dim:
            self.res_fc = nn.Linear(in_dim, total_out_dim, bias=False)
            nn.init.xavier_uniform_(self.res_fc.weight)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_src.weight)
        nn.init.xavier_uniform_(self.fc_dst.weight)
        if self.fc_edge is not None:
            nn.init.xavier_uniform_(self.fc_edge.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, g, h, edge_weight):
        with g.local_scope():
            if not self.allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    pass

            h_in = h
            h = self.feat_drop(h)
            
            # [N, H, D]
            feat_src = self.fc_src(h).view(-1, self.num_heads, self.out_dim)
            feat_dst = self.fc_dst(h).view(-1, self.num_heads, self.out_dim)

            g.srcdata['h_src'] = feat_src
            g.dstdata['h_dst'] = feat_dst

            # Edge features
            if edge_weight.dim() == 1:
                edge_weight = edge_weight.unsqueeze(-1)
            
            if self.fc_edge is not None:
                ew_feat = self.fc_edge(edge_weight).view(-1, self.num_heads, self.out_dim)
            else:
                ew_feat = torch.zeros(g.num_edges(), self.num_heads, self.out_dim, device=h.device)
            g.edata['ew_feat'] = ew_feat

            # Attention computation
            def edge_attention(edges):
                z = edges.src['h_src'] + edges.dst['h_dst'] + edges.data['ew_feat']
                z = self.leaky_relu(z)
                e = (z * self.attn).sum(dim=-1) # [E, H]
                return {'e': e}

            g.apply_edges(edge_attention)
            
            e = g.edata.pop('e')
            alpha = edge_softmax(g, e)     # [E, H]
            alpha = self.attn_drop(alpha)
            
            # 🔥🔥🔥 Critical fix 🔥🔥🔥
            # Change [E, H] to [E, H, 1] for broadcast multiplication with h_src [E, H, D]
            g.edata['alpha'] = alpha.unsqueeze(-1)

            # Message passing
            g.update_all(
                fn.u_mul_e('h_src', 'alpha', 'm'),
                fn.sum('m', 'h_out')
            )

            h_out = g.ndata['h_out'] 

            if self.concat:
                h_out = h_out.reshape(-1, self.num_heads * self.out_dim)
            else:
                h_out = h_out.mean(dim=1)

            # Residual connection
            if self.res_fc is not None:
                h_out = h_out + self.res_fc(h_in)
            else:
                h_out = h_out + h_in

            return h_out


class TwoLayerPPIGATv2(nn.Module):
    """
    PPI model based on GATv2 and LayerNorm.
    """
    def __init__(
        self,
        in_dim,          # 1024
        hid_dim=256,     # Hidden
        gat_out_dim=256, # Output
        num_classes=100, 
        num_heads_1=4,
        num_heads_2=4,
        edge_dim=1,
        feat_drop=0.1,
        attn_drop=0.1,
    ):
        super().__init__()

        # Layer 1: GATv2
        self.gat1 = EdgeWeightedGATv2Layer(
            in_dim=in_dim,
            out_dim=hid_dim,
            num_heads=num_heads_1,
            edge_dim=edge_dim,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            concat=True,
        )
        # Norm 1
        self.norm1 = nn.LayerNorm(hid_dim * num_heads_1)

        # Layer 2: GATv2
        self.gat2 = EdgeWeightedGATv2Layer(
            in_dim=hid_dim * num_heads_1,
            out_dim=gat_out_dim,
            num_heads=num_heads_2,
            edge_dim=edge_dim,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            concat=False,
        )
        # Norm 2
        self.norm2 = nn.LayerNorm(gat_out_dim)

 

    def forward(self, g):
        x = g.ndata['x']
        ew = g.edata['weight']

        # Layer 1
        h = self.gat1(g, x, ew)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=0.1, training=self.training)

        # Layer 2
        h = self.gat2(g, h, ew)
        h = self.norm2(h)

        u, v = g.edges()                   # [E]
        is_real_edge = (u != v).float()    # Self-loop = 0
        real_deg = torch.zeros(g.num_nodes(), device=h.device)
        real_deg.index_add_(0, v, is_real_edge)  # Sum of in-degrees
        has_ppi_node = (real_deg > 0)      # [N] bool

        # Select only center nodes
        is_center = g.ndata['is_center']   # [N] bool
        h_center_ppi = h[is_center]        # [B, gat_out_dim]
        ppi_mask = has_ppi_node[is_center].float().unsqueeze(1)  # [B,1]

        return h_center_ppi, ppi_mask






class inter_model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(inter_model, self).__init__()
        
        self.embedding_layer = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)

        self.linearLayer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU()
        )
    
    def forward(self, inter_feature):
        inter_feature = F.relu(self.embedding_layer(*inter_feature))
        inter_feature = self.linearLayer(inter_feature)
        
        return inter_feature



class BiDirectionalTransformerBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, head=1):
        super(BiDirectionalTransformerBlock, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim

        # =========================================================
        # Stream A: Structure Queries Domain (extract structure-related domain information)
        # Q=Residue, K=Inter, V=Inter (note that V is Inter here)
        # =========================================================
        self.stream_a_q = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_a_k = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_a_v = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        # =========================================================
        # Stream B: Domain Queries Structure (extract structural information matching the domain)
        # Q=Inter, K=Residue, V=Residue (note that V is Residue here)
        # =========================================================
        self.stream_b_q = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_b_k = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_b_v = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        # =========================================================
        # Fusion layer (fuse bidirectional features)
        # Input dimension: (head * hidden) * 2 (because there are two streams)
        # =========================================================
        self.fusion_trans = nn.Linear((hidden_dim * head) * 2, hidden_dim, bias=False)

        # Feed Forward Network (FFN)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # LayerNorms
        self.layernorm1 = nn.LayerNorm(in_dim)
        self.layernorm2 = nn.LayerNorm(in_dim)
    
    def forward(self, g, residue_h, inter_h):
        # residue_h: Structural features (N_nodes, in_dim)
        # inter_h:   Domain features (N_nodes, in_dim)
        # Note: inter_h is assumed to have been broadcast/expanded to the same shape as residue_h
        self.attention_scores = {
            'stream_a': [], # Structure actively queries the sequence (the key advantage)
            'stream_b': []  # Sequence queries the structure (for validation)
        }
        output_stream_a = []
        output_stream_b = []

        for i in range(self.head):
            # -------------------------------------------------
            # Stream A computation: Structure -> Domain
            # Logic: "I am this structure; provide matching domain information"
            # -------------------------------------------------
            qa = self.stream_a_q[i](residue_h) # Q = Structure
            ka = self.stream_a_k[i](inter_h)   # K = Domain
            va = self.stream_a_v[i](inter_h)   # V = Domain (retrieve domain content)
            
            # Compute attention scores
            att_a = torch.sum(torch.mul(qa, ka) / torch.sqrt(torch.tensor(float(self.hidden_dim))), dim=1, keepdim=True)
            
            # Apply softmax over the graph structure (local normalization)
            with g.local_scope():
                g.ndata['att_a'] = att_a.reshape(-1)
                alpha_a = dgl.softmax_nodes(g, 'att_a').reshape((va.size(0), 1))
                self.attention_scores['stream_a'].append(alpha_a.detach().cpu())
                out_a = va * alpha_a # Weighted domain features
            output_stream_a.append(out_a)

            # -------------------------------------------------
            # Stream B computation: Domain -> Structure
            # Logic: "I am this domain; highlight the matching structural region"
            # -------------------------------------------------
            qb = self.stream_b_q[i](inter_h)   # Q = Domain
            kb = self.stream_b_k[i](residue_h) # K = Structure
            vb = self.stream_b_v[i](residue_h) # V = Structure (retrieve structural content)
            
            att_b = torch.sum(torch.mul(qb, kb) / torch.sqrt(torch.tensor(float(self.hidden_dim))), dim=1, keepdim=True)
            
            with g.local_scope():
                g.ndata['att_b'] = att_b.reshape(-1)
                alpha_b = dgl.softmax_nodes(g, 'att_b').reshape((vb.size(0), 1))
                self.attention_scores['stream_b'].append(alpha_b.detach().cpu())
                out_b = vb * alpha_b # Weighted structural features
            output_stream_b.append(out_b)

        # -------------------------------------------------
        # Fusion and output
        # -------------------------------------------------
        
        # 1. Concatenate multi-head results
        multi_out_a = torch.cat(output_stream_a, dim=1) # [N, head*hidden]
        multi_out_b = torch.cat(output_stream_b, dim=1) # [N, head*hidden]
        
        # 2. Concatenate bidirectional results: [Structure-Aware-Domain, Domain-Aware-Structure]
        combined = torch.cat([multi_out_a, multi_out_b], dim=1) # [N, 2*head*hidden]
        
        # 3. Fuse linearly and reduce dimensionality
        fused_h = self.fusion_trans(combined) # [N, hidden]

        # 4. Residual connection + LayerNorm (retain original structural features as the base)
        fused_h = self.layernorm1(fused_h + residue_h)

        # 5. FFN + Residual
        final_output = self.layernorm2(self.ff(fused_h) + fused_h)

        return final_output

# class GCN(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, head):
#         super(GCN, self).__init__()
#         self.dropout = nn.Dropout(0.3)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
        
#         self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
#         self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        
#         # self.transformer_block = transformer_block(hidden_dim, hidden_dim, head)
#         self.transformer_block = BiDirectionalTransformerBlock(hidden_dim, hidden_dim, head)

#     def forward(self, g, h, inter_f):
        
#         with g.local_scope():
#             g.ndata['h'] = h
#             init_avg_h = dgl.mean_nodes(g, 'h')

#         pre = h
#         h = self.bn1(h)
#         h = pre + self.dropout(F.relu(self.conv1(g, h))) # , edge_weight=ew

#         pre = h
#         h = self.bn2(h)
#         h = pre + self.dropout(F.relu(self.conv2(g, h)))

#         with g.local_scope():
#             g.ndata['inter'] = dgl.broadcast_nodes(g, inter_f)
#             residue_h = h
#             inter_h = g.ndata['inter']
#             hg = self.transformer_block(g, residue_h, inter_h)
#             g.ndata['output'] = hg
#             readout = dgl.sum_nodes(g, "output")
#             return readout, init_avg_h
        

class GCN(nn.Module):    # Ablation
    def __init__(self, in_dim, hidden_dim, n_classes, head, use_transformer=True): # 🔥 New switch
        super(GCN, self).__init__()
        self.use_transformer = use_transformer 
        
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        
        # 🟢 Key ablation detail: Do not initialize the Transformer when it is disabled
        if self.use_transformer:
            self.transformer_block = BiDirectionalTransformerBlock(hidden_dim, hidden_dim, head)
        else:
            self.transformer_block = None

    def forward(self, g, h, inter_f):
            # ✅ First, compute and store the mean of the initial features (init_avg_h)
            # This must happen first; otherwise h changes after convolution, or g does not contain 'h'
            with g.local_scope():
                g.ndata['h_init'] = h  # Store input h in the graph as 'h_init'
                init_avg_h = dgl.mean_nodes(g, 'h_init') # Compute the mean

            # ... (keep the intermediate convolution layers unchanged) ...
            pre = h
            h = self.bn1(h)
            h = pre + self.dropout(F.relu(self.conv1(g, h)))

            pre = h
            h = self.bn2(h)
            h = pre + self.dropout(F.relu(self.conv2(g, h)))

            # ... (interaction module / ablation branch) ...
            if self.use_transformer and inter_f is not None:
                # Baseline: With interaction
                with g.local_scope():
                    g.ndata['inter'] = dgl.broadcast_nodes(g, inter_f)
                    residue_h = h
                    inter_h = g.ndata['inter']
                    hg = self.transformer_block(g, residue_h, inter_h)
                    g.ndata['output'] = hg
                    readout = dgl.sum_nodes(g, "output")
                    
                    # ✅ Return the previously computed init_avg_h instead of calling mean_nodes here
                    return readout, init_avg_h 
            else:
                # 🔥 Ablation: No interaction (this was the source of the error)
                with g.local_scope():
                    g.ndata['h_final'] = h
                    readout = dgl.sum_nodes(g, "h_final")
                    
                    # ✅ Fix: Return the init_avg_h computed above directly
                    # Do not call dgl.mean_nodes(g, 'h') because g does not contain 'h' at this point
                    return readout, init_avg_h



# class MaskedGatedFusion(nn.Module):
#     """
#     [Improved version] Gated fusion with a PPI mask
#     Logic:
#     1. Enforced mask: If PPI is unavailable, force PPI features to zero to prevent noise.
#     2. Residual fusion: Base + Gate * PPI. Base remains primary, while PPI provides supplementary information.
#     """
#     def __init__(self, dim_base, dim_ppi, dim_fused):
#         super().__init__()
#         self.proj_base = nn.Linear(dim_base, dim_fused)
#         self.proj_ppi  = nn.Linear(dim_ppi, dim_fused)
        
#         # Gate network: Determines the importance of PPI features
#         # Input is Base + PPI; output is a coefficient from 0 to 1
#         self.gate_layer = nn.Sequential(
#             nn.Linear(2 * dim_fused, dim_fused),
#             nn.Sigmoid()
#         )
        
#         # Normalize to accelerate convergence
#         self.norm = nn.LayerNorm(dim_fused)
#         self.activation = nn.ReLU()

#     def forward(self, x_base, x_ppi, ppi_mask):
#         """
#         x_base: [B, D_base]
#         x_ppi : [B, D_ppi]
#         ppi_mask: [B, 1] (1=real PPI available, 0=unavailable)
#         """
#         # 1. Project to the same dimension
#         xb = self.activation(self.proj_base(x_base))  # [B, Df]
#         xp = self.activation(self.proj_ppi(x_ppi))    # [B, Df]

#         # 2. 🔥 Core fix: Enforced masking
#         # xp has a value only when ppi_mask=1; otherwise it is forced to zero
#         xp = xp * ppi_mask.expand_as(xp)

#         # 3. Compute the gate (PPI importance)
#         # Only the interaction between xb and xp needs to be considered
#         combined = torch.cat([xb, xp], dim=-1)
#         gate = self.gate_layer(combined) # [B, Df]

#         # 4. 🔥 Fusion method: Residual complement
#         # Logic: Base features + (importance * PPI features)
#         # Thus, when mask=0, xp is zero and the result is pure xb regardless of the gate, with no noise
#         h_fused = xb + gate * xp
        
#         return self.norm(h_fused)

class MaskedGatedFusion(nn.Module):
    """
    [Improved version] Gated fusion with a PPI mask + cold-start initialization.
    Logic:
    1. Enforced mask: If PPI is unavailable, force PPI features to zero.
    2. Residual fusion: Base + Gate * PPI.
    3. Cold start: Force the gate closed initially to prevent PPI noise from disrupting backbone features.
    """
    def __init__(self, dim_base, dim_ppi, dim_fused):
        super().__init__()
        self.proj_base = nn.Linear(dim_base, dim_fused)
        self.proj_ppi  = nn.Linear(dim_ppi, dim_fused)
        
        # Gate network: Determines the importance of PPI features
        # Input is Base + PPI; output is a coefficient from 0 to 1
        self.gate_layer = nn.Sequential(
            nn.Linear(2 * dim_fused, dim_fused),
            nn.Sigmoid()
        )

        # 🔥🔥🔥 Core change 🔥🔥🔥
        # Force the bias of the linear layer in the gate to a negative initial value (for example, -3.0)
        # Rationale: Sigmoid(-3.0) ≈ 0.047
        # Effect: At the start of training, the gate output is very small (approximately zero), so PPI features barely contribute.
        #         The model initially falls back to using only base features, avoiding distortion from random PPI noise.
        #         As training progresses, the model can gradually increase this bias and introduce PPI information.
        nn.init.constant_(self.gate_layer[0].bias, -3.0)
        
        # Normalize to accelerate convergence
        self.norm = nn.LayerNorm(dim_fused)
        self.activation = nn.ReLU()

    def forward(self, x_base, x_ppi, ppi_mask):
        """
        x_base: [B, D_base]
        x_ppi : [B, D_ppi]
        ppi_mask: [B, 1] (1=real PPI available, 0=unavailable)
        """
        # 1. Project to the same dimension
        xb = self.activation(self.proj_base(x_base))   # [B, Df]
        xp = self.activation(self.proj_ppi(x_ppi))     # [B, Df]

        # 2. Enforced masking
        # xp has a value only when ppi_mask=1; otherwise it is forced to zero
        xp = xp * ppi_mask.expand_as(xp)

        # 3. Compute the gate (PPI importance)
        combined = torch.cat([xb, xp], dim=-1)
        gate = self.gate_layer(combined) # [B, Df]

        # 4. Fusion method: Residual complement
        # Initially gate ≈ 0, so h_fused ≈ xb, fully preserving the base features
        h_fused = xb + gate * xp
        
        return self.norm(h_fused)


# class combine_inter_model(nn.Module):
#     def __init__(self, inter_size, inter_hid, graph_size, graph_hid, label_num, head,ppi_in_dim, ppi_hid_dim=256,ppi_out_dim=256,ppi_heads1=4,ppi_heads2=4,fused_dim=256,  ):
#         super(combine_inter_model, self).__init__()
#         self.inter_embedding = inter_model(inter_size, inter_hid)

#         self.GNN = GCN(graph_size, graph_hid, label_num, head)
        
#         self.base_dim = graph_size + graph_hid
#         self.PPI_GNN = TwoLayerPPIGATv2(
#             in_dim=ppi_in_dim,
#             hid_dim=ppi_hid_dim,
#             gat_out_dim=ppi_out_dim,
#             num_classes=label_num,   # num_classes is unused here and serves only as a placeholder
#             num_heads_1=ppi_heads1,
#             num_heads_2=ppi_heads2,
#             edge_dim=1,
#         )
#         self.fusion = MaskedGatedFusion(
#             dim_base=self.base_dim,
#             dim_ppi=ppi_out_dim,
#             dim_fused=fused_dim,
#         )
#         classify_in_dim = fused_dim + inter_hid

#         # ===== New classifier: Input dimension = fused_dim =====
#         self.classify = nn.Sequential(
#             nn.BatchNorm1d(classify_in_dim),
#             nn.Linear(classify_in_dim, fused_dim * 2),
#             nn.Dropout(0.3),
#             nn.ReLU(),
#             nn.Linear(fused_dim * 2, fused_dim * 2),
#             nn.Dropout(0.3),
#             nn.ReLU(),
#             nn.Linear(fused_dim * 2, label_num),
#         )
    

#     def forward(self, inter_feature, graph, graph_h, ppi_graph):
#         """
#         inter_feature : [B, inter_size]   Tri-modal features (same as before)
#         graph         : Structure/residue graph -> GCN
#         graph_h       : Initial graph-node features -> GCN
#         ppi_graph     : PPI graph (batched DGLGraph containing x / weight / is_center)
#         """

#         # ===== 1. Original tri-modal fusion logic: Completely unchanged =====
#         inter_feature = self.inter_embedding(inter_feature)
#         graph_feature, init_feature = self.GNN(graph, graph_h, inter_feature)

#         base_feat = torch.cat((init_feature, graph_feature), dim=1)  # [B, base_dim]

       
#         ppi_feature, ppi_mask = self.PPI_GNN(ppi_graph)

#         # 3. Fusion
#         # ppi_feature is now [B, 256] and can be processed by proj_ppi(256->256)
#         fused_feat = self.fusion(base_feat, ppi_feature, ppi_mask)  # [B, fused_dim]
        
#         final_vec = torch.cat([fused_feat, inter_feature], dim=1)
#         # 4. Classify
#         logits = self.classify(final_vec)
#         return logits
        

class combine_inter_model(nn.Module):
    def __init__(
        self, 
        inter_size, inter_hid, graph_size, graph_hid, label_num, head,
        ppi_in_dim, ppi_hid_dim=256, ppi_out_dim=256, 
        ppi_heads1=4, ppi_heads2=4, fused_dim=256,
        # 🔥🔥🔥 Four ablation switches 🔥🔥🔥
        use_ppi=True,           # Experiment 1: Set False for "w/o PPI"
        use_interpro=True,      # Experiment 2: Set False for "w/o InterPro"
        fusion_type='gated',    # Experiment 3: Set 'concat' for "Replace with Concat"
        use_transformer=True    # Experiment 4: Set False for "w/o Bi-Transformer"
    ):
        super(combine_inter_model, self).__init__()
        
        self.use_ppi = use_ppi
        self.use_interpro = use_interpro
        self.fusion_type = fusion_type
        
        # === 1. InterPro module (w/o InterPro) ===
        if self.use_interpro:
            self.inter_embedding = inter_model(inter_size, inter_hid)
            inter_feat_dim = inter_hid
        else:
            self.inter_embedding = None
            inter_feat_dim = 0

        # === 2. Structure GCN (w/o Bi-Transformer) ===
        # If InterPro is removed, the Transformer in the GCN cannot run, so disable it forcibly
        gcn_use_transformer = use_transformer and use_interpro
        self.GNN = GCN(graph_size, graph_hid, label_num, head, use_transformer=gcn_use_transformer)
        
        self.base_dim = graph_size + graph_hid # GCN output dimension
        
        # === 3. PPI module (w/o PPI) ===
        if self.use_ppi:
            self.PPI_GNN = TwoLayerPPIGATv2(
                in_dim=ppi_in_dim, hid_dim=ppi_hid_dim, gat_out_dim=ppi_out_dim,
                num_classes=label_num, num_heads_1=ppi_heads1, num_heads_2=ppi_heads2
            )
            self.ppi_out_dim = ppi_out_dim
            self.base_projector = None
        else:
            self.PPI_GNN = None
            self.ppi_out_dim = 0
            self.base_projector = nn.Sequential(
                nn.Linear(self.base_dim, fused_dim),
                nn.ReLU()  # An activation function improves performance
            )

        # === 4. Fusion strategy (Replace with Concat) ===
        # Compute the dimension of the fused structural features
        if self.use_ppi:
            if self.fusion_type == 'gated':
                # Gated fusion projects to fused_dim
                self.fusion = MaskedGatedFusion(dim_base=self.base_dim, dim_ppi=ppi_out_dim, dim_fused=fused_dim)
                self.struct_final_dim = fused_dim
            elif self.fusion_type == 'concat':
                # Concat concatenates directly
                self.fusion = None
                self.struct_final_dim = self.base_dim + ppi_out_dim
            else:
                raise ValueError("fusion_type must be 'gated' or 'concat'")
        else:
            # Without PPI, no fusion is needed; use the GCN output directly
            self.fusion = None
            self.struct_final_dim = fused_dim

        # === 5. Classifier ===
        # Final classifier input = structural component + sequence component
        classify_in_dim = self.struct_final_dim + inter_feat_dim
        
        self.classify = nn.Sequential(
            nn.BatchNorm1d(classify_in_dim),
            nn.Linear(classify_in_dim, fused_dim * 2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(fused_dim * 2, fused_dim * 2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(fused_dim * 2, label_num),
        )

    def forward(self, inter_feature, graph, graph_h, ppi_graph):
        # 1. Process InterPro
        if self.use_interpro:
            inter_feat = self.inter_embedding(inter_feature)
        else:
            inter_feat = None

        # 2. Process structure (GCN)
        # The GCN automatically decides whether to use the Bi-Transformer based on use_transformer and whether inter_feat is None
        graph_feature, init_feature = self.GNN(graph, graph_h, inter_feat)
        base_feat = torch.cat((init_feature, graph_feature), dim=1)

        # 3. Process PPI and fusion
        final_struct_feat = base_feat
        
        if self.use_ppi:
            ppi_feature, ppi_mask = self.PPI_GNN(ppi_graph)
            
            if self.fusion_type == 'gated':
                # 🟢 Use gated fusion (baseline)
                final_struct_feat = self.fusion(base_feat, ppi_feature, ppi_mask)
            
            elif self.fusion_type == 'concat':
                # 🟢 Use concatenation fusion (ablation)
                # Even with concatenation, mask PPI features for invalid nodes to ensure a fair comparison
                ppi_feature = ppi_feature * ppi_mask.expand_as(ppi_feature)
                final_struct_feat = torch.cat([base_feat, ppi_feature], dim=1)
        else:
            # 🔥🔥🔥 Critical fix: Apply the alignment layer 🔥🔥🔥
            # Previously: final_struct_feat = base_feat
            # Now:
            final_struct_feat = self.base_projector(base_feat)

        # 4. Final concatenation (late fusion)
        if self.use_interpro:
            final_vec = torch.cat([final_struct_feat, inter_feat], dim=1)
        else:
            final_vec = final_struct_feat

        logits = self.classify(final_vec)
        return logits

