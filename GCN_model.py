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
    [修复版] 带边权重的 GATv2 层
    修复点：alpha 维度扩展，解决 broadcasting error
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

        # 残差连接
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

            # 边特征
            if edge_weight.dim() == 1:
                edge_weight = edge_weight.unsqueeze(-1)
            
            if self.fc_edge is not None:
                ew_feat = self.fc_edge(edge_weight).view(-1, self.num_heads, self.out_dim)
            else:
                ew_feat = torch.zeros(g.num_edges(), self.num_heads, self.out_dim, device=h.device)
            g.edata['ew_feat'] = ew_feat

            # Attention 计算
            def edge_attention(edges):
                z = edges.src['h_src'] + edges.dst['h_dst'] + edges.data['ew_feat']
                z = self.leaky_relu(z)
                e = (z * self.attn).sum(dim=-1) # [E, H]
                return {'e': e}

            g.apply_edges(edge_attention)
            
            e = g.edata.pop('e')
            alpha = edge_softmax(g, e)     # [E, H]
            alpha = self.attn_drop(alpha)
            
            # 🔥🔥🔥【关键修复在这里】🔥🔥🔥
            # 将 [E, H] 变成 [E, H, 1]，以便和 h_src [E, H, D] 广播相乘
            g.edata['alpha'] = alpha.unsqueeze(-1)

            # 消息传递
            g.update_all(
                fn.u_mul_e('h_src', 'alpha', 'm'),
                fn.sum('m', 'h_out')
            )

            h_out = g.ndata['h_out'] 

            if self.concat:
                h_out = h_out.reshape(-1, self.num_heads * self.out_dim)
            else:
                h_out = h_out.mean(dim=1)

            # 残差连接
            if self.res_fc is not None:
                h_out = h_out + self.res_fc(h_in)
            else:
                h_out = h_out + h_in

            return h_out


class TwoLayerPPIGATv2(nn.Module):
    """
    基于 GATv2 + LayerNorm 的 PPI 模型
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
        is_real_edge = (u != v).float()    # 自环=0
        real_deg = torch.zeros(g.num_nodes(), device=h.device)
        real_deg.index_add_(0, v, is_real_edge)  # 入度和
        has_ppi_node = (real_deg > 0)      # [N] bool

        # 只取中心节点
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
        # Stream A: Structure Queries Domain (提取与结构相关的Domain信息)
        # Q=Residue, K=Inter, V=Inter (注意这里V是Inter)
        # =========================================================
        self.stream_a_q = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_a_k = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_a_v = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        # =========================================================
        # Stream B: Domain Queries Structure (提取与Domain匹配的结构信息)
        # Q=Inter, K=Residue, V=Residue (注意这里V是Residue)
        # =========================================================
        self.stream_b_q = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_b_k = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.stream_b_v = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        # =========================================================
        # Fusion Layer (融合双向特征)
        # 输入维度是: (head * hidden) * 2 (因为有两个流)
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
        # residue_h: 结构特征 (N_nodes, in_dim)
        # inter_h:   Domain特征 (N_nodes, in_dim) 
        # 注意: 这里假设 inter_h 已经广播/扩展到了与 residue_h 相同的形状
        self.attention_scores = {
            'stream_a': [], # 结构主动查序列 (你的独家优势)
            'stream_b': []  # 序列反查结构 (验证用)
        }
        output_stream_a = []
        output_stream_b = []

        for i in range(self.head):
            # -------------------------------------------------
            # Stream A 计算: Structure -> Domain
            # 逻辑: "我是这个结构，请给我匹配的Domain信息"
            # -------------------------------------------------
            qa = self.stream_a_q[i](residue_h) # Q = Structure
            ka = self.stream_a_k[i](inter_h)   # K = Domain
            va = self.stream_a_v[i](inter_h)   # V = Domain (获取Domain内容)
            
            # 计算注意力分数
            att_a = torch.sum(torch.mul(qa, ka) / torch.sqrt(torch.tensor(float(self.hidden_dim))), dim=1, keepdim=True)
            
            # 使用图结构进行 Softmax (局部归一化)
            with g.local_scope():
                g.ndata['att_a'] = att_a.reshape(-1)
                alpha_a = dgl.softmax_nodes(g, 'att_a').reshape((va.size(0), 1))
                self.attention_scores['stream_a'].append(alpha_a.detach().cpu())
                out_a = va * alpha_a # 加权后的 Domain 特征
            output_stream_a.append(out_a)

            # -------------------------------------------------
            # Stream B 计算: Domain -> Structure
            # 逻辑: "我是这个Domain，请高亮匹配的结构区域"
            # -------------------------------------------------
            qb = self.stream_b_q[i](inter_h)   # Q = Domain
            kb = self.stream_b_k[i](residue_h) # K = Structure
            vb = self.stream_b_v[i](residue_h) # V = Structure (获取结构内容)
            
            att_b = torch.sum(torch.mul(qb, kb) / torch.sqrt(torch.tensor(float(self.hidden_dim))), dim=1, keepdim=True)
            
            with g.local_scope():
                g.ndata['att_b'] = att_b.reshape(-1)
                alpha_b = dgl.softmax_nodes(g, 'att_b').reshape((vb.size(0), 1))
                self.attention_scores['stream_b'].append(alpha_b.detach().cpu())
                out_b = vb * alpha_b # 加权后的 Structure 特征
            output_stream_b.append(out_b)

        # -------------------------------------------------
        # 融合与输出
        # -------------------------------------------------
        
        # 1. 拼接多头结果
        multi_out_a = torch.cat(output_stream_a, dim=1) # [N, head*hidden]
        multi_out_b = torch.cat(output_stream_b, dim=1) # [N, head*hidden]
        
        # 2. 拼接双向结果: [Structure-Aware-Domain, Domain-Aware-Structure]
        combined = torch.cat([multi_out_a, multi_out_b], dim=1) # [N, 2*head*hidden]
        
        # 3. 线性融合降维
        fused_h = self.fusion_trans(combined) # [N, hidden]

        # 4. 残差连接 + LayerNorm (保留原始结构特征作为Base)
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
        

class GCN(nn.Module):    #消融
    def __init__(self, in_dim, hidden_dim, n_classes, head, use_transformer=True): # 🔥 新增开关
        super(GCN, self).__init__()
        self.use_transformer = use_transformer 
        
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        
        # 🟢 消融实验关键点：如果不用 Transformer，就不初始化它
        if self.use_transformer:
            self.transformer_block = BiDirectionalTransformerBlock(hidden_dim, hidden_dim, head)
        else:
            self.transformer_block = None

    def forward(self, g, h, inter_f):
            # ✅ 第一步：先计算初始特征的均值 (init_avg_h) 并存起来
            # 这一步必须最先做，否则后面 h 经过卷积后就变了，或者 g 里面没有 'h'
            with g.local_scope():
                g.ndata['h_init'] = h  # 把输入的 h 存入图，命名为 'h_init'
                init_avg_h = dgl.mean_nodes(g, 'h_init') # 计算均值

            # ... (中间的卷积层保持不变) ...
            pre = h
            h = self.bn1(h)
            h = pre + self.dropout(F.relu(self.conv1(g, h)))

            pre = h
            h = self.bn2(h)
            h = pre + self.dropout(F.relu(self.conv2(g, h)))

            # ... (交互模块 / 消融实验分支) ...
            if self.use_transformer and inter_f is not None:
                # Baseline: 有交互
                with g.local_scope():
                    g.ndata['inter'] = dgl.broadcast_nodes(g, inter_f)
                    residue_h = h
                    inter_h = g.ndata['inter']
                    hg = self.transformer_block(g, residue_h, inter_h)
                    g.ndata['output'] = hg
                    readout = dgl.sum_nodes(g, "output")
                    
                    # ✅ 返回刚才算好的 init_avg_h，而不是现场去 mean_nodes
                    return readout, init_avg_h 
            else:
                # 🔥 Ablation: 无交互 (这里是你报错的地方)
                with g.local_scope():
                    g.ndata['h_final'] = h
                    readout = dgl.sum_nodes(g, "h_final")
                    
                    # ✅ 修复：直接返回上面算好的 init_avg_h
                    # 不要写 dgl.mean_nodes(g, 'h')，因为此时 g 里没有 'h'
                    return readout, init_avg_h



# class MaskedGatedFusion(nn.Module):
#     """
#     [改进版] 带 PPI mask 的门控融合
#     逻辑：
#     1. 强制 mask：如果没有 PPI，强制将 PPI 特征置零，不让噪声进入。
#     2. 残差融合：Base + Gate * PPI。以 Base 为主，PPI 为锦上添花的补充。
#     """
#     def __init__(self, dim_base, dim_ppi, dim_fused):
#         super().__init__()
#         self.proj_base = nn.Linear(dim_base, dim_fused)
#         self.proj_ppi  = nn.Linear(dim_ppi, dim_fused)
        
#         # Gate 网络：决定 PPI 特征的重要性
#         # 输入是 Base + PPI，输出一个 0~1 的系数
#         self.gate_layer = nn.Sequential(
#             nn.Linear(2 * dim_fused, dim_fused),
#             nn.Sigmoid()
#         )
        
#         # 归一化，加速收敛
#         self.norm = nn.LayerNorm(dim_fused)
#         self.activation = nn.ReLU()

#     def forward(self, x_base, x_ppi, ppi_mask):
#         """
#         x_base: [B, D_base]
#         x_ppi : [B, D_ppi]
#         ppi_mask: [B, 1] (1=有真实PPI, 0=无)
#         """
#         # 1. 投影到相同维度
#         xb = self.activation(self.proj_base(x_base))  # [B, Df]
#         xp = self.activation(self.proj_ppi(x_ppi))    # [B, Df]

#         # 2. 🔥 核心修正：强制 Masking
#         # 只有当 ppi_mask=1 时，xp 才有值；否则强制为 0
#         xp = xp * ppi_mask.expand_as(xp)

#         # 3. 计算 Gate (PPI 重要性)
#         # 只需要看 xb 和 xp 的结合情况
#         combined = torch.cat([xb, xp], dim=-1)
#         gate = self.gate_layer(combined) # [B, Df]

#         # 4. 🔥 融合方式：残差补充 (Residual Complement)
#         # 逻辑：基础特征 + (重要性 * PPI特征)
#         # 这样即使 mask=0，xp是0，gate不管是什么，结果就是纯粹的 xb，绝无噪声
#         h_fused = xb + gate * xp
        
#         return self.norm(h_fused)

class MaskedGatedFusion(nn.Module):
    """
    [改进版] 带 PPI mask 的门控融合 + 🔥冷启动初始化🔥
    逻辑：
    1. 强制 mask：如果没有 PPI，强制将 PPI 特征置零。
    2. 残差融合：Base + Gate * PPI。
    3. 冷启动：初始时刻强制关闭 Gate，防止 PPI 噪声破坏主干特征。
    """
    def __init__(self, dim_base, dim_ppi, dim_fused):
        super().__init__()
        self.proj_base = nn.Linear(dim_base, dim_fused)
        self.proj_ppi  = nn.Linear(dim_ppi, dim_fused)
        
        # Gate 网络：决定 PPI 特征的重要性
        # 输入是 Base + PPI，输出一个 0~1 的系数
        self.gate_layer = nn.Sequential(
            nn.Linear(2 * dim_fused, dim_fused),
            nn.Sigmoid()
        )

        # 🔥🔥🔥 【核心修改在这里】 🔥🔥🔥
        # 强制将 Gate 中线性层的偏置 (bias) 初始化为负数 (例如 -3.0)
        # 原理: Sigmoid(-3.0) ≈ 0.047
        # 效果: 训练刚开始时，Gate 输出的值非常小 (约等于0)，PPI 特征几乎不参与融合。
        #       这意味着模型在初期会退化为 "只用 Base 特征" 的状态，避免被 PPI 的随机噪声带偏。
        #       随着训练进行，模型会自己学会把这个 bias 慢慢调大，逐渐引入 PPI 信息。
        nn.init.constant_(self.gate_layer[0].bias, -3.0)
        
        # 归一化，加速收敛
        self.norm = nn.LayerNorm(dim_fused)
        self.activation = nn.ReLU()

    def forward(self, x_base, x_ppi, ppi_mask):
        """
        x_base: [B, D_base]
        x_ppi : [B, D_ppi]
        ppi_mask: [B, 1] (1=有真实PPI, 0=无)
        """
        # 1. 投影到相同维度
        xb = self.activation(self.proj_base(x_base))   # [B, Df]
        xp = self.activation(self.proj_ppi(x_ppi))     # [B, Df]

        # 2. 强制 Masking
        # 只有当 ppi_mask=1 时，xp 才有值；否则强制为 0
        xp = xp * ppi_mask.expand_as(xp)

        # 3. 计算 Gate (PPI 重要性)
        combined = torch.cat([xb, xp], dim=-1)
        gate = self.gate_layer(combined) # [B, Df]

        # 4. 融合方式：残差补充 (Residual Complement)
        # 初期 gate ≈ 0，这里就变成了 h_fused ≈ xb，完美保护了你的 Base 特征
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
#             num_classes=label_num,   # 这里的 num_classes 用不到，只是占位
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

#         # ===== 新的分类器：输入维度 = fused_dim =====
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
#         inter_feature : [B, inter_size]   三模态特征 (和以前一样)
#         graph         : 结构/残基图       -> GCN
#         graph_h       : 图节点初始特征   -> GCN
#         ppi_graph     : PPI 图 (batched DGLGraph，含 x / weight / is_center)
#         """

#         # ===== 1. 原来的“三模态融合逻辑”：完全不变 =====
#         inter_feature = self.inter_embedding(inter_feature)
#         graph_feature, init_feature = self.GNN(graph, graph_h, inter_feature)

#         base_feat = torch.cat((init_feature, graph_feature), dim=1)  # [B, base_dim]

       
#         ppi_feature, ppi_mask = self.PPI_GNN(ppi_graph)

#         # 3. Fusion
#         # 现在传入的 ppi_feature 是 [B, 256]，可以被 proj_ppi(256->256) 处理了
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
        # 🔥🔥🔥 四大消融实验开关 🔥🔥🔥
        use_ppi=True,           # 实验1: Set False for "w/o PPI"
        use_interpro=True,      # 实验2: Set False for "w/o InterPro"
        fusion_type='gated',    # 实验3: Set 'concat' for "Replace with Concat"
        use_transformer=True    # 实验4: Set False for "w/o Bi-Transformer"
    ):
        super(combine_inter_model, self).__init__()
        
        self.use_ppi = use_ppi
        self.use_interpro = use_interpro
        self.fusion_type = fusion_type
        
        # === 1. InterPro 模块 (w/o InterPro) ===
        if self.use_interpro:
            self.inter_embedding = inter_model(inter_size, inter_hid)
            inter_feat_dim = inter_hid
        else:
            self.inter_embedding = None
            inter_feat_dim = 0

        # === 2. Structure GCN (w/o Bi-Transformer) ===
        # 如果去掉了 InterPro，GCN 里的 Transformer 也没法跑了，所以强制关闭
        gcn_use_transformer = use_transformer and use_interpro
        self.GNN = GCN(graph_size, graph_hid, label_num, head, use_transformer=gcn_use_transformer)
        
        self.base_dim = graph_size + graph_hid # GCN 输出维度
        
        # === 3. PPI 模块 (w/o PPI) ===
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
                nn.ReLU()  # 加个激活函数效果更好
            )

        # === 4. Fusion 策略 (Replace with Concat) ===
        # 计算融合后的结构特征维度
        if self.use_ppi:
            if self.fusion_type == 'gated':
                # Gated 会投影到 fused_dim
                self.fusion = MaskedGatedFusion(dim_base=self.base_dim, dim_ppi=ppi_out_dim, dim_fused=fused_dim)
                self.struct_final_dim = fused_dim
            elif self.fusion_type == 'concat':
                # Concat 直接拼接
                self.fusion = None
                self.struct_final_dim = self.base_dim + ppi_out_dim
            else:
                raise ValueError("fusion_type must be 'gated' or 'concat'")
        else:
            # 没有 PPI，不需要融合，直接用 GCN 输出
            self.fusion = None
            self.struct_final_dim = fused_dim

        # === 5. Classifier ===
        # 最终分类输入 = (结构部分) + (序列部分)
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
        # 1. 处理 InterPro
        if self.use_interpro:
            inter_feat = self.inter_embedding(inter_feature)
        else:
            inter_feat = None

        # 2. 处理 Structure (GCN)
        # GCN 内部会根据 use_transformer 和 inter_feat 是否为 None 自动决定走不走 Bi-Transformer
        graph_feature, init_feature = self.GNN(graph, graph_h, inter_feat)
        base_feat = torch.cat((init_feature, graph_feature), dim=1)

        # 3. 处理 PPI & Fusion
        final_struct_feat = base_feat
        
        if self.use_ppi:
            ppi_feature, ppi_mask = self.PPI_GNN(ppi_graph)
            
            if self.fusion_type == 'gated':
                # 🟢 使用门控融合 (Baseline)
                final_struct_feat = self.fusion(base_feat, ppi_feature, ppi_mask)
            
            elif self.fusion_type == 'concat':
                # 🟢 使用 Concat 融合 (Ablation)
                # 即使是 Concat，也最好把无效节点的 PPI 特征 mask 掉，保证公平对比
                ppi_feature = ppi_feature * ppi_mask.expand_as(ppi_feature)
                final_struct_feat = torch.cat([base_feat, ppi_feature], dim=1)
        else:
            # 🔥🔥🔥【关键修复】应用对齐层🔥🔥🔥
            # 原来是: final_struct_feat = base_feat
            # 现在改为:
            final_struct_feat = self.base_projector(base_feat)

        # 4. 最终拼接 (Late Fusion)
        if self.use_interpro:
            final_vec = torch.cat([final_struct_feat, inter_feat], dim=1)
        else:
            final_vec = final_struct_feat

        logits = self.classify(final_vec)
        return logits

