import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from layers import act_layer, norm_layer


def get_token_dropout_view(x, p=0.05):
    """
    Token-level dropout: randomly drop entire token vectors (residues) with probability `p`.

    Args:
        x: [B, L, D]
        p: probability to drop a token

    Returns:
        x_masked: [B, L, D]
    """
    b, l, d = x.shape
    device = x.device
    # keep mask with probability (1-p)
    keep = (torch.rand((b, l, 1), device=device) > p).to(x.dtype)
    return x * keep


class CleanNoisyMarginContrastiveLoss(nn.Module):
    """
    clean/noisy 对比损失。

    正样本：
        同一个残基位置的 clean 表示和 noisy 表示。
        z_clean[i] <-> z_noisy[i]

    负样本：
        不同残基位置之间的 clean/noisy 表示。
        z_clean[i] <-> z_noisy[j], i != j
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, z_clean, z_noisy):
        """
        Args:
            z_clean: [B, L, P]
            z_noisy: [B, L, P]
        """
        b, l, p = z_clean.shape

        z_clean = z_clean.reshape(b * l, p)
        z_noisy = z_noisy.reshape(b * l, p)

        z_clean = F.normalize(z_clean, dim=-1)
        z_noisy = F.normalize(z_noisy, dim=-1)

        dist = torch.cdist(z_clean, z_noisy, p=2)

        n = dist.size(0)
        device = dist.device

        pos_dist = torch.diag(dist)
        pos_loss = torch.mean(pos_dist.pow(2))

        neg_mask = ~torch.eye(n, dtype=torch.bool, device=device)

        if neg_mask.sum() > 0:
            neg_dist = dist[neg_mask]
            neg_loss = torch.mean(F.relu(self.margin - neg_dist).pow(2))
        else:
            neg_loss = torch.tensor(0.0, device=device)

        return pos_loss + neg_loss


class WiKGStyleCLProjectionHead(nn.Module):
    """
    clean/original view:
        h_clean -> fc1 -> z_clean

    mask/noisy view:
        h_mask -> fc2 -> z_mask
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        projection_dim=128
    ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, projection_dim)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, projection_dim)
        )

    def forward(self, h_clean, h_mask):
        z_clean = self.fc1(h_clean)
        z_mask = self.fc2(h_mask)
        return z_clean, z_mask


class LocalGateAttenHypergraphConstructor(nn.Module):
    """
    按参考代码中 gateAtten 的思想构图：

    不是 dense L x L x C gate attention；
    而是：
        1. 先用 cosine KNN 找 top-k 候选邻居；
        2. 只在 top-k 邻居内部计算 gate attention score；
        3. 用 gate score 的 softmax 作为 H 中的连续权重。

    这样复杂度约为:
        O(B * L * L) for cosine top-k
        +
        O(B * L * k * C) for local gated attention

    避免 O(B * L * L * C) 的巨大开销。
    """

    def __init__(
        self,
        channels,
        k=9,
        gate_hidden_channels=None,
        temperature=0.2,
        dropout=0.0,
        add_self_loop=True
    ):
        super().__init__()

        self.channels = channels
        self.k = k
        self.temperature = temperature
        self.add_self_loop = add_self_loop

        gate_hidden_channels = gate_hidden_channels or channels

        self.center_v = nn.Linear(channels, gate_hidden_channels)
        self.neighbor_v = nn.Linear(channels, gate_hidden_channels)

        self.center_u = nn.Linear(channels, gate_hidden_channels)
        self.neighbor_u = nn.Linear(channels, gate_hidden_channels)

        self.score_proj = nn.Linear(gate_hidden_channels, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _get_topk_neighbors(self, x):
        """
        先用 cosine similarity 找 top-k 候选邻居。

        Args:
            x: [B, L, C]

        Returns:
            knn_index: [B, L, k]
            topk_sim: [B, L, k]
        """
        b, l, c = x.shape
        device = x.device
        k = min(self.k, l)

        with torch.no_grad():
            x_norm = F.normalize(x.detach(), dim=-1)

            sim = torch.matmul(
                x_norm,
                x_norm.transpose(1, 2)
            )  # [B, L, L]

            if self.add_self_loop:
                eye = torch.eye(
                    l,
                    device=device,
                    dtype=torch.bool
                ).unsqueeze(0)

                row_max = sim.max(dim=-1, keepdim=True).values
                sim = sim.masked_scatter(
                    eye,
                    row_max.expand_as(sim)[eye] + 1.0
                )

            topk_sim, knn_index = torch.topk(
                sim,
                k=k,
                dim=-1,
                largest=True
            )

        return knn_index, topk_sim

    def forward(self, x):
        """
        Args:
            x: [B, L, C]

        Returns:
            H: [B, L, E], E = L
               H[b, node_j, edge_i] 表示 node_j 属于以 node_i 为中心的超边的权重。

            knn_index: [B, L, k]
               每个中心残基选择的 top-k 邻居。

            gate_score: [B, L, k]
               每个中心残基对 top-k 邻居的 gate attention score。
        """
        b, l, c = x.shape
        device = x.device
        dtype = x.dtype

        k = min(self.k, l)

        # -----------------------------------------------------
        # 1. Sparse candidate neighbors from cosine KNN
        # -----------------------------------------------------
        knn_index, _ = self._get_topk_neighbors(x)

        # -----------------------------------------------------
        # 2. Gather top-k neighbor features
        # -----------------------------------------------------
        # x_expand: [B, L_center, L_node, C], expand 是 view，不真实复制大张量
        x_expand = x.unsqueeze(1).expand(-1, l, -1, -1)

        gather_index = knn_index.unsqueeze(-1).expand(-1, -1, -1, c)

        neighbor_x = torch.gather(
            x_expand,
            dim=2,
            index=gather_index
        )  # [B, L, k, C]

        center_x = x.unsqueeze(2).expand(-1, -1, k, -1)  # [B, L, k, C]

        # -----------------------------------------------------
        # 3. Local gate attention score
        # -----------------------------------------------------
        value_branch = torch.tanh(
            self.center_v(center_x) + self.neighbor_v(neighbor_x)
        )

        gate_branch = torch.sigmoid(
            self.center_u(center_x) + self.neighbor_u(neighbor_x)
        )

        gated_feature = value_branch * gate_branch
        gated_feature = self.dropout(gated_feature)

        gate_score = self.score_proj(gated_feature).squeeze(-1)  # [B, L, k]

        # -----------------------------------------------------
        # 4. Local softmax weights
        # -----------------------------------------------------
        temp = max(float(self.temperature), 1e-6)
        topk_weight = F.softmax(gate_score / temp, dim=-1).to(dtype)  # [B, L, k]

        # -----------------------------------------------------
        # 5. Build weighted incidence matrix H
        # H_by_edge[b, edge_i, node_j] = weight
        # H[b, node_j, edge_i] = weight
        # -----------------------------------------------------
        H_by_edge = torch.zeros(
            b,
            l,
            l,
            device=device,
            dtype=dtype
        )

        H_by_edge = H_by_edge.scatter(
            dim=2,
            index=knn_index,
            src=topk_weight
        )

        H = H_by_edge.transpose(1, 2).contiguous()  # [B, L_node, L_edge]

        return H, knn_index, gate_score


class HypergraphConvLayer(nn.Module):
    """
    加权超图卷积层。

    信息传播：
        1. Node -> Hyperedge
        2. Hyperedge -> Node
    """

    def __init__(
        self,
        channels,
        num_heads=6,
        dropout=0.1,
        use_membership_bias=True
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.use_membership_bias = use_membership_bias

        self.head_dim = math.ceil(channels / num_heads)
        self.inner_dim = self.head_dim * num_heads

        self.q_node = nn.Linear(channels, self.inner_dim)
        self.k_node = nn.Linear(channels, self.inner_dim)
        self.v_node = nn.Linear(channels, self.inner_dim)

        self.k_edge = nn.Linear(self.inner_dim, self.inner_dim)
        self.v_edge = nn.Linear(self.inner_dim, self.inner_dim)

        self.out_proj = nn.Linear(self.inner_dim, channels)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, H):
        """
        Args:
            x: [B, L, C]
            H: [B, L, E], weighted incidence matrix

        Returns:
            out: [B, L, C]
        """
        b, l, c = x.shape
        e = H.size(-1)

        q_node = self.q_node(x)
        k_node = self.k_node(x)
        v_node = self.v_node(x)

        q_node = q_node.view(
            b,
            l,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k_node = k_node.view(
            b,
            l,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v_node = v_node.view(
            b,
            l,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # -----------------------------------------------------
        # Node -> Hyperedge weighted aggregation
        # -----------------------------------------------------
        H_e = H.transpose(1, 2)  # [B, E, L]

        edge_degree = H_e.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        edge_v = torch.einsum(
            "bel,bhld->bhed",
            H_e,
            v_node
        ) / edge_degree.unsqueeze(1)

        edge_v_flat = edge_v.transpose(1, 2).contiguous().view(
            b,
            e,
            self.inner_dim
        )

        edge_k_flat = self.k_edge(edge_v_flat)
        edge_v_flat = self.v_edge(edge_v_flat)

        edge_k = edge_k_flat.view(
            b,
            e,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        edge_v = edge_v_flat.view(
            b,
            e,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # -----------------------------------------------------
        # Hyperedge -> Node attention
        # -----------------------------------------------------
        attn = torch.einsum(
            "bhld,bhed->bhle",
            q_node,
            edge_k
        ) * self.scale

        hyperedge_mask = H.unsqueeze(1) > 0
        attn = attn.masked_fill(~hyperedge_mask, float("-inf"))

        if self.use_membership_bias:
            membership_bias = torch.log(H.clamp(min=1e-6)).unsqueeze(1)
            attn = attn + membership_bias

        all_inf_mask = torch.isinf(attn).all(dim=-1, keepdim=True)
        attn = attn.masked_fill(all_inf_mask, 0.0)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum(
            "bhle,bhed->bhld",
            attn,
            edge_v
        )

        out = out.transpose(1, 2).contiguous().view(
            b,
            l,
            self.inner_dim
        )

        out = self.out_proj(out)

        return out


class HypergraphBlock(nn.Module):
    """
    Local GateAtten Hypergraph Block.

    Structure:
        cosine top-k candidate
        local gate attention reweighting
        x = x + DropPath(HypergraphConv(Norm(x), H))
        x = x + DropPath(FFN(Norm(x)))
    """

    def __init__(
        self,
        channels,
        k=9,
        num_heads=6,
        mlp_ratio=4,
        act="gelu",
        norm="layer",
        dropout=0.1,
        drop_path=0.0,
        gate_hidden_channels=None,
        gate_temperature=0.2,
        use_membership_bias=True
    ):
        super().__init__()

        self.k = k

        self.gate_constructor = LocalGateAttenHypergraphConstructor(
            channels=channels,
            k=k,
            gate_hidden_channels=gate_hidden_channels,
            temperature=gate_temperature,
            dropout=dropout,
            add_self_loop=True
        )

        self.norm1 = norm_layer(norm, channels)

        self.hypergraph_conv = HypergraphConvLayer(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
            use_membership_bias=use_membership_bias
        )

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(norm, channels)

        hidden_features = channels * mlp_ratio

        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_features),
            norm_layer(norm, hidden_features),
            act_layer(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, channels),
            norm_layer(norm, channels),
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x,
        return_hypergraph=False,
        return_knn_index=False,
        return_gate_score=False
    ):
        H, knn_index, gate_score = self.gate_constructor(x)

        shortcut = x
        x_norm = self.norm1(x)
        x_hyper = self.hypergraph_conv(x_norm, H)
        x = shortcut + self.drop_path1(x_hyper)

        shortcut = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = shortcut + self.drop_path2(x_ffn)

        extra_outputs = []

        if return_hypergraph:
            extra_outputs.append(H)

        if return_knn_index:
            extra_outputs.append(knn_index)

        if return_gate_score:
            extra_outputs.append(gate_score)

        if len(extra_outputs) == 0:
            return x

        return (x, *extra_outputs)


class ProteinGCN(nn.Module):
    """
    保持类名 ProteinGCN，兼容你现有训练脚本。

    实际结构：
        Protein Local-GateAtten Hypergraph Network
        + WiKG-style clean/mask contrastive projection head
    """

    def __init__(self, opt):
        super().__init__()

        dim_in = opt.dim_in
        channels = opt.n_filters
        n_classes = opt.n_classes

        self.k = opt.k
        self.n_blocks = opt.n_blocks

        act = opt.act
        norm = opt.norm
        dropout = opt.dropout
        drop_path = opt.drop_path

        # contrastive learning params
        self.use_residual_cl = opt.use_residual_cl
        self.CL_margin = opt.CL_margin
        self.CL_noise_max = opt.CL_noise_max
        self.CL_loss_weight = opt.CL_loss_weight

        # local gateAtten params
        self.gate_hidden_channels = opt.gate_hidden_channels
        self.gate_temperature = opt.gate_temperature
        self.use_membership_bias = opt.use_membership_bias

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(dim_in, channels),
            act_layer(act)
        )

        # Local gateAtten hypergraph backbone
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]

        self.backbone = nn.ModuleList([
            HypergraphBlock(
                channels=channels,
                k=self.k,
                num_heads=opt.num_heads,
                mlp_ratio=4,
                act=act,
                norm=norm,
                dropout=dropout,
                drop_path=dpr[i],
                gate_hidden_channels=self.gate_hidden_channels,
                gate_temperature=self.gate_temperature,
                use_membership_bias=self.use_membership_bias
            )
            for i in range(self.n_blocks)
        ])

        self.norm = norm_layer(norm, channels)

        # Node classification head
        self.node_prediction_head = nn.Linear(channels, n_classes)

        # WiKG-style CL projection head
        if self.use_residual_cl:
            self.CL_head = WiKGStyleCLProjectionHead(
                in_channels=channels,
                hidden_channels=opt.hidden_channels,
                projection_dim=128
            )

            self.CL_loss_fn = CleanNoisyMarginContrastiveLoss(
                margin=opt.CL_margin
            )
        else:
            self.CL_head = None
            self.CL_loss_fn = None

    def encode_features(
        self,
        inputs,
        return_edge=False,
        return_knn_index=False,
        return_gate_score=False
    ):
        """
        Shared encoder.

        Args:
            inputs: [B, L, D]

        Returns:
            h: [B, L, C]
            hypergraph_list if return_edge=True
            knn_index_list if return_knn_index=True
            gate_score_list if return_gate_score=True
        """

        x = self.stem(inputs)

        hypergraph_list = []
        knn_index_list = []
        gate_score_list = []

        for block in self.backbone:
            block_out = block(
                x,
                return_hypergraph=return_edge,
                return_knn_index=return_knn_index,
                return_gate_score=return_gate_score
            )

            if isinstance(block_out, tuple):
                x = block_out[0]
                extra = block_out[1:]

                ptr = 0

                if return_edge:
                    hypergraph_list.append(extra[ptr])
                    ptr += 1

                if return_knn_index:
                    knn_index_list.append(extra[ptr])
                    ptr += 1

                if return_gate_score:
                    gate_score_list.append(extra[ptr])
                    ptr += 1

            else:
                x = block_out

        x = self.norm(x)

        outputs = [x]

        if return_edge:
            outputs.append(hypergraph_list)

        if return_knn_index:
            outputs.append(knn_index_list)

        if return_gate_score:
            outputs.append(gate_score_list)

        if len(outputs) == 1:
            return x

        return tuple(outputs)

    def new_mask_data(self, inputs, max_noise=None):
        if max_noise is None:
            max_noise = self.CL_noise_max
        return get_token_dropout_view(inputs, p=float(max_noise))

    def forward(
        self,
        inputs,
        labels=None,
        return_edge=False,
        return_knn_index=False,
        return_gate_score=False,
        return_cl=False,
        return_loss=False
    ):
        """
        Args:
            inputs: [B, L, D]
            labels: [B, L]

        Normal output:
            logits: [B, L, num_classes]

        If return_edge=True:
            outputs["hypergraphs"]

        If return_knn_index=True:
            outputs["knn_indices"]

        If return_gate_score=True:
            outputs["gate_scores"]

        If return_cl=True:
            outputs["cl_loss"]
            outputs["z_clean"]
            outputs["z_noisy"]

        If return_loss=True:
            outputs["cls_loss"]
            outputs["loss"]
        """

        encoder_outputs = self.encode_features(
            inputs,
            return_edge=return_edge,
            return_knn_index=return_knn_index,
            return_gate_score=return_gate_score
        )

        if isinstance(encoder_outputs, tuple):
            h_clean = encoder_outputs[0]
            extra = encoder_outputs[1:]

            ptr = 0

            if return_edge:
                hypergraph_list = extra[ptr]
                ptr += 1
            else:
                hypergraph_list = None

            if return_knn_index:
                knn_index_list = extra[ptr]
                ptr += 1
            else:
                knn_index_list = None

            if return_gate_score:
                gate_score_list = extra[ptr]
                ptr += 1
            else:
                gate_score_list = None

        else:
            h_clean = encoder_outputs
            hypergraph_list = None
            knn_index_list = None
            gate_score_list = None

        logits = self.node_prediction_head(h_clean)

        outputs = {
            "logits": logits
        }

        if return_edge:
            outputs["hypergraphs"] = hypergraph_list

        if return_knn_index:
            outputs["knn_indices"] = knn_index_list

        if return_gate_score:
            outputs["gate_scores"] = gate_score_list


        if return_cl and self.use_residual_cl:
            mask_inputs = self.new_mask_data(
                inputs,
                max_noise=self.CL_noise_max
            )

            h_mask = self.encode_features(
                mask_inputs,
                return_edge=False,
                return_knn_index=False,
                return_gate_score=False
            )

            z_clean, z_noisy = self.CL_head(h_clean, h_mask)

            cl_loss = self.CL_loss_fn(z_clean, z_noisy)

            outputs["cl_loss"] = cl_loss
            outputs["z_clean"] = z_clean
            outputs["z_noisy"] = z_noisy


        if return_loss:
            if labels is None:
                raise ValueError("return_loss=True requires labels.")

            cls_loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                labels.contiguous().view(-1).long()
            )

            outputs["cls_loss"] = cls_loss

            if return_cl and self.use_residual_cl:
                total_loss = cls_loss + self.CL_loss_weight * outputs["cl_loss"]
            else:
                total_loss = cls_loss

            outputs["loss"] = total_loss

        if (
            not return_edge
            and not return_knn_index
            and not return_gate_score
            and not return_cl
            and not return_loss
        ):
            return logits

        return outputs


class OptInit:
    def __init__(
        self,
        dim_in=1280,
        num_classes=2,
        drop_path_rate=0.0,
        num_knn=12,
        **kwargs
    ):
        self.dim_in = dim_in
        self.k = num_knn

        self.act = "silu"
        self.norm = "layer"

        # param_search['num_layers']
        self.n_blocks = kwargs.get("num_layers", 2)

        # param_search['hidden_channels']
        self.n_filters = kwargs.get("hidden_channels", 256)

        self.n_classes = num_classes

        self.drop_path = drop_path_rate
        self.dropout = kwargs.get("dropout", 0.25)

        # param_search
        self.hidden_channels = kwargs.get("hidden_channels", 256)
        self.num_heads = kwargs.get("num_heads", 6)
        self.num_layers = kwargs.get("num_layers", 2)
        self.CL_margin = kwargs.get("CL_margin", 0.5)
        self.CL_noise_max = kwargs.get("CL_noise_max", 0.05)


        # 名称保持不变，兼容 trainer 里的 use_residual_cl 判断
        self.use_residual_cl = kwargs.get("use_residual_cl", True)

        # not in param_search, but needed for training
        self.CL_loss_weight = kwargs.get("CL_loss_weight", 0.05)

        # gateAtten config, 对齐参考代码中的 self.knn="gateAtten"; self.gate=True
        self.knn = "gateAtten"
        self.gate = True

        # local gateAtten params
        self.gate_hidden_channels = kwargs.get("gate_hidden_channels", self.n_filters)
        self.gate_temperature = kwargs.get("gate_temperature", 0.5)

        # hypergraph attention propagation
        self.use_membership_bias = kwargs.get("use_membership_bias", False)
