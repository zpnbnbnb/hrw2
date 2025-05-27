import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, \
    precision_recall_curve
import sklearn.metrics as metrics


class FocalLoss(nn.Module):
    """数值稳定的Focal Loss"""

    def __init__(self, alpha=0.6, gamma=1.8, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        # 添加数值稳定性检查
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("Warning: NaN or Inf detected in focal loss inputs")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 检查结果
        if torch.isnan(focal_loss).any():
            print("Warning: NaN in focal loss output, using cross entropy")
            return F.cross_entropy(inputs, targets, weight=self.class_weights)

        return focal_loss.mean()


class AttentionPooling(nn.Module):
    """Attention Pooling Mechanism"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=0)
        return torch.sum(attn_weights * x, dim=0, keepdim=True)


class ResidualGCNLayer(nn.Module):
    """数值稳定的GCN Layer with Residual Connection"""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.bn = BatchNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        # 检查输入
        if torch.isnan(x).any():
            print("Warning: NaN in GCN input")
            x = torch.nan_to_num(x, nan=0.0)

        identity = self.residual(x)
        out = self.gcn(x, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        # 检查输出
        result = out + identity
        if torch.isnan(result).any():
            print("Warning: NaN in GCN output")
            result = torch.nan_to_num(result, nan=0.0)

        return result


class StableCSGDN(nn.Module):
    """数值稳定的CSGDN模型"""

    def __init__(self, args, layer_num=3):
        super().__init__()
        self.args = args
        self.device = args.device
        self.in_channels = args.feature_dim
        self.out_channels = args.feature_dim
        self.layer_num = layer_num

        # 存储特征矩阵
        self.x = None
        self.debug_loss = False

        # Activation function
        self.activation = nn.ReLU()

        # 计算拼接维度
        concat_dim = 2 * args.feature_dim

        # 数值稳定的特征变换
        self.transform = nn.Sequential(
            nn.Linear(concat_dim, args.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(args.feature_dim * 2),
            nn.Linear(args.feature_dim * 2, args.feature_dim),
            nn.Dropout(0.1)
        ).to(self.device)

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            ResidualGCNLayer(self.in_channels, self.out_channels, dropout=0.3)
            for _ in range(layer_num)
        ])

        # Attention mechanism
        self.attention_pooling = AttentionPooling(args.feature_dim)

        # GAT layer
        self.gat = GATConv(self.in_channels, self.out_channels, heads=4, dropout=0.3, concat=False)

        # 数值稳定的预测器
        if args.predictor == "1":
            self.predictor = nn.Sequential(
                nn.Linear(args.feature_dim, args.feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(args.feature_dim // 2, 3)
            ).to(self.device)
        elif args.predictor == "2":
            self.predictor = nn.Sequential(
                nn.Linear(args.feature_dim, args.feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(args.feature_dim // 2),
                nn.Linear(args.feature_dim // 2, args.feature_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(args.feature_dim // 4, 3)
            ).to(self.device)
        else:
            self.predictor = nn.Sequential(
                nn.Linear(args.feature_dim, args.feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(args.feature_dim // 2, 3)
            ).to(self.device)

        # 数值稳定的Focal Loss
        self.focal_loss = FocalLoss(alpha=0.6, gamma=1.8)
        self.class_weights = None

        self.to(self.device)

    def compute_class_weights(self, pos_samples, neg_samples):
        """智能的类别权重计算"""
        total = pos_samples + neg_samples
        if total == 0:
            return torch.tensor([1.0, 1.0, 1.0]).to(self.device)

        # 计算样本比例
        pos_ratio = pos_samples / total

        # 根据不平衡程度动态调整
        if pos_ratio > 0.7:  # 正样本过多
            pos_weight = 0.6
            neg_weight = 1.8
        elif pos_ratio < 0.3:  # 负样本过多
            pos_weight = 1.8
            neg_weight = 0.6
        else:  # 相对平衡
            pos_weight = 1.0
            neg_weight = 1.0

        weights = torch.tensor([pos_weight, neg_weight, 1.0]).to(self.device)
        return weights

    def encode_improved(self, edge_index_a, edge_index_b, x):
        """数值稳定的编码函数"""
        # 检查输入
        if torch.isnan(x).any():
            print("Warning: NaN in encode input")
            x = torch.nan_to_num(x, nan=0.0)

        # GAT encoding
        if edge_index_a.size(1) > 0:
            x_gat_a = F.elu(self.gat(x, edge_index_a))
        else:
            x_gat_a = torch.zeros_like(x)

        if edge_index_b.size(1) > 0:
            x_gat_b = F.elu(self.gat(x, edge_index_b))
        else:
            x_gat_b = torch.zeros_like(x)

        # GCN encoding
        x_gcn_a, x_gcn_b = x.clone(), x.clone()
        for gcn_layer in self.gcn_layers:
            if edge_index_a.size(1) > 0:
                x_gcn_a = gcn_layer(x_gcn_a, edge_index_a)
            if edge_index_b.size(1) > 0:
                x_gcn_b = gcn_layer(x_gcn_b, edge_index_b)

        # Feature fusion
        x_a = torch.cat([x_gat_a, x_gcn_a], dim=-1)
        x_b = torch.cat([x_gat_b, x_gcn_b], dim=-1)

        # 数值稳定的特征变换
        if x_a.size(0) > 0:
            x_a = self.transform(x_a)
            # 检查NaN
            if torch.isnan(x_a).any():
                print("Warning: NaN in transform output A")
                x_a = torch.nan_to_num(x_a, nan=0.0)

        if x_b.size(0) > 0:
            x_b = self.transform(x_b)
            # 检查NaN
            if torch.isnan(x_b).any():
                print("Warning: NaN in transform output B")
                x_b = torch.nan_to_num(x_b, nan=0.0)

        return x_a, x_b

    def forward(self, edge_indices, x):
        """前向传播"""
        self.x = x

        (train_pos_edge_index_a, train_neg_edge_index_a, train_pos_edge_index_b, train_neg_edge_index_b,
         diff_pos_edge_index_a, diff_neg_edge_index_a, diff_pos_edge_index_b, diff_neg_edge_index_b) = edge_indices

        train_pos_x_a, train_pos_x_b = self.encode_improved(train_pos_edge_index_a, train_pos_edge_index_b, x)
        train_neg_x_a, train_neg_x_b = self.encode_improved(train_neg_edge_index_a, train_neg_edge_index_b, x)

        diff_pos_x_a, diff_pos_x_b = self.encode_improved(diff_pos_edge_index_a, diff_pos_edge_index_b, x)
        diff_neg_x_a, diff_neg_x_b = self.encode_improved(diff_neg_edge_index_a, diff_neg_edge_index_b, x)

        return (train_pos_x_a, train_pos_x_b, diff_pos_x_a, diff_pos_x_b,
                train_neg_x_a, train_neg_x_b, diff_neg_x_a, diff_neg_x_b)

    def predict(self, x, edge_index):
        """数值稳定的预测方法"""
        if edge_index.size(1) == 0:
            return torch.empty(0, 3).to(self.device)

        h = x

        # 检查输入
        if torch.isnan(h).any():
            print("Warning: NaN in predict input")
            h = torch.nan_to_num(h, nan=0.0)

        # 通过GCN层
        if edge_index.size(1) > 0:
            for gcn_layer in self.gcn_layers:
                h = gcn_layer(h, edge_index)

        # 边嵌入
        row, col = edge_index
        edge_embeddings = h[row] * h[col]

        # 检查嵌入
        if torch.isnan(edge_embeddings).any():
            print("Warning: NaN in edge embeddings")
            edge_embeddings = torch.nan_to_num(edge_embeddings, nan=0.0)

        # 预测
        predictions = self.predictor(edge_embeddings)

        # 检查预测结果
        if torch.isnan(predictions).any():
            print("Warning: NaN in predictions")
            predictions = torch.nan_to_num(predictions, nan=0.0)

        return F.log_softmax(predictions, dim=1)

    def similarity_score(self, x_a, x_b):
        """数值稳定的相似度计算"""
        if x_a.size(0) == 0 or x_b.size(0) == 0:
            return torch.tensor(0.0).to(self.device)

        # 检查输入
        if torch.isnan(x_a).any() or torch.isnan(x_b).any():
            print("Warning: NaN in similarity input")
            x_a = torch.nan_to_num(x_a, nan=0.0)
            x_b = torch.nan_to_num(x_b, nan=0.0)

        # 温和的dropout
        x_a = F.dropout(x_a, p=0.1, training=self.training)
        x_b = F.dropout(x_b, p=0.1, training=self.training)

        # 维度匹配
        if x_a.size(0) != x_b.size(0):
            min_size = min(x_a.size(0), x_b.size(0))
            x_a = x_a[:min_size]
            x_b = x_b[:min_size]

        # 使用数值稳定的余弦相似度
        x_a_norm = F.normalize(x_a, p=2, dim=1, eps=1e-8)
        x_b_norm = F.normalize(x_b, p=2, dim=1, eps=1e-8)

        # 计算余弦相似度
        sim_score = torch.sum(x_a_norm * x_b_norm, dim=1, keepdim=True)

        # 限制范围并应用sigmoid
        sim_score = torch.clamp(sim_score, min=-1.0, max=1.0)
        sim_score = torch.sigmoid(sim_score / max(self.args.tau, 0.01))  # 防止tau太小

        # 最终检查
        if torch.isnan(sim_score).any() or torch.isinf(sim_score).any():
            print("Warning: NaN/Inf in similarity score, using default")
            sim_score = torch.full_like(sim_score, 0.5)

        return sim_score

    def loss(self, x_concat, train_pos_x_a, train_pos_x_b, train_neg_x_a, train_neg_x_b,
             diff_pos_x_a, diff_pos_x_b, diff_neg_x_a, diff_neg_x_b,
             train_pos_edge_index, train_neg_edge_index):
        """数值稳定的损失函数"""

        # 动态计算类别权重
        pos_samples = train_pos_edge_index.size(1)
        neg_samples = train_neg_edge_index.size(1)
        self.class_weights = self.compute_class_weights(pos_samples, neg_samples)

        # 更新focal loss权重
        self.focal_loss.class_weights = self.class_weights

        total_loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        # 1. 对比学习损失（添加数值稳定性）
        contrastive_loss = torch.tensor(0.0).to(self.device)
        contrastive_count = 0

        if train_pos_x_a.size(0) > 0 and train_pos_x_b.size(0) > 0:
            pos_sim_score = self.similarity_score(train_pos_x_a, train_pos_x_b)
            if pos_sim_score.numel() > 0 and not torch.isnan(pos_sim_score).any():
                pos_loss = -torch.log(torch.clamp(pos_sim_score, min=1e-8, max=1 - 1e-8)).mean()
                if not torch.isnan(pos_loss):
                    contrastive_loss += pos_loss
                    contrastive_count += 1

        if train_neg_x_a.size(0) > 0 and train_neg_x_b.size(0) > 0:
            neg_sim_score = self.similarity_score(train_neg_x_a, train_neg_x_b)
            if neg_sim_score.numel() > 0 and not torch.isnan(neg_sim_score).any():
                neg_loss = -torch.log(torch.clamp(1 - neg_sim_score, min=1e-8, max=1 - 1e-8)).mean()
                if not torch.isnan(neg_loss):
                    contrastive_loss += neg_loss
                    contrastive_count += 1

        if contrastive_count > 0:
            contrastive_loss = contrastive_loss / contrastive_count

        # 2. 扩散图对比损失（添加数值稳定性）
        diffusion_loss = torch.tensor(0.0).to(self.device)
        diffusion_count = 0

        if diff_pos_x_a.size(0) > 0 and diff_pos_x_b.size(0) > 0:
            diff_pos_sim = self.similarity_score(diff_pos_x_a, diff_pos_x_b)
            if diff_pos_sim.numel() > 0 and not torch.isnan(diff_pos_sim).any():
                diff_pos_loss = -torch.log(torch.clamp(diff_pos_sim, min=1e-8, max=1 - 1e-8)).mean()
                if not torch.isnan(diff_pos_loss):
                    diffusion_loss += diff_pos_loss
                    diffusion_count += 1

        if diff_neg_x_a.size(0) > 0 and diff_neg_x_b.size(0) > 0:
            diff_neg_sim = self.similarity_score(diff_neg_x_a, diff_neg_x_b)
            if diff_neg_sim.numel() > 0 and not torch.isnan(diff_neg_sim).any():
                diff_neg_loss = -torch.log(torch.clamp(1 - diff_neg_sim, min=1e-8, max=1 - 1e-8)).mean()
                if not torch.isnan(diff_neg_loss):
                    diffusion_loss += diff_neg_loss
                    diffusion_count += 1

        if diffusion_count > 0:
            diffusion_loss = diffusion_loss / diffusion_count

        # 3. 预测损失
        prediction_loss = torch.tensor(0.0).to(self.device)
        try:
            all_preds = []
            all_labels = []

            if train_pos_edge_index.size(1) > 0:
                pos_pred = self.predict(self.x, train_pos_edge_index)
                pos_labels = torch.zeros(pos_pred.size(0), dtype=torch.long).to(self.device)
                all_preds.append(pos_pred)
                all_labels.append(pos_labels)

            if train_neg_edge_index.size(1) > 0:
                neg_pred = self.predict(self.x, train_neg_edge_index)
                neg_labels = torch.ones(neg_pred.size(0), dtype=torch.long).to(self.device)
                all_preds.append(neg_pred)
                all_labels.append(neg_labels)

            if all_preds:
                combined_preds = torch.cat(all_preds, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)
                prediction_loss = self.focal_loss(combined_preds, combined_labels)

        except Exception as e:
            if self.debug_loss:
                print(f"Prediction loss error: {e}")
            prediction_loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        # 动态权重分配
        pos_ratio = pos_samples / (pos_samples + neg_samples) if (pos_samples + neg_samples) > 0 else 0.5

        if pos_ratio > 0.7 or pos_ratio < 0.3:  # 数据不平衡
            contrastive_weight = self.args.alpha * 0.5  # 降低对比学习权重
            diffusion_weight = self.args.beta * 0.5  # 降低扩散权重
            prediction_weight = 2.0  # 增加预测权重
        else:  # 数据相对平衡
            contrastive_weight = self.args.alpha * 0.3  # 进一步降低
            diffusion_weight = self.args.beta * 0.3  # 进一步降低
            prediction_weight = 1.5  # 适中的预测权重

        # 检查损失值是否有效
        if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
            contrastive_loss = torch.tensor(0.0).to(self.device)
        if torch.isnan(diffusion_loss) or torch.isinf(diffusion_loss):
            diffusion_loss = torch.tensor(0.0).to(self.device)
        if torch.isnan(prediction_loss) or torch.isinf(prediction_loss):
            prediction_loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        total_loss = (total_loss +
                      contrastive_weight * contrastive_loss +
                      diffusion_weight * diffusion_loss +
                      prediction_weight * prediction_loss)

        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: Invalid total loss, using prediction loss only")
            total_loss = prediction_loss

        # 调试信息
        if self.debug_loss:
            print(f"\nLoss components - Contrastive: {contrastive_loss:.4f} (w={contrastive_weight:.2f}), "
                  f"Diffusion: {diffusion_loss:.4f} (w={diffusion_weight:.2f}), "
                  f"Prediction: {prediction_loss:.4f} (w={prediction_weight:.2f})")
            print(f"Data ratio - Pos: {pos_ratio:.3f}, Class weights: {self.class_weights}")

        return total_loss

    def test(self, y_score, y_true):
        """测试函数"""
        y_score = y_score.cpu().numpy()
        y_true = y_true.cpu().numpy()

        try:
            acc = accuracy_score(y_true, y_score)
            auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
            f1 = f1_score(y_true, y_score, zero_division=0)
            precision = precision_score(y_true, y_score, average='binary', zero_division=0)
            recall = recall_score(y_true, y_score, average='binary', zero_division=0)
            micro_f1 = f1_score(y_true, y_score, average='micro', zero_division=0)
            macro_f1 = f1_score(y_true, y_score, average='macro', zero_division=0)

            if len(np.unique(y_true)) > 1:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
                aupr = metrics.auc(recall_curve, precision_curve)
            else:
                aupr = 0.0

        except Exception as e:
            print(f"Metrics calculation error: {e}")
            acc = auc = f1 = precision = recall = micro_f1 = macro_f1 = aupr = 0.0

        return acc, auc, f1, micro_f1, macro_f1, aupr, precision, recall


# 兼容性别名
CSGDN = StableCSGDN
ImprovedCSGDN = StableCSGDN
GentleImprovedCSGDN = StableCSGDN
OptimizedCSGDN = StableCSGDN