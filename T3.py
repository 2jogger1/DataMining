# ============================================================================
# 导入必要的库
# ============================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import math
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# 1. 数据预处理模块
# ============================================================================
class WeatherDataPreprocessor:
    """气象数据预处理类"""

    def __init__(self, window_size=12, test_size=0.2):
        self.window_size = window_size
        self.test_size = test_size
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

    def create_time_features(self, df):
        """创建时间周期特征"""
        timestamps = pd.to_datetime(df['date'])

        # 提取时间成分
        hours = timestamps.dt.hour
        days = timestamps.dt.dayofyear
        months = timestamps.dt.month

        # 正弦-余弦编码
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        df['day_sin'] = np.sin(2 * np.pi * days / 365)
        df['day_cos'] = np.cos(2 * np.pi * days / 365)
        df['month_sin'] = np.sin(2 * np.pi * months / 12)
        df['month_cos'] = np.cos(2 * np.pi * months / 12)

        return df

    def remove_outliers(self, df, n_std=3):
        """使用3σ原则去除异常值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(mean - n_std * std, mean + n_std * std)
        return df

    def create_sliding_windows(self, data, target):
        """创建滑动窗口"""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(target[i + self.window_size])
        return np.array(X), np.array(y)

    def prepare_data(self, data_path):
        """主预处理流程"""
        # 加载数据
        df = pd.read_csv(data_path)

        # 创建时间特征
        df = self.create_time_features(df)

        # 分离特征和目标
        feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
        features = df[feature_cols].values
        target = df['OT'].values.reshape(-1, 1)

        # 去除异常值
        features_df = pd.DataFrame(features, columns=feature_cols)
        features_df = self.remove_outliers(features_df)
        features = features_df.values

        # 划分训练测试集（保持时间顺序）
        split_idx = int(len(features) * (1 - self.test_size))

        # 训练集标准化
        X_train_raw = features[:split_idx]
        y_train_raw = target[:split_idx]

        X_train_scaled = self.feature_scaler.fit_transform(X_train_raw)
        y_train_scaled = self.target_scaler.fit_transform(y_train_raw)

        # 测试集标准化（使用训练集的参数）
        X_test_raw = features[split_idx:]
        y_test_raw = target[split_idx:]

        X_test_scaled = self.feature_scaler.transform(X_test_raw)
        y_test_scaled = self.target_scaler.transform(y_test_raw)

        # 创建滑动窗口
        X_train, y_train = self.create_sliding_windows(X_train_scaled, y_train_scaled.flatten())
        X_test, y_test = self.create_sliding_windows(X_test_scaled, y_test_scaled.flatten())

        # 从训练集中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, shuffle=False
        )

        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'scalers': (self.feature_scaler, self.target_scaler)
        }


# ============================================================================
# 2. 模型组件定义
# ============================================================================
class ConvLSTMBlock(nn.Module):
    """卷积LSTM模块 - 短期特征提取"""

    def __init__(self, input_dim, hidden_dim=64):
        super(ConvLSTMBlock, self).__init__()

        # 1D卷积LSTM
        self.conv_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 残差连接
        self.residual_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]

        # LSTM处理
        lstm_out, _ = self.conv_lstm(x)  # [batch, seq_len, hidden_dim]

        # 转置进行卷积
        lstm_out_t = lstm_out.transpose(1, 2)  # [batch, hidden_dim, seq_len]

        # 时间卷积
        conv_out = self.temporal_conv(lstm_out_t)

        # 残差连接
        if lstm_out_t.shape == conv_out.shape:
            output = conv_out + lstm_out_t
        else:
            residual = self.residual_conv(lstm_out_t)
            output = conv_out + residual

        return output.transpose(1, 2)  # [batch, seq_len, hidden_dim]


class InceptionLSTMBlock(nn.Module):
    """Inception-LSTM模块 - 中期特征提取"""

    def __init__(self, input_dim, hidden_dim=128):
        super(InceptionLSTMBlock, self).__init__()

        # 多尺度卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.branch5 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # 特征融合
        self.fusion_conv = nn.Conv1d(96, hidden_dim, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape

        # 转置进行卷积
        x_t = x.transpose(1, 2)  # [batch, input_dim, seq_len]

        # 多尺度特征提取
        branch1_out = self.branch1(x_t)
        branch3_out = self.branch3(x_t)
        branch5_out = self.branch5(x_t)

        # 特征拼接
        concat_features = torch.cat([branch1_out, branch3_out, branch5_out], dim=1)

        # 特征融合
        fused_features = F.relu(self.bn_fusion(self.fusion_conv(concat_features)))

        # 转置回LSTM需要的维度
        fused_features = fused_features.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # LSTM处理
        lstm_out, _ = self.lstm(fused_features)

        return lstm_out


class TemporalTransformerBlock(nn.Module):
    """时间Transformer模块 - 长期特征提取"""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TemporalTransformerBlock, self).__init__()

        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出投影
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]

        # 输入投影
        projected = self.input_projection(x)

        # 添加位置编码
        encoded = self.positional_encoding(projected)

        # Transformer编码
        transformer_out = self.transformer_encoder(encoded)

        # 输出投影
        output = self.output_projection(transformer_out)

        return output


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class CrossModalAttention(nn.Module):
    """跨模态时空注意力"""

    def __init__(self, feature_dim):
        super(CrossModalAttention, self).__init__()

        # 模态间注意力
        self.modal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 时间注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, feature_dim]

        # 模态间注意力（在特征维度上）
        batch_size, seq_len, feature_dim = x.shape

        # 重塑为 [batch*seq_len, 1, feature_dim] 用于模态注意力
        x_reshaped = x.reshape(-1, 1, feature_dim)

        modal_attn_out, modal_weights = self.modal_attention(
            x_reshaped, x_reshaped, x_reshaped
        )

        modal_attn_out = modal_attn_out.reshape(batch_size, seq_len, feature_dim)
        modal_out = self.norm1(x + modal_attn_out)

        # 时间注意力
        temporal_attn_out, temporal_weights = self.temporal_attention(
            modal_out, modal_out, modal_out
        )
        temporal_out = self.norm2(modal_out + temporal_attn_out)

        # 前馈网络
        ffn_out = self.ffn(temporal_out)
        output = self.norm2(temporal_out + ffn_out)

        return output, (modal_weights, temporal_weights)


class GatedFusion(nn.Module):

    def __init__(self, input_dims):
        super(GatedFusion, self).__init__()
        self.num_branches = len(input_dims)
        # forward 中先投影到 64 维再拼接，故输入维度应为 64 * 分支数
        total_dim = 64 * self.num_branches

        # 门控网络 - 修正输入维度
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, 64),  # total_dim = 64*3 = 192
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_branches),
            nn.Softmax(dim=-1)
        )

        # 特征投影（统一维度）
        self.projection_layers = nn.ModuleList([
            nn.Linear(dim, 64) for dim in input_dims
        ])

    def forward(self, branch_outputs):
        # branch_outputs: 各分支输出的列表
        # 每个分支输出: [batch, seq_len, feature_dim]

        batch_size = branch_outputs[0].shape[0]

        # 统一维度
        projected_outputs = []
        for i, output in enumerate(branch_outputs):
            # 对每个分支的特征进行投影
            projected = self.projection_layers[i](output)  # [batch, seq_len, 64]
            projected_outputs.append(projected)

        # 提取最后一个时间步的特征用于计算门控权重
        last_step_features = []
        for out in projected_outputs:
            last_step = out[:, -1, :]  # [batch, 64]
            last_step_features.append(last_step)

        # 拼接特征: [batch, 64*3] = [batch, 192]
        concat_features = torch.cat(last_step_features, dim=-1)

        # 计算门控权重: [batch, 3]
        gate_weights = self.gate_network(concat_features)

        # 加权融合
        # 初始化融合特征 [batch, seq_len, 64]
        fused = torch.zeros(batch_size, projected_outputs[0].shape[1], 64)
        fused = fused.to(projected_outputs[0].device)

        for i in range(self.num_branches):
            # 为每个样本的每个时间步应用权重
            weight = gate_weights[:, i].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            fused += weight * projected_outputs[i]

        return fused, gate_weights

class ProbabilisticDecoder(nn.Module):
    """概率预测解码器"""

    def __init__(self, input_dim, hidden_dim=128):
        super(ProbabilisticDecoder, self).__init__()

        # 均值预测分支
        self.mean_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 方差预测分支
        self.variance_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保方差为正
        )

    def forward(self, x):
        # x: 融合后的特征 [batch, feature_dim]

        # 均值预测
        mean = self.mean_predictor(x)

        # 方差预测（不确定性估计）
        variance = self.variance_predictor(x) + 1e-6  # 避免除零

        return mean, variance


# ============================================================================
# 3. 主模型定义
# ============================================================================
class MSANet(nn.Module):
    """多尺度时空注意力网络"""

    def __init__(self, input_dim, window_size=12):
        super(MSANet, self).__init__()

        self.input_dim = input_dim
        self.window_size = window_size

        # 多尺度编码器分支
        self.short_term_branch = ConvLSTMBlock(input_dim, hidden_dim=64)
        self.mid_term_branch = InceptionLSTMBlock(input_dim, hidden_dim=128)
        self.long_term_branch = TemporalTransformerBlock(input_dim, d_model=64)

        # 跨模态注意力
        self.cross_attention = CrossModalAttention(feature_dim=64)

        # 门控融合 - 输入维度为各分支的输出维度
        self.gated_fusion = GatedFusion(input_dims=[64, 128, 64])

        # 概率解码器
        self.decoder = ProbabilisticDecoder(input_dim=64)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        """前向传播"""
        # x shape: [batch, window_size, input_dim]
        batch_size = x.shape[0]

        # ========== 多尺度特征提取 ==========
        short_term_feat = self.short_term_branch(x)  # [batch, seq_len, 64]
        mid_term_feat = self.mid_term_branch(x)  # [batch, seq_len, 128]
        long_term_feat = self.long_term_branch(x)  # [batch, seq_len, 64]

        # ========== 跨模态注意力（仅在短期特征上）==========
        attended_feat, attention_weights = self.cross_attention(short_term_feat)

        # ========== 门控融合 ==========
        # 将注意力增强的短期特征替换原始短期特征
        branch_outputs = [attended_feat, mid_term_feat, long_term_feat]
        fused_features, gate_weights = self.gated_fusion(branch_outputs)

        # ========== 时间池化 ==========
        # 使用最后一个时间步的特征
        final_features = fused_features[:, -1, :]  # [batch, 64]

        # ========== 概率预测 ==========
        mean_pred, variance_pred = self.decoder(final_features)

        outputs = {
            'mean': mean_pred,
            'variance': variance_pred,
            'attention_weights': attention_weights,
            'gate_weights': gate_weights
        }

        return outputs


# ============================================================================
# 4. 损失函数
# ============================================================================
class CompositeLoss(nn.Module):
    """复合损失函数"""

    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha  # NLL损失权重
        self.beta = beta  # 平滑损失权重
        self.gamma = gamma  # 物理约束损失权重

    def forward(self, pred_mean, pred_var, target):
        # 1. 负对数似然损失（概率预测）
        nll_loss = 0.5 * torch.log(2 * math.pi * pred_var) + \
                   0.5 * ((target - pred_mean) ** 2) / pred_var
        nll_loss = nll_loss.mean()

        # 2. 平滑损失（时间连续性约束）
        smooth_loss = F.mse_loss(pred_mean[1:], pred_mean[:-1])

        # 3. 物理约束损失（温度变化率约束）
        temp_change = torch.abs(pred_mean[1:] - pred_mean[:-1])
        # 假设10分钟内温度变化不应超过5°C（按标准化后计算）
        physical_loss = F.relu(temp_change - 5.0).mean()

        # 总损失
        total_loss = (self.alpha * nll_loss +
                      self.beta * smooth_loss +
                      self.gamma * physical_loss)

        return total_loss, {'nll': nll_loss, 'smooth': smooth_loss, 'physical': physical_loss}


# ============================================================================
# 5. 训练和评估工具
# ============================================================================
class WeatherPredictorTrainer:
    """模型训练器"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = CompositeLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {'nll': 0, 'smooth': 0, 'physical': 0}

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(data)
            pred_mean, pred_var = outputs['mean'], outputs['variance']

            # 计算损失
            loss, components = self.criterion(pred_mean, pred_var, target)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key].item()

            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)

        return avg_loss, loss_components

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                pred_mean, pred_var = outputs['mean'], outputs['variance']

                loss, _ = self.criterion(pred_mean, pred_var, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=100, patience=15):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)

            # 训练
            train_loss, loss_components = self.train_epoch(train_loader)

            # 验证
            val_loss = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)

            # 打印结果
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
            print(f'Loss Components - NLL: {loss_components["nll"]:.4f}, '
                  f'Smooth: {loss_components["smooth"]:.4f}, '
                  f'Physical: {loss_components["physical"]:.4f}')

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pth')
                print('Model saved!')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

        # 加载最佳模型
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return self.history


# ============================================================================
# 6. 评估工具
# ============================================================================
class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model, target_scaler, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.target_scaler = target_scaler
        self.device = device
        self.model.eval()

    def predict(self, test_loader, return_uncertainty=True):
        """在测试集上进行预测"""
        all_preds = []
        all_targets = []
        all_vars = []
        all_attention_weights = []
        all_gate_weights = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)

                outputs = self.model(data)
                pred_mean = outputs['mean'].cpu().numpy()
                pred_var = outputs['variance'].cpu().numpy()

                # 反标准化
                pred_mean_original = self.target_scaler.inverse_transform(pred_mean)
                target_original = self.target_scaler.inverse_transform(target.numpy())

                # 方差需要调整尺度
                pred_var_original = pred_var * (self.target_scaler.scale_[0] ** 2)

                all_preds.append(pred_mean_original)
                all_targets.append(target_original)
                all_vars.append(pred_var_original)

                if 'attention_weights' in outputs:
                    all_attention_weights.append(outputs['attention_weights'])
                if 'gate_weights' in outputs:
                    all_gate_weights.append(outputs['gate_weights'].cpu().numpy())

        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        variances = np.vstack(all_vars)

        results = {
            'predictions': preds,
            'targets': targets,
            'variances': variances if return_uncertainty else None
        }

        if all_attention_weights:
            results['attention_weights'] = all_attention_weights
        if all_gate_weights:
            results['gate_weights'] = np.vstack(all_gate_weights)

        return results

    def calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        # 对称平均绝对百分比误差
        smape = 2.0 * np.mean(np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets))) * 100

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'sMAPE': smape
        }

        return metrics

    def calculate_uncertainty_metrics(self, predictions, targets, variances, confidence_level=0.95):
        """计算不确定性评估指标"""
        from scipy import stats

        # 计算z-score
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # 预测区间
        upper_bound = predictions + z_score * np.sqrt(variances)
        lower_bound = predictions - z_score * np.sqrt(variances)

        # 覆盖率
        coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))

        # 平均区间宽度
        interval_width = np.mean(upper_bound - lower_bound)

        # 校准误差
        expected_proportion = confidence_level
        calibration_error = np.abs(coverage - expected_proportion)

        return {
            'coverage': coverage,
            'interval_width': interval_width,
            'calibration_error': calibration_error,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        }

    def plot_predictions(self, predictions, targets, save_path='predictions_plot.png'):
        """可视化预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 预测值与真实值对比
        axes[0, 0].plot(targets[:200], label='True', alpha=0.7)
        axes[0, 0].plot(predictions[:200], label='Predicted', alpha=0.7)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('OT')
        axes[0, 0].set_title('Predictions vs True Values (First 200 Steps)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 残差图
        residuals = targets - predictions
        axes[0, 1].scatter(predictions, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Predicted Temperature')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 误差分布
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 分位数图
        from scipy import stats
        qq_data = stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig


def main():
    """主函数"""
    print("气象时间序列预测系统")
    print("=" * 50)

    # 1. 数据预处理
    print("\n1. 数据预处理...")
    preprocessor = WeatherDataPreprocessor(window_size=12, test_size=0.2)
    processed_data = preprocessor.prepare_data('weather_utf8.csv')

    X_train, y_train = processed_data['train']
    X_val, y_val = processed_data['val']
    X_test, y_test = processed_data['test']
    feature_scaler, target_scaler = processed_data['scalers']

    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"特征维度: {X_train.shape[2]}")
    print(f"目标变量范围: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # 检查维度
    input_dim = X_train.shape[2]
    print(f"\n输入维度: {input_dim}")
    print(f"窗口大小: {12}")
    print(f"批次大小: {32}")

    # 2. 创建数据加载器
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. 创建模型
    print("\n2. 创建模型...")
    model = MSANet(input_dim=input_dim, window_size=12)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试一个小批量
    print("\n3. 测试模型前向传播...")
    test_batch = next(iter(train_loader))
    data, target = test_batch
    print(f"测试批次数据形状: {data.shape}")
    print(f"测试批次目标形状: {target.shape}")

    # 测试模型输出
    with torch.no_grad():
        model.eval()
        outputs = model(data)
        print(f"均值预测形状: {outputs['mean'].shape}")
        print(f"方差预测形状: {outputs['variance'].shape}")
        if 'gate_weights' in outputs:
            print(f"门控权重形状: {outputs['gate_weights'].shape}")

    # 4. 训练模型
    print("\n4. 训练模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    trainer = WeatherPredictorTrainer(model, device=device)
    history = trainer.train(train_loader, val_loader, epochs=100, patience=15)

    # 5. 评估模型
    print("\n5. 评估模型...")
    evaluator = ModelEvaluator(model, target_scaler, device=device)

    # 测试集预测
    test_results = evaluator.predict(test_loader, return_uncertainty=True)

    # 计算指标
    metrics = evaluator.calculate_metrics(
        test_results['predictions'],
        test_results['targets']
    )

    print("\n测试集性能指标:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # 不确定性评估
    if test_results['variances'] is not None:
        uncertainty_metrics = evaluator.calculate_uncertainty_metrics(
            test_results['predictions'],
            test_results['targets'],
            test_results['variances'],
            confidence_level=0.95
        )

        print("\n不确定性评估:")
        print("-" * 30)
        print(f"95%置信区间覆盖率: {uncertainty_metrics['coverage']:.3f}")
        print(f"平均区间宽度: {uncertainty_metrics['interval_width']:.3f}°C")
        print(f"校准误差: {uncertainty_metrics['calibration_error']:.3f}")

    # 6. 可视化
    print("\n6. 生成可视化图表...")
    fig = evaluator.plot_predictions(
        test_results['predictions'][:1000],  # 只绘制前1000个点
        test_results['targets'][:1000],
        save_path='predictions_analysis.png'
    )

    # 7. 保存结果
    print("\n7. 保存结果...")
    results_df = pd.DataFrame({
        'True_Temperature': test_results['targets'].flatten(),
        'Predicted_Temperature': test_results['predictions'].flatten(),
        'Prediction_Variance': test_results['variances'].flatten() if test_results['variances'] is not None else None
    })

    results_df.to_csv('prediction_results.csv', index=False)

    # 保存评估指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('evaluation_metrics.csv', index=False)

    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'learning_rate': history['lr']
    })
    history_df.to_csv('training_history.csv', index=False)

    print("\n训练和评估完成！")
    print("生成的文件:")
    print("  - best_model.pth: 最佳模型权重")
    print("  - predictions_analysis.png: 可视化图表")
    print("  - prediction_results.csv: 预测结果")
    print("  - evaluation_metrics.csv: 评估指标")
    print("  - training_history.csv: 训练历史")


if __name__ == "__main__":
    main()