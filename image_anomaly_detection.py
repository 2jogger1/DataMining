import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AnomalyDataset(Dataset):
    """异常检测数据集类"""
    def __init__(self, data_dirs, transform=None, mode='train'):
        """
        Args:
            data_dirs: 数据目录列表
            transform: 图像变换
            mode: 'train' 或 'test'
        """
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []
        self.categories = []
        
        print(f"\nInitializing dataset for {mode} mode")
        print(f"Data directories to search: {data_dirs}")
        
        for data_dir in data_dirs:
            # 获取类别名称
            if 'train' in data_dir or 'test' in data_dir:
                # 从路径中提取类别名称，例如 'hazelnut/train' -> 'hazelnut'
                category = data_dir.split('/')[0] if '/' in data_dir else data_dir.split('\\')[0]
            else:
                category = os.path.basename(data_dir)
            
            print(f"\nProcessing category: {category}")
            print(f"Full path: {os.path.abspath(data_dir)}")
            
            if mode == 'train':
                # 训练集：加载good和bad
                good_dir = os.path.join(data_dir, 'good')
                bad_dir = os.path.join(data_dir, 'bad')
                
                print(f"Looking for good images in: {good_dir}")
                print(f"Exists: {os.path.exists(good_dir)}")
                
                if os.path.exists(good_dir):
                    good_images = [f for f in os.listdir(good_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    print(f"Found {len(good_images)} good images")
                    
                    for img_name in good_images:
                        img_path = os.path.join(good_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(0)  # 0表示正常
                        self.categories.append(category)
                
                print(f"Looking for bad images in: {bad_dir}")
                print(f"Exists: {os.path.exists(bad_dir)}")
                
                if os.path.exists(bad_dir):
                    bad_images = [f for f in os.listdir(bad_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    print(f"Found {len(bad_images)} bad images")
                    
                    for img_name in bad_images:
                        img_path = os.path.join(bad_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(1)  # 1表示异常
                        self.categories.append(category)
            
            elif mode == 'test':
                # 测试集：直接加载test目录下的所有图片
                print(f"Looking for test images in: {data_dir}")
                print(f"Exists: {os.path.exists(data_dir)}")
                
                if os.path.exists(data_dir):
                    test_images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    print(f"Found {len(test_images)} test images")
                    
                    for img_name in test_images:
                        img_path = os.path.join(data_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(-1)  # 测试集标签未知
                        self.categories.append(category)
        
        if len(self.image_paths) == 0:
            print(f"\nWARNING: No images found for {mode} mode!")
            print("Current working directory:", os.getcwd())
            print("Available directories:", os.listdir('.'))
            
            # 检查hazelnut和zipper目录是否存在
            if os.path.exists('hazelnut'):
                print("\nhazelnut directory contents:", os.listdir('hazelnut'))
                if os.path.exists('hazelnut/train'):
                    print("hazelnut/train contents:", os.listdir('hazelnut/train'))
                if os.path.exists('hazelnut/test'):
                    print("hazelnut/test contents:", os.listdir('hazelnut/test'))
            
            if os.path.exists('zipper'):
                print("\nzipper directory contents:", os.listdir('zipper'))
                if os.path.exists('zipper/train'):
                    print("zipper/train contents:", os.listdir('zipper/train'))
                if os.path.exists('zipper/test'):
                    print("zipper/test contents:", os.listdir('zipper/test'))
        else:
            print(f"\nSuccessfully loaded {len(self.image_paths)} images for {mode} mode")
            print(f"Categories distribution:")
            unique_categories = set(self.categories)
            for cat in unique_categories:
                cat_count = sum(1 for c in self.categories if c == cat)
                print(f"  {cat}: {cat_count} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx], self.categories[idx], img_path

class Autoencoder(nn.Module):
    """自编码器模型"""
    def __init__(self, input_dim=2048, latent_dim=512):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  # 输出在0-1之间
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class FeatureExtractor:
    """特征提取器"""
    def __init__(self, model_name='resnet50', use_pretrained=True):
        self.model_name = model_name
        self.device = device
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 加载预训练模型
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Initialized feature extractor with {model_name}")
    
    def extract_features(self, dataloader):
        """提取所有图像的特征"""
        features = []
        labels = []
        categories = []
        image_paths = []
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_categories, batch_paths in tqdm(dataloader, desc="Extracting features"):
                batch_images = batch_images.to(self.device)
                
                # 提取特征
                batch_features = self.model(batch_images)
                batch_features = batch_features.view(batch_features.size(0), -1)
                
                features.append(batch_features.cpu().numpy())
                labels.append(batch_labels.numpy())
                categories.extend(batch_categories)
                image_paths.extend(batch_paths)
        
        features = np.vstack(features)
        labels = np.concatenate(labels)
        
        print(f"Extracted features shape: {features.shape}")
        return features, labels, categories, image_paths

class AnomalyDetector:
    """异常检测器"""
    def __init__(self, category, input_dim=2048, latent_dim=512):
        self.category = category
        self.autoencoder = Autoencoder(input_dim, latent_dim).to(device)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.threshold = None
        
    def train(self, normal_features, epochs=50, batch_size=32):
        """训练自编码器（仅使用正常样本）"""
        self.autoencoder.train()
        
        # 转换为张量
        normal_tensor = torch.FloatTensor(normal_features).to(device)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(normal_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch_data = batch[0]
                
                self.optimizer.zero_grad()
                reconstructed = self.autoencoder(batch_data)
                loss = self.criterion(reconstructed, batch_data)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # 计算正常样本的重构误差作为阈值基准
        self.autoencoder.eval()
        with torch.no_grad():
            normal_tensor = torch.FloatTensor(normal_features).to(device)
            reconstructed = self.autoencoder(normal_tensor)
            normal_errors = torch.mean((reconstructed - normal_tensor) ** 2, dim=1).cpu().numpy()
        
        # 设置阈值为正常样本重构误差的均值+3倍标准差
        self.threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
        print(f"  Set threshold for {self.category}: {self.threshold:.6f}")
        
        return train_losses
    
    def predict(self, features):
        """预测样本是否异常"""
        self.autoencoder.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(device)
            reconstructed = self.autoencoder(features_tensor)
            errors = torch.mean((reconstructed - features_tensor) ** 2, dim=1).cpu().numpy()
            
            # 根据阈值判断是否异常
            predictions = (errors > self.threshold).astype(int)
            
        return predictions, errors

def evaluate_predictions(y_true, y_pred, y_scores=None):
    """评估预测结果"""
    metrics = {}
    
    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC（如果提供了分数）
    if y_scores is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['auc_roc'] = 0.0
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    if 'auc_roc' in metrics:
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics

def visualize_results(categories, errors, predictions, true_labels=None, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    unique_categories = np.unique(categories)
    
    for i, category in enumerate(unique_categories):
        if i >= 4:  # 最多显示4个类别
            break
            
        row, col = i // 2, i % 2
        cat_mask = np.array(categories) == category
        
        cat_errors = np.array(errors)[cat_mask]
        cat_preds = np.array(predictions)[cat_mask]
        
        # 创建子图
        if true_labels is not None:
            cat_true = np.array(true_labels)[cat_mask]
            normal_errors = cat_errors[cat_true == 0]
            anomaly_errors = cat_errors[cat_true == 1]
            
            axes[row, col].hist(normal_errors, bins=30, alpha=0.7, label='Normal', color='green')
            axes[row, col].hist(anomaly_errors, bins=30, alpha=0.7, label='Anomaly', color='red')
        else:
            normal_errors = cat_errors[cat_preds == 0]
            anomaly_errors = cat_errors[cat_preds == 1]
            
            axes[row, col].hist(normal_errors, bins=30, alpha=0.7, label='Pred Normal', color='blue')
            axes[row, col].hist(anomaly_errors, bins=30, alpha=0.7, label='Pred Anomaly', color='orange')
        
        axes[row, col].set_title(f'{category} - Error Distribution')
        axes[row, col].set_xlabel('Reconstruction Error')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def save_model(detector, feature_extractor, category, save_dir):
    """保存模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存自编码器
    model_path = os.path.join(save_dir, f'{category}_autoencoder.pth')
    torch.save(detector.autoencoder.state_dict(), model_path)
    
    # 保存阈值
    threshold_path = os.path.join(save_dir, f'{category}_threshold.npy')
    np.save(threshold_path, detector.threshold)
    
    print(f"Saved model for {category} to {save_dir}")

def load_model(category, input_dim=2048, latent_dim=512, save_dir='models'):
    """加载模型"""
    detector = AnomalyDetector(category, input_dim, latent_dim)
    
    # 加载自编码器
    model_path = os.path.join(save_dir, f'{category}_autoencoder.pth')
    detector.autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    detector.autoencoder.eval()
    
    # 加载阈值
    threshold_path = os.path.join(save_dir, f'{category}_threshold.npy')
    detector.threshold = np.load(threshold_path)
    
    return detector

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"切换工作目录到: {script_dir}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"当前目录内容: {os.listdir('.')}")

    # 配置参数
    DATA_DIR = "."  # 当前目录
    LABEL_FILE = "image_anomaly_labels.json"
    OUTPUT_DIR = "anomaly_results"
    MODEL_DIR = "anomaly_models"
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 2.1 处理图像特征
    print("=" * 60)
    print("Step 1: Processing Image Features")
    print("=" * 60)
    
    # 初始化特征提取器
    feature_extractor = FeatureExtractor(model_name='resnet50', use_pretrained=True)
    
    # 准备训练和测试数据路径
    categories = ['hazelnut', 'zipper']

    train_dirs = []
    test_dirs = []

    for category in categories:
        train_dir = os.path.join(category, 'train')
        test_dir = os.path.join(category, 'test')
        
        print(f"\nChecking category: {category}")
        print(f"Train dir exists: {os.path.exists(train_dir)}")
        print(f"Test dir exists: {os.path.exists(test_dir)}")
        
        if os.path.exists(train_dir):
            train_dirs.append(train_dir)
        
        if os.path.exists(test_dir):
            test_dirs.append(test_dir)
    
    # 创建数据集
    train_dataset = AnomalyDataset(train_dirs, transform=feature_extractor.transform, mode='train')
    test_dataset = AnomalyDataset(test_dirs, transform=feature_extractor.transform, mode='test')
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 提取特征
    print("\nExtracting training features...")
    train_features, train_labels, train_categories, train_paths = feature_extractor.extract_features(train_dataloader)
    
    print("\nExtracting testing features...")
    test_features, test_labels, test_categories, test_paths = feature_extractor.extract_features(test_dataloader)
    
    # 加载测试标签
    print("\nLoading test labels...")
    with open(LABEL_FILE, 'r') as f:
        test_label_dict = json.load(f)
    
    # 为测试集创建真实标签
    test_true_labels = []
    for path in test_paths:
        # 从路径中提取相对路径
        rel_path = os.path.relpath(path, DATA_DIR)
        rel_path = rel_path.replace('\\', '/')  # 统一为Linux路径格式
        
        if rel_path in test_label_dict:
            label_str = test_label_dict[rel_path]['label']
            test_true_labels.append(0 if label_str == 'good' else 1)
        else:
            # 如果没有找到标签，标记为-1
            test_true_labels.append(-1)
    
    test_true_labels = np.array(test_true_labels)
    valid_test_mask = test_true_labels != -1
    print(f"Valid test samples: {np.sum(valid_test_mask)}/{len(test_true_labels)}")
    
    # 2.2 完成异常检测模型
    print("\n" + "=" * 60)
    print("Step 2: Building Anomaly Detection Models")
    print("=" * 60)
    
    detectors = {}
    all_predictions = []
    all_errors = []
    
    for category in categories:
        print(f"\nTraining anomaly detector for {category}...")
        
        # 获取该类别的训练正常样本
        category_mask = np.array(train_categories) == category
        normal_mask = (train_labels == 0) & category_mask
        
        if np.sum(normal_mask) == 0:
            print(f"No normal samples found for {category}, skipping...")
            continue
        
        normal_features = train_features[normal_mask]
        
        # 创建并训练异常检测器
        detector = AnomalyDetector(category, input_dim=train_features.shape[1])
        train_losses = detector.train(normal_features, epochs=50, batch_size=32)
        
        # 保存模型
        save_model(detector, feature_extractor, category, MODEL_DIR)
        
        # 对测试集进行预测
        category_test_mask = np.array(test_categories) == category
        if np.sum(category_test_mask) > 0:
            category_test_features = test_features[category_test_mask]
            predictions, errors = detector.predict(category_test_features)
            
            # 存储结果
            all_predictions.extend(predictions)
            all_errors.extend(errors)
            
            detectors[category] = detector
            
            # 评估该类别的结果（如果有真实标签）
            category_valid_mask = category_test_mask & valid_test_mask
            if np.sum(category_valid_mask) > 0:
                category_true = test_true_labels[category_valid_mask]
                category_pred = np.array(predictions)[np.array(category_test_mask)[category_valid_mask]]
                category_scores = np.array(errors)[np.array(category_test_mask)[category_valid_mask]]
                
                print(f"\nResults for {category}:")
                evaluate_predictions(category_true, category_pred, category_scores)
    
    all_predictions = np.array(all_predictions)
    all_errors = np.array(all_errors)
    
    # 2.3 评估异常检测效果
    print("\n" + "=" * 60)
    print("Step 3: Evaluating Anomaly Detection Performance")
    print("=" * 60)
    
    # 总体评估
    print("\nOverall Results:")
    overall_metrics = evaluate_predictions(
        test_true_labels[valid_test_mask],
        all_predictions[valid_test_mask],
        all_errors[valid_test_mask]
    )
    
    # 可视化结果
    print("\n" + "=" * 60)
    print("Visualizing Results")
    print("=" * 60)
    
    visualize_results(
        test_categories,
        all_errors,
        all_predictions,
        test_true_labels,
        save_path=os.path.join(OUTPUT_DIR, 'anomaly_detection_results.png')
    )
    
    # 保存预测结果
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    results = {}
    for i, (path, pred, error) in enumerate(zip(test_paths, all_predictions, all_errors)):
        rel_path = os.path.relpath(path, DATA_DIR)
        rel_path = rel_path.replace('\\', '/')
        
        results[rel_path] = {
            'prediction': 'anomaly' if pred == 1 else 'normal',
            'error_score': float(error),
            'true_label': 'unknown'
        }
        
        if rel_path in test_label_dict:
            results[rel_path]['true_label'] = test_label_dict[rel_path]['label']
    
    results_file = os.path.join(OUTPUT_DIR, 'anomaly_predictions.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to {results_file}")
    
    # 保存评估指标
    metrics_file = os.path.join(OUTPUT_DIR, 'anomaly_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_file}")
    
    # 保存错误分析
    error_analysis = {
        'thresholds': {cat: float(detectors[cat].threshold) for cat in detectors},
        'mean_error': float(np.mean(all_errors)),
        'std_error': float(np.std(all_errors)),
        'min_error': float(np.min(all_errors)),
        'max_error': float(np.max(all_errors))
    }
    
    error_file = os.path.join(OUTPUT_DIR, 'error_analysis.json')
    with open(error_file, 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    print(f"Error analysis saved to {error_file}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"1. Trained anomaly detectors for categories: {list(detectors.keys())}")
    print(f"2. Overall F1 Score: {overall_metrics['f1_score']:.4f}")
    print(f"3. Models saved to: {MODEL_DIR}")
    print(f"4. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()