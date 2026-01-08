import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ImageDataset(Dataset):
    """图像数据集类"""
    def __init__(self, data_dir, label_file=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.image_names = []
        self.labels = []
        
        # 收集所有图像文件
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {data_dir}")
        
        print(f"Found {len(image_files)} image files")
        
        for img_name in sorted(image_files):
            img_path = os.path.join(data_dir, img_name)
            self.image_paths.append(img_path)
            self.image_names.append(img_name)
        
        # 如果有标签文件，加载标签
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r') as f:
                label_dict = json.load(f)
            
            # 按文件名匹配标签
            for img_name in self.image_names:
                if img_name in label_dict:
                    self.labels.append(label_dict[img_name])
                else:
                    self.labels.append(None)
        else:
            self.labels = [None] * len(self.image_paths)
        
        print(f"Dataset initialized with {len(self.image_paths)} images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx] if self.labels[idx] is not None else -1
        
        return image, label, self.image_names[idx]

class FeatureExtractor:
    """特征提取器"""
    def __init__(self, model_name='resnet50', use_pretrained=True):
        self.model_name = model_name
        self.device = device
        
        # 定义图像预处理（与ImageNet预训练模型相同）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 加载预训练模型
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            # 移除最后的全连接层，获取特征
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None)
            # 移除最后的分类层
            self.model = torch.nn.Sequential(*list(model.features), 
                                             torch.nn.AdaptiveAvgPool2d((7, 7)),
                                             torch.nn.Flatten())
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        print(f"Initialized feature extractor with {model_name}")
    
    def extract_features(self, dataloader):
        """提取所有图像的特征"""
        features = []
        labels = []
        image_names = []
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_names in tqdm(dataloader, desc="Extracting features"):
                batch_images = batch_images.to(self.device)
                
                # 提取特征
                batch_features = self.model(batch_images)
                batch_features = batch_features.view(batch_features.size(0), -1)
                
                features.append(batch_features.cpu().numpy())
                labels.append(np.array(batch_labels))
                image_names.extend(batch_names)
        
        features = np.vstack(features)
        labels = np.concatenate(labels)
        
        print(f"Extracted features shape: {features.shape}")
        return features, labels, image_names

class ClusteringModel:
    """聚类模型"""
    def __init__(self, n_clusters=6, method='kmeans', n_components=50):
        self.n_clusters = n_clusters
        self.method = method
        self.n_components = n_components
        self.pca = None
        self.cluster_model = None
        
    def reduce_dimensions(self, features, method='pca'):
        """降维处理"""
        if method == 'pca':
            n_comp = min(self.n_components, features.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=42)
            reduced_features = self.pca.fit_transform(features)
            print(f"PCA reduced to {n_comp} dimensions, explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        elif method == 'tsne':
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
            reduced_features = tsne.fit_transform(features)
        else:
            reduced_features = features
        
        return reduced_features
    
    def fit_predict(self, features):
        """训练聚类模型并预测"""
        # 降维
        reduced_features = self.reduce_dimensions(features, method='pca')
        
        # 选择聚类算法
        if self.method == 'kmeans':
            n_clusters = min(self.n_clusters, len(features))
            self.cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")
        
        # 训练并预测
        cluster_labels = self.cluster_model.fit_predict(reduced_features)
        
        return cluster_labels, reduced_features
    
    def evaluate(self, cluster_labels, true_labels=None, features=None):
        """评估聚类效果"""
        metrics = {}
        
        if true_labels is not None and len(np.unique(true_labels[true_labels != -1])) > 1:
            # 过滤掉未标记的数据
            valid_labels = ['cable', 'tile', 'bottle', 'pill', 'leather', 'transistor']
            mask = np.array([label in valid_labels for label in true_labels])
            if mask.sum() > 0:
                filtered_clusters = cluster_labels[mask]
                filtered_true = true_labels[mask]
                
                # 调整兰德指数
                ari = adjusted_rand_score(filtered_true, filtered_clusters)
                metrics['ARI'] = ari
                print(f"Adjusted Rand Index: {ari:.4f}")
                
                # 归一化互信息
                nmi = normalized_mutual_info_score(filtered_true, filtered_clusters)
                metrics['NMI'] = nmi
                print(f"Normalized Mutual Information: {nmi:.4f}")
        
        if features is not None and len(features) > 1:
            # 轮廓系数（不需要真实标签）
            try:
                if len(np.unique(cluster_labels)) > 1:
                    silhouette = silhouette_score(features, cluster_labels)
                    metrics['Silhouette'] = silhouette
                    print(f"Silhouette Score: {silhouette:.4f}")
            except:
                print("Could not compute silhouette score")
        
        return metrics

def visualize_clusters(features_2d, cluster_labels, true_labels=None, save_path=None):
    """可视化聚类结果"""
    plt.figure(figsize=(15, 5))
    
    # 子图1：聚类结果
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=cluster_labels, cmap='tab20', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title('Clustering Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # 子图2：真实标签（如果有）
    if true_labels is not None and len(np.unique(true_labels[true_labels != -1])) > 1:
        plt.subplot(1, 2, 2)
        valid_labels = ['cable', 'tile', 'bottle', 'pill', 'leather', 'transistor']
        mask = np.array([label in valid_labels for label in true_labels])
        if mask.sum() > 0:
            # 将字符串标签转换为数字标签
            unique_labels = np.unique(true_labels[mask])
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_to_idx[label] if label in label_to_idx else -1 
                                       for label in true_labels])
            
            scatter = plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                  c=numeric_labels[mask], cmap='tab20', alpha=0.7, s=50)
            plt.colorbar(scatter)
            plt.title('Ground Truth Labels')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def save_clustering_results(image_names, cluster_labels, true_labels, output_file='clustering_results.json'):
    """保存聚类结果"""
    results = {}
    for i, img_name in enumerate(image_names):
        results[img_name] = {
            'cluster_id': int(cluster_labels[i]),
            'true_label': true_labels[i] if true_labels[i] != -1 else 'unknown'
        }
    
    # 统计每个簇的大小
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = {f'cluster_{int(c)}': int(count) for c, count in zip(unique_clusters, cluster_counts)}
    
    results['statistics'] = cluster_stats
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Clustering results saved to {output_file}")
    
    # 打印统计信息
    print("\nCluster Statistics:")
    for cluster_id in sorted(unique_clusters):
        print(f"  Cluster {int(cluster_id)}: {cluster_counts[cluster_id]} samples")

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置参数
    DATA_DIR = os.path.join(current_dir, "dataset")  # 图像文件夹路径
    LABEL_FILE = os.path.join(current_dir, "cluster_labels.json")  # 标签文件路径
    OUTPUT_DIR = os.path.join(current_dir, "results")  # 输出目录
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1.1 处理图像特征
    print("=" * 50)
    print("Step 1: Processing Image Features")
    print("=" * 50)
    
    # 首先创建特征提取器（这会包含transform）
    feature_extractor = FeatureExtractor(model_name='resnet50', use_pretrained=True)
    
    # 创建数据集，传入特征提取器的transform
    dataset = ImageDataset(DATA_DIR, LABEL_FILE, transform=feature_extractor.transform)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # 提取特征
    features, true_labels, image_names = feature_extractor.extract_features(dataloader)
    
    # 1.2 选择合适的聚类算法
    print("\n" + "=" * 50)
    print("Step 2: Clustering with K-means Algorithm")
    print("=" * 50)
    
    # 初始化聚类模型
    clustering_model = ClusteringModel(
        n_clusters=6,  # 已知有6个类别
        method='kmeans',
        n_components=50  # PCA降维到50维
    )
    
    # 执行聚类
    cluster_labels, reduced_features = clustering_model.fit_predict(features)
    
    # 1.3 评估聚类效果
    print("\n" + "=" * 50)
    print("Step 3: Evaluating Clustering Performance")
    print("=" * 50)
    
    # 评估聚类结果
    metrics = clustering_model.evaluate(
        cluster_labels, 
        true_labels, 
        reduced_features
    )
    
    # 可视化
    print("\n" + "=" * 50)
    print("Visualizing Clustering Results")
    print("=" * 50)
    
    # 使用t-SNE进行2D可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # 可视化聚类结果
    visualize_clusters(
        features_2d, 
        cluster_labels, 
        true_labels,
        save_path=os.path.join(OUTPUT_DIR, 'clustering_visualization.png')
    )
    
    # 保存结果
    save_clustering_results(
        image_names,
        cluster_labels,
        true_labels,
        output_file=os.path.join(OUTPUT_DIR, 'clustering_results.json')
    )
    
    # 保存评估指标
    metrics_file = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()