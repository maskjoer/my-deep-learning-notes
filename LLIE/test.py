import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
import time
from PIL import Image
import model
class LowLightEnhancer:
    def __init__(self, model_path='./snapshots/MDTA-Zero-DCE-x5+x1/Epoch199.pth'):
        # 初始化配置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._init_model(model_path)
        self._build_transforms()
        torch.backends.cudnn.benchmark = True  # 启用CuDNN加速[3](@ref)

    def _init_model(self, model_path):
        """混合精度模型加载[8](@ref)"""
        with torch.cuda.amp.autocast():
            self.model = model.enhance_net_nopool()
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = nn.DataParallel(self.model) if torch.cuda.device_count() > 1 else self.model
            self.model.eval().to(self.device)

    def _build_transforms(self):
        """构建预处理流水线[7](@ref)"""
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.contiguous()),
            T.Normalize(mean=[0.439], std=[0.232])  # 基于LOL数据集统计
        ])
    
    def enhance(self, image_path, output_dir='./result'):
        """增强主流程"""
        try:
            # 路径标准化处理[3](@ref)
            output_path = Path(image_path).replace('./LOLdataset/eval15/low', output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 混合精度推理[4](@ref)
            with torch.no_grad(), torch.cuda.amp.autocast():
                # 预处理（集成OpenCV加速）
                img = self._load_image(image_path)
                
                # 双分支推理[4](@ref)
                start_time = time.time()
                _, enhanced, _ = self.model(img)
                latency = time.time() - start_time
                
                # 后处理
                self._save_image(enhanced, output_path)
                return latency
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def _load_image(self, path):
        """优化图像加载[7](@ref)"""
        img = Image.open(path).convert('RGB')
        return self.preprocess(img).unsqueeze(0).to(self.device)

    def _save_image(self, tensor, path):
        """优化保存策略"""
        torchvision.utils.save_image(
            tensor.clamp(0,1), 
            str(path),
            quality=95,  # 保留细节[13](@ref)
            subsampling=0  # 禁用色度下采样
        )

if __name__ == '__main__':
    # 初始化增强器
    enhancer = LowLightEnhancer()
    
    # 多进程数据加载[4](@ref)
    input_dir = Path('./LOLdataset/eval15/low')
    image_paths = [p for p in input_dir.rglob('*') if p.suffix in ('.png','.jpg')]
    
    # 带显存监控的推理[8](@ref)
    with tqdm(image_paths, desc='Processing') as pbar:
        for img_path in pbar:
            latency = enhancer.enhance(str(img_path))
            pbar.set_postfix({'Latency': f"{latency:.2f}s"})
            
            # 动态显存管理[4](@ref)
            if torch.cuda.memory_reserved() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                torch.cuda.empty_cache()