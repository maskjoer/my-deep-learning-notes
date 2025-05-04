import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import os
import time
from torchvision import transforms
from PIL import Image
import glob

# 导入自定义模型模块
import model  # 确保 model.py 在相同目录下，或者使用正确的模块路径

# 环境配置（只需设置一次）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True  # 加速卷积运算

# 初始化模型（全局加载一次）
def initialize_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = model.EnhanceNet().to(device)  # 使用 model_instance 避免命名冲突
    model_instance.load_state_dict(torch.load(model_path))
    model_instance.eval()  # 设置为评估模式
    return model_instance

# 图像处理函数
def process_image(image_path, model_instance, output_size=(1024,1024)):
    # 预处理管道
    preprocess = transforms.Compose([
        transforms.ToTensor(),
         #transforms.Resize(output_size) if output_size else lambda x: x
    ])

    with torch.no_grad():  # 禁用梯度计算
        # 加载并预处理图像
        orig_image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(orig_image).unsqueeze(0).cuda()

        # 推理计时
        start_time = time.time()
        _, enhanced, _ = model_instance(input_tensor)
        inference_time = time.time() - start_time
        #修改保存路径
        #output_dir = os.path.join('./result', 'LIME_EN')#####################
        output_dir = os.path.join('./result','re_new_model0.7_o')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving to: {output_dir}")
        output_path = os.path.join(output_dir, os.path.basename(image_path))

        # 保存结果
        torchvision.utils.save_image(enhanced, output_path)
        return inference_time

if __name__ == '__main__':
    # 初始化模型
    model_instance = initialize_model('snapshots/reflectance1.0/Epoch99.pth')
    #print(model_instance)
    # 遍历处理图像
    #base_path = '/home/shiyu/Code/VScode/Github/LLIE/zero_dce (copy)/Zero-DCE_code/data/test'
    #base_path = '/home/shiyu/Downloads/LLIE_Datasets/焊接/lowlight/image'
    #base_path = '/home/shiyu/Code/VScode/Github/LLIE/RetinexNet_PyTorch/results//re/LOLv1/low'
    base_path = '/home/shiyu/Downloads/LLIE_Datasets/LOLdataset/eval15/low'
    #base_path = '/home/shiyu/Downloads/LLIE_Datasets/LOLv2/Real_captured/Test/Low'
    #base_path = './data/re_lowlight_image'
    total_time = 0
    file_count = 0

    # 支持多种图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for root, _, _ in os.walk(base_path):
        for ext in image_extensions:
            for image_path in glob.glob(os.path.join(root, ext)):
                print(f"Processing: {image_path}")
                process_time = process_image(image_path, model_instance)
                total_time += process_time
                file_count += 1
                print(f"Inference time: {process_time:.4f}s")

    # 输出统计信息
    if file_count > 0:
        print(f"\nTotal processed: {file_count} images")
        print(f"Average time per lowlight_image: {total_time/file_count:.4f}s")
    else:
        print("No images found in the specified directory.")