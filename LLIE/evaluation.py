import os
import cv2
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def ssim_index(original, recovered):
    """计算结构相似性指数 (SSIM)"""
    score, _ = ssim(original, recovered, full=True)
    return score

def psnr_index(original, recovered):
    """计算峰值信噪比 (PSNR)"""
    score = psnr(original, recovered)
    return score

# def calculate_mse(original, recovered):
#     """计算均方误差 (MSE)"""
#     return np.mean((original - recovered) ** 2)

# def calculate_psnr(mse, max_pixel_value=255.0):
#     """计算峰值信噪比 (PSNR)"""
#     if mse == 0:
#         return 100.0  # 完全一致时PSNR设为100
#     return 10.0 * np.log10((max_pixel_value ** 2) / mse)

# def calculate_simm(original, recovered):
#     """计算SIMM（假设为MSE）"""
#     return calculate_mse(original, recovered)

def get_image_files_from_folder(folder_path):
    """获取文件夹中所有图像文件的路径及其所在文件夹名称"""
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                image_files.append((full_path, folder_name))
    return image_files

# 设置原始和恢复图像的根目录
original_root_folder = './LOLdataset/eval15/high'
recovered_root_folder = './result/Zero-DCE_eval/low'

# 检查文件夹是否存在
if not os.path.exists(original_root_folder):
    print(f"错误：原始文件夹不存在 - {original_root_folder}")
    exit()
if not os.path.exists(recovered_root_folder):
    print(f"错误：恢复文件夹不存在 - {recovered_root_folder}")
    exit()

# 获取所有原始图像文件
original_image_files = get_image_files_from_folder(original_root_folder)

results = []  # 存储所有图像结果
folder_results = {}  # 存储各文件夹的统计结果

for original_image_path, original_folder in original_image_files:
    # 构造对应的恢复图像路径
    relative_path = os.path.relpath(original_image_path, original_root_folder)
    recovered_image_path = os.path.join(recovered_root_folder, relative_path)
    
    # 检查恢复图像是否存在
    if not os.path.isfile(recovered_image_path):
        print(f"警告：跳过未找到的恢复图像 - {recovered_image_path}")
        continue
    
    # 读取图像并检查有效性
    original_image = cv2.imread(original_image_path)
    recovered_image = cv2.imread(recovered_image_path)
    if original_image is None or recovered_image is None:
        print(f"错误：无法读取图像 - {original_image_path} 或 {recovered_image_path}")
        continue
    
    # 确保图像为float32类型
    original_image = original_image.astype(np.float32)
    recovered_image = recovered_image.astype(np.float32)
    
    # 检查图像尺寸一致性
    if original_image.shape != recovered_image.shape:
        print(f"错误：图像尺寸不匹配 - {original_image_path} 和 {recovered_image_path}")
        continue
    
    # 计算评估指标
    #mse = calculate_mse(original_image, recovered_image)
    psnr = psnr_index(original_image, recovered_image)
    simm = simm(original_image, recovered_image)
    
    # 记录结果
    image_name = os.path.basename(original_image_path)
    results.append([original_folder, image_name, psnr, simm])
    
    # 更新文件夹统计
    if original_folder not in folder_results:
        folder_results[original_folder] = {'psnr': [], 'simm': []}
    folder_results[original_folder]['psnr'].append(psnr)
    folder_results[original_folder]['simm'].append(simm)

# 写入CSV文件
csv_file_path = './Zero-DCE_code/CSV/Zero-DCE_eval-evaluation.csv'
header = ['Folder Name', 'Image Name', 'PSNR', 'SIMM']

with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(results)
    
    # 添加文件夹平均结果
    writer.writerow([])  # 空行分隔
    writer.writerow(['Folder Name', 'Average PSNR', 'Average SIMM'])
    for folder, metrics in folder_results.items():
        avg_psnr = np.mean(metrics['psnr'])
        avg_simm = np.mean(metrics['simm'])
        writer.writerow([folder, avg_psnr, avg_simm])

print(f"评估完成！结果已保存至：{csv_file_path}")