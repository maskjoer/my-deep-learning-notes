import torch
import torch.nn as nn
from torchvision.models import vgg16
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = vgg16(weights=vgg16().IMAGENET1K_V1).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}

        # 通过每一层并保存特征
        x = self.to_relu_1_2[:4](x)  # 经过第一组卷积
        features['relu_1_2'] = x  # 保存第一组卷积层的特征

        x = self.to_relu_2_2[:5](x)  # 经过第二组卷积
        features['relu_2_2'] = x  # 保存第二组卷积层的特征

        x = self.to_relu_3_3[:7](x)  # 经过第三组卷积
        features['relu_3_3'] = x  # 保存第三组卷积层的特征

        # 返回特征字典
        return features

# 实例化Vgg16模型
model = Vgg16().cuda()

# 定义图像预处理（调整大小、裁剪、转为Tensor、归一化）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = 'Zero-DCE_code/data/test_data/coverdata/00-21-49-06-1E-0E_2024-11-12-15-49-10_00016.jpg'
img = Image.open(img_path)
img_tensor = preprocess(img).unsqueeze(0).cuda()

# 使用Vgg16模型来提取特征
output = model(img_tensor)
features = output['relu_2_2']
print(features)
plt.imshow(features[0, 0].cpu().detach().numpy(), cmap='gray')
plt.show()

