# import torch
# import os
# import torch.nn as nn
# import torch.nn.functional as F
# from MGDB_MDTA_GDFN_CVPR2022 import MGDB, GDFN, MDTA
# from LCA import LCA
# from torch.utils.tensorboard import SummaryWriter
#
# # writer_path = 'runs/test'
# # writer = SummaryWriter(writer_path)
# # if not os.path.exists(writer_path):
# #     os.makedirs(writer_path, exist_ok=True)
#
#
# class EnhanceNet(nn.Module):
#     def __init__(self, base_dim=32, num_heads=4):
#         super().__init__()
#
#         # 基础参数
#         self.base_dim = base_dim
#         self.num_heads = num_heads
#         self.relu = nn.ReLU(inplace=True)
#
#         # 第一阶段特征提取
#         self.e_conv1 = nn.Sequential(
#             nn.Conv2d(3, base_dim, 3, 1, 1),
#             nn.InstanceNorm2d(base_dim)
#         )
#
#         # 下采样层
#         self.downsample = nn.Sequential(
#             nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(base_dim)
#         )
#
#         # 多尺度处理分支
#         self.mgdb1 = MGDB(dim=base_dim)
#         self.mgdb2 = MGDB(dim=base_dim)
#
#         # 注意力模块
#         self.lca1 = LCA(dim=base_dim, num_heads=num_heads)
#         self.lca2 = LCA(dim=base_dim, num_heads=num_heads)
#
#         # 改进的卷积层（深度可分离卷积）
#         self.e_conv2 = self._ds_conv(base_dim, base_dim)
#         self.e_conv3 = self._ds_conv(base_dim, base_dim)
#         self.e_conv4 = self._ds_conv(base_dim, base_dim)
#         #self
#         # 上采样层
#         self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
#
#         # 通道调整层
#         self.channel_adjust = nn.Conv2d(base_dim * 2, base_dim, 1)
#
#         # 最终卷积层
#         self.e_conv5 = self._ds_conv(base_dim * 2, base_dim)
#         self.e_conv6 = self._ds_conv(base_dim * 2, base_dim)
#         self.e_conv7 = nn.Conv2d(base_dim * 2, 24, 3, 1, 1)
#
#         # 初始化
#         self._init_weights()
#
#     def _ds_conv(self, in_ch, out_ch):
#         """深度可分离卷积块"""
#         return nn.Sequential(
#             nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
#             nn.Conv2d(in_ch, out_ch, 1),
#             nn.InstanceNorm2d(out_ch)
#         )
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # 确保输入已被归一化，例如在数据加载时处理
#         # 输入x应在[0,1]或[-1,1]范围内
#
#         # 第一阶段特征提取
#         x1 = self.relu(self.e_conv1(x))
#         x2 = self.downsample(x1)
#         x2 = self.relu(self.e_conv2(x2))
#         x3 = self.relu(self.e_conv3(x2))
#         x3 = self.mgdb1(x3)
#         x3 = self.relu(self.e_conv3(x3))
#         # 跨层注意力，效果不好
#         x1_downsampled = F.interpolate(x1, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False)
#         x3 = self.lca1(x3, x1_downsampled)
#         #尝试自注意力
#         #x3 = self.lca1(x3, x3)
#         x3 = self.upsample(x3)
#
#         # 第二阶段处理
#         x4 = self.relu(self.e_conv4(x3))
#         x4 = self.mgdb2(x4)
#         x4 = self.lca2(x4, x4)
#         x5 = self.channel_adjust(torch.cat([x3, x4], 1))
#         x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
#
#         x2_resized = F.interpolate(x2, size=(x5.size(2),x5.size(3)), mode='bilinear', align_corners=False)
#         x6 = self.relu(self.e_conv6(torch.cat([x2_resized, x5], 1)))
#
#         # 最终输出，限制r_i的幅度
#         x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))*0.6  # 缩小系数
#         r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
#
#         # 残差增强处理，逐步更新x
#         # 确保输入x是归一化的，例如在[-1,1]范围内
#         x = x.clone()  # 避免修改原始输入
#         x = x + r1 * (torch.pow(x, 2) - x)
#         x = x + r2 * (torch.pow(x, 2) - x)
#         x = x + r3 * (torch.pow(x, 2) - x)
#         enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
#         x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
#         x = x + r6 * (torch.pow(x, 2) - x)
#         x = x + r7 * (torch.pow(x, 2) - x)
#         enhance_image = x + r8 * (torch.pow(x, 2) - x)
#
#         return enhance_image_1, enhance_image, x_r
#
#
# # 示例使用，确保输入数据归一化
# if __name__ == "__main__":
#     net = EnhanceNet(base_dim=32).cuda()
#
#     # 假设输入数据已经归一化到[-1,1]
#     dummy_input = torch.randn(4, 3, 256, 256).cuda() * 0.5 + 0.5  # 测试时模拟归一化数据
#
#     writer.add_graph(net, dummy_input)
#     out1, out2, _ = net(dummy_input)
#     print(f"Output1 shape: {out1.shape}")
#     print(f"Output2 shape: {out2.shape}")
#     # writer.close()






import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from MGDB_MDTA_GDFN_CVPR2022 import MGDB, GDFN, MDTA
from LCA import LCA
from torch.utils.tensorboard import SummaryWriter

# writer_path = 'runs/test'
# writer = SummaryWriter(writer_path)
# if not os.path.exists(writer_path):
#     os.makedirs(writer_path, exist_ok=True)


class EnhanceNet(nn.Module):
    def __init__(self, base_dim=32, num_heads=4):
        super().__init__()

        # 基础参数
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.relu = nn.ReLU(inplace=True)

        # 第一阶段特征提取
        self.e_conv1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 3, 1, 1),
            nn.InstanceNorm2d(base_dim)
        )

        # 下采样层
        self.downsample = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_dim)
        )

        # 多尺度处理分支
        self.mgdb1 = MGDB(dim=base_dim)
        self.mgdb2 = MGDB(dim=base_dim)

        # 注意力模块
        self.lca1 = LCA(dim=base_dim, num_heads=num_heads)
        self.lca2 = LCA(dim=base_dim, num_heads=num_heads)

        # 改进的卷积层（深度可分离卷积）
        self.e_conv2 = self._ds_conv(base_dim, base_dim)
        self.e_conv3 = self._ds_conv(base_dim, base_dim)
        self.e_conv4 = self._ds_conv(base_dim, base_dim)
        #self
        # 上采样层
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # 通道调整层
        self.channel_adjust = nn.Conv2d(base_dim * 2, base_dim, 1)

        # 最终卷积层
        self.e_conv5 = self._ds_conv(base_dim * 2, base_dim)
        self.e_conv6 = self._ds_conv(base_dim * 2, base_dim)
        self.e_conv7 = nn.Conv2d(base_dim * 2, 24, 3, 1, 1)

        # 初始化
        self._init_weights()

    def _ds_conv(self, in_ch, out_ch):
        """深度可分离卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.InstanceNorm2d(out_ch)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 确保输入已被归一化，例如在数据加载时处理
        # 输入x应在[0,1]或[-1,1]范围内

        # 第一阶段特征提取
        x1 = self.relu(self.e_conv1(x))
        x2 = self.downsample(x1)
        x2 = self.relu(self.e_conv2(x2))
        x3 = self.relu(self.e_conv3(x2))
        x3 = self.mgdb1(x3)
        x3 = self.relu(self.e_conv3(x3))
        # 跨层注意力
        x1_downsampled = F.interpolate(x1, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False)
        x3 = self.lca1(x3, x1_downsampled)
        x3 = self.upsample(x3)

        # 第二阶段处理
        x4 = self.relu(self.e_conv4(x3))
        x4 = self.mgdb2(x4)
        x4 = self.lca2(x4, x4)
        x5 = self.channel_adjust(torch.cat([x3, x4], 1))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))

        x2_resized = F.interpolate(x2, size=(x5.size(2),x5.size(3)), mode='bilinear', align_corners=False)
        x6 = self.relu(self.e_conv6(torch.cat([x2_resized, x5], 1)))

        # 最终输出，限制r_i的幅度
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1))) # 缩小系数
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # 残差增强处理，逐步更新x
        # 确保输入x是归一化的，例如在[-1,1]范围内
        x = x.clone()  # 避免修改原始输入
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        return enhance_image_1, enhance_image, x_r



class Zero_DCE(nn.Module):

    def __init__(self):
        super(Zero_DCE, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)
        # Adapool
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # print(x3.shape)

        #x3 = mgdb_model(x2)
        # p3 = self.maxpool(x3)

        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r


if __name__ == "__main__":
    net = EnhanceNet(base_dim=32).cuda()

    # 假设输入数据已经归一化到[-1,1]
    dummy_input = torch.randn(4, 3, 256, 256).cuda() * 0.5 + 0.5  # 测试时模拟归一化数据

    writer.add_graph(net, dummy_input)
    out1, out2, _ = net(dummy_input)
    print(f"Output1 shape: {out1.shape}")
    print(f"Output2 shape: {out2.shape}")
    # writer.close()