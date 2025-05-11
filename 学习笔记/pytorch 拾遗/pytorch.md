
# PyTorch 中的 `squeeze()` 和 `unsqueeze()` 函数详解

## 1. 一句话总结

- **`unsqueeze(dim)`**：在指定的维度 `dim` 上**增加**一个大小为 1 的维度（升维）。
- **`squeeze(dim)`**：在指定的维度 `dim` 上**移除**大小为 1 的维度（降维），如果不指定 `dim`，则移除所有大小为 1 的维度。

---

## 2. 函数详解 & 示例

### （1）`unsqueeze(dim)`：增加维度

**作用**：在 `dim` 指定的位置插入一个大小为 1 的新维度。

#### 示例 1：1D → 2D

```python
import torch

x = torch.tensor([1, 2, 3])  # shape: (3,)
x_unsqueeze = x.unsqueeze(0)  # 在第 0 维增加一个维度
print(x_unsqueeze.shape)      # 输出: torch.Size([1, 3])
```

**结果**：

```
tensor([[1, 2, 3]])  # 变成 2D 矩阵（1行3列）
```

#### 示例 2：2D → 3D（适合 CNN 输入）

```python
x = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
x_unsqueeze = x.unsqueeze(0)         # 在第 0 维增加一个维度
print(x_unsqueeze.shape)             # 输出: torch.Size([1, 2, 2])
```

**结果**：

```
tensor([[[1, 2],
         [3, 4]]])  # 变成 3D 张量（batch_size=1, 2行2列）
```

---

### （2）`squeeze(dim)`：减少维度

**作用**：移除 `dim` 指定的维度（该维度大小必须为 1），如果不指定 `dim`，则移除所有大小为 1 的维度。

#### 示例 1：移除所有大小为 1 的维度

```python
x = torch.tensor([[[1, 2, 3]]])  # shape: (1, 1, 3)
x_squeeze = x.squeeze()           # 移除所有大小为 1 的维度
print(x_squeeze.shape)            # 输出: torch.Size([3])
```

**结果**：

```
tensor([1, 2, 3])  # 变成 1D 向量
```

#### 示例 2：指定维度移除

```python
x = torch.randn(1, 3, 1, 2)  # shape: (1, 3, 1, 2)
x_squeeze = x.squeeze(2)      # 只移除第 2 维（大小为1）
print(x_squeeze.shape)        # 输出: torch.Size([1, 3, 2])
```

---

## 3. 常见用途

| 场景                      | 使用函数         | 说明                                                                                  |
| ------------------------- | ---------------- | ------------------------------------------------------------------------------------- |
| **CNN 输入数据**    | `unsqueeze(0)` | 将 `(H, W)` 的图像变成 `(1, H, W)`，符合 `(batch, channel, height, width)` 格式 |
| **移除 batch 维度** | `squeeze(0)`   | 如果 batch_size=1，移除第 0 维                                                        |
| **匹配矩阵运算**    | `unsqueeze(1)` | 将 `(N,)` 变成 `(N, 1)`，便于广播计算                                             |
| **降维**            | `squeeze()`    | 移除所有不必要的维度                                                                  |

---

## 4. 总结

| 函数               | 作用                               | 适用场景               |
| ------------------ | ---------------------------------- | ---------------------- |
| `unsqueeze(dim)` | 在 `dim` 位置增加 1 个维度       | 扩展维度，适配网络输入 |
| `squeeze(dim)`   | 移除 `dim` 位置的大小为 1 的维度 | 压缩不必要的维度       |

**记忆技巧**：

- `unsqueeze` → "un"（解开） + "squeeze"（挤压） → **解压**（增加维度）。
- `squeeze` → **挤压**（减少维度）。

希望这个 Markdown 格式的说明对你有帮助！ 🚀
