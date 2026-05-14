# Signal_System_Project

## 人员1
把原始的含噪音频变成了可直接训练的纯净数据。具体核心产出为以下四点：
1. 基础建设： 搞定了 Python 环境依赖，搭建了 GitHub 协作仓库并完成了团队授权。
2. 可视化与报告： 跑出了项目要求的 4 张核心图表（波形图、频谱图、语谱图、MFCC），并写完了它们对应的物理原理解析。
3. 极限数据清洗（高光点）： 优化了端点检测（VAD）算法，利用“时长过滤”和“能量排行”机制，完美剔除了录音里的杂音和呼吸声，精准截取了 0~9 的发音片段。
4. 交付“黄金数据”： 将变长的物理声音降维成了 13 维定长的数学特征，打包生成了纯净、零错位的 speech_features.npz 供下一个负责模型的同学直接调用。
一句话总结：
“把一堆MP3，提炼成了自带标签的完美数学矩阵，铺平了后面建模型和跑测试的路”

后续根据训练需求修改：特征归一化与时序重构：对输入的静态MFCC均值特征（13维）进行标准化处理；后续升级为完整的40帧×39维动态MFCC时序特征（含一阶、二阶差分），保留了发音的动态过程。

## 人员2

把成员1提供的纯净特征数据训练成了一个可用的数字分类模型。具体核心产出为以下三点：

1. **模型设计与调优**：搭建了轻量级1D-CNN网络（深度可分离卷积 + 全局平均池化），参数量仅5.2万，适配小样本。引入Mixup数据增强、Focal Loss损失函数，并通过早停与学习率衰减防止过拟合。
2. **性能评估与报告**：在干净验证集上达到75%识别准确率，输出了混淆矩阵及分类报告（数字0‑4全对，5‑9各对1个）。完成了模型结构、训练参数、结果分析的文字说明，供报告使用。
3. **交付“可部署模型”**：保存了训练好的Keras模型（.h5文件）及配套的归一化器（.pkl文件），确保噪声测试时预处理与训练完全一致。

一句话总结：  
“把成员1的数学矩阵，训练成了一个能听懂数字、能抗住噪声的神经网络，并打包成即插即用的模型文件。”

---

## 给人员3的产出文件及使用方法

### 产出文件清单
1. `speech_model.h5` — 训练好的1D-CNN模型（Keras格式，含网络结构与权重）
2. `scaler.pkl` — 特征归一化器（StandardScaler，用训练集拟合）


### 如何使用（噪声测试时）

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# 1. 加载模型和归一化器
model = load_model('speech_model.h5')
scaler = joblib.load('scaler.pkl')

# 2. 预处理函数（必须与训练时完全相同）
def preprocess(X_raw):
    # X_raw: (n_samples, 40, 39) 的 numpy 数组，dtype=float
    original_shape = X_raw.shape
    X_flat = X_raw.reshape(-1, X_raw.shape[-1])
    X_flat_scaled = scaler.transform(X_flat)
    return X_flat_scaled.reshape(original_shape).astype(np.float32)

# 3. 对带噪验证集进行测试
X_test_noisy = ...   # 你已经添加好噪声的数据，形状 (20,40,39)
X_test_scaled = preprocess(X_test_noisy)
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

# 4. 计算准确率
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_val, y_pred)   # Y_val 是真实标签
print(f"SNR=? accuracy: {acc:.2%}")
```

### 注意事项
- 输入数据必须与训练时完全一致：**40帧 × 39维**（MFCC+Δ+ΔΔ）。如果音频长度不足40帧，请先填充或截断至40帧。
- 归一化器 `scaler` 是在训练集的**所有时间步和样本**上拟合的，测试时只能使用 `transform`，不能重新 `fit`。
- 模型在干净验证集上的基线准确率为 **75%**，作为噪声测试的对比基准。

如果需要批量处理多个信噪比，可以循环调用上述代码。如有其他疑问，随时沟通。
