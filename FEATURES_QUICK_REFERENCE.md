# PEAN 高级功能快速参考

## 四大新增功能

### 1️⃣ 独立可视化文件夹
📂 **位置**: `./ckpt/visualizations/`
- ✅ 自动创建
- ✅ 与checkpoint分离
- ✅ 包含所有可视化图表

### 2️⃣ 自适应学习率
🎯 **策略**: ReduceLROnPlateau
```python
参数:
- factor=0.5       # 每次减半
- patience=3       # 等待3次验证
- min_lr=1e-6     # 最小学习率
```
📊 **监控**: 在 training_curves.png 右上角查看LR曲线

### 3️⃣ 扩散模型特征可视化  
🔥 **文件**: `diffusion_process.png`
- 📈 特征热力图（上排）
- 📊 通道激活柱状图（下排）
- ⏱️ 显示去噪过程关键时间步

### 4️⃣ 正确/错误案例对比
✅❌ **文件**: `prediction_examples.png`
- ✅ 前5个正确预测（绿色）
- ❌ 前5个错误预测（红色）
- 📸 三列展示: LR → SR → HR

---

## 快速启动

```powershell
# 激活环境
conda activate pean

# 正常训练（所有功能自动启用）
python main.py --batch_size=8 --mask --rec="aster" --srb=1
```

---

## 可视化文件说明

| 文件名 | 内容 | 更新频率 |
|--------|------|----------|
| `training_curves.png` | 9宫格训练曲线仪表盘 | 每320次迭代 |
| `metrics_table.png` | 验证指标汇总表 | 每320次迭代 |
| `diffusion_process.png` | 扩散特征可视化 | 首次验证后 |
| `prediction_examples.png` | 预测案例对比 | 每次验证 |

---

## 训练曲线仪表盘详解

```
┌─────────────┬─────────────┬─────────────┐
│ Loss+MA曲线 │ Loss+MA曲线 │  学习率曲线 │
├─────────────┼─────────────┼─────────────┤
│  PSNR曲线   │  SSIM曲线   │  准确率曲线 │
├─────────────┼─────────────┼─────────────┤
│ PSNR柱状图  │ SSIM柱状图  │ 准确率柱状图│
└─────────────┴─────────────┴─────────────┘
```
每个指标都包含三个测试集: Easy, Medium, Hard

---

## 学习率调度示例

```
初始: lr = 1e-3
┌─────────────────────────────────────┐
│ 验证1: acc=80% → lr保持              │
│ 验证2: acc=82% → lr保持              │
│ 验证3: acc=83% → lr保持              │
│ 验证4: acc=83% → lr保持 (patience 1)│
│ 验证5: acc=83% → lr保持 (patience 2)│
│ 验证6: acc=83% → lr保持 (patience 3)│
│ 验证7: acc=83% → lr=5e-4 (降低!)    │
└─────────────────────────────────────┘
```

---

## 技术特性

✅ **自动化**
- 无需手动配置，开箱即用
- 异常容错，可视化失败不影响训练

✅ **高效**
- 仅验证时生成图表
- 智能采样，避免内存溢出
- CPU渲染，不占用GPU

✅ **学术级**
- 300 DPI高清输出
- 专业配色方案
- 适合论文/报告使用

---

## 常见日志

✅ **正常输出**:
```
Training curves saved to: ./ckpt/visualizations/training_curves.png
Metrics table saved to: ./ckpt/visualizations/metrics_table.png
Diffusion features saved to: ./ckpt/visualizations/diffusion_process.png
Prediction examples saved to: ./ckpt/visualizations/prediction_examples.png
```

⚠️ **警告（可忽略）**:
```
Warning: Failed to generate plots: xxx
```
说明: 可视化失败不影响训练，可能是matplotlib版本问题

📉 **学习率降低**:
```
Epoch 00000: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
```

---

## 文件结构

```
PEAN/
├── ckpt/
│   ├── visualizations/          # ✨ 新增独立文件夹
│   │   ├── training_curves.png
│   │   ├── metrics_table.png
│   │   ├── diffusion_process.png
│   │   └── prediction_examples.png
│   ├── PEAN_final.pth
│   └── TPEM_final.pth
├── utils/
│   └── visualization.py         # ✨ 新增可视化函数
├── interfaces/
│   └── super_resolution.py      # ✨ 增强训练逻辑
└── ADVANCED_FEATURES.md         # 详细文档
```

---

## 下一步

1. ✅ 启动训练
2. ✅ 等待第一次验证（320次迭代）
3. ✅ 检查 `./ckpt/visualizations/` 文件夹
4. ✅ 查看生成的可视化图表
5. ✅ 监控学习率调整（如有）

---

**完整文档**: 查看 `ADVANCED_FEATURES.md`
**问题反馈**: 检查训练日志和可视化输出
