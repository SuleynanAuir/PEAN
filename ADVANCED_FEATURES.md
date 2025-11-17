# PEAN 高级训练功能说明

本文档说明了为PEAN训练流程添加的四项高级功能，提升训练的专业性和可视化效果。

## 功能概览

### 1. 可视化独立文件夹 ✓
**状态**: 已实现

**位置**: `./ckpt/visualizations/`

**说明**: 
- 所有训练可视化图表现在保存在独立的 `visualizations` 文件夹中
- 与checkpoint文件分开存储，便于管理和展示
- 自动创建文件夹结构（如果不存在）

**生成的文件**:
- `training_curves.png` - 训练曲线仪表盘（9个子图）
- `metrics_table.png` - 验证指标汇总表
- `diffusion_process.png` - 扩散模型特征可视化
- `prediction_examples.png` - 预测案例对比图

---

### 2. 自适应学习率调度 ✓
**状态**: 已实现

**策略**: ReduceLROnPlateau

**参数配置**:
```python
scheduler = ReduceLROnPlateau(
    optimizer_G,
    mode='max',           # 监控指标最大化（准确率）
    factor=0.5,           # 学习率衰减因子（减半）
    patience=3,           # 等待3次验证周期
    verbose=True,         # 打印调整信息
    min_lr=1e-6          # 最小学习率下限
)
```

**工作原理**:
1. 每次验证后计算三个测试集（easy, medium, hard）的平均准确率
2. 如果连续3次验证准确率未提升，学习率减半
3. 学习率不会低于 1e-6
4. 自动打印学习率调整信息到控制台

**优势**:
- 自动调整学习率，无需手动干预
- 基于验证集准确率，避免过拟合
- 在训练曲线图中可视化学习率变化（对数坐标）

---

### 3. 扩散模型特征可视化 ✓
**状态**: 已实现

**文件**: `./ckpt/visualizations/diffusion_process.png`

**可视化内容**:
- **上排**: 不同时间步的特征热力图（Feature Heatmap）
  - 纵轴：特征通道（Feature Channels）
  - 横轴：序列位置（Sequence Position）
  - 颜色：激活强度（使用viridis配色方案）

- **下排**: 每个通道的平均激活值柱状图
  - 显示文本先验在扩散过程中的通道响应

**时间步**:
- 展示扩散模型去噪过程的关键阶段
- 默认显示 t=1000, t=500, t=0（可根据采集数量调整）

**技术细节**:
- 从TPEM扩散模型的输出中提取特征
- 每次验证从第一个数据集采集样本
- 自动处理批次维度和张量格式

---

### 4. 正确/错误案例可视化 ✓
**状态**: 已实现

**文件**: `./ckpt/visualizations/prediction_examples.png`

**可视化内容**:
每个案例显示三列对比：
1. **LR Input** - 低分辨率输入图像
2. **SR Output** - 超分辨率输出图像（带预测文本）
   - 正确预测：绿色标题
   - 错误预测：红色标题
3. **HR Ground Truth** - 高分辨率真值图像（带标签文本）

**案例选择**:
- **正确案例**: 展示前5个识别正确的样本 ✓
- **错误案例**: 展示前5个识别错误的样本 ✗

**分组展示**:
- 上半部分：正确预测案例（绿色标记）
- 下半部分：错误预测案例（红色标记）
- 侧边栏文字说明区分两个部分

**采集策略**:
- 仅从第一个验证数据集（easy）采集
- 每次验证最多收集10个正确案例和10个错误案例
- 避免重复存储，减少内存占用

---

## 使用方法

### 启动训练
正常启动训练即可，所有高级功能自动激活：

```powershell
conda activate pean
python main.py --batch_size=8 --mask --rec="aster" --srb=1
```

### 查看可视化结果

训练过程中，每隔320次迭代（`valInterval`）自动生成/更新可视化图表：

```
PEAN/
└── ckpt/
    └── visualizations/
        ├── training_curves.png      # 训练曲线仪表盘
        ├── metrics_table.png        # 指标汇总表
        ├── diffusion_process.png    # 扩散特征（首次验证后）
        └── prediction_examples.png  # 预测案例（首次验证后）
```

### 监控学习率

在训练日志中查看学习率调整信息：
```
Epoch 00000: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
```

在 `training_curves.png` 的右上角图表中可视化学习率变化曲线（对数坐标）。

---

## 技术实现

### 代码结构

**1. 可视化函数** (`utils/visualization.py`)
- `plot_training_curves()` - 9宫格训练曲线
- `plot_comparison_table()` - 指标对比表
- `plot_diffusion_features()` - 扩散特征热力图 ✨ NEW
- `plot_prediction_examples()` - 预测案例对比 ✨ NEW
- `tensor_to_image()` - 张量转图像辅助函数 ✨ NEW

**2. 训练逻辑** (`interfaces/super_resolution.py`)
- 第32-38行: ReduceLROnPlateau 调度器初始化
- 第175-178行: 验证时收集样本和扩散状态
- 第198-208行: 聚合收集的案例和特征
- 第261-270行: 调度器步进
- 第273-290行: 生成所有可视化图表

**3. 评估函数** (`interfaces/super_resolution.py`, line 310)
- 新增参数 `collect_examples=False`
- 验证循环中收集正确/错误案例（最多各10个）
- 收集扩散模型的特征状态
- 返回字典中包含 `examples` 和 `diffusion_states`

### 性能优化

**内存管理**:
- 限制案例收集数量（各10个，展示前5个）
- 及时将张量移至CPU并detach
- 使用 `torch.cuda.empty_cache()` 清理显存

**可视化性能**:
- 使用 `matplotlib.use('Agg')` 后端（无GUI，更快）
- 仅在验证时生成图表（不影响训练速度）
- 异常处理确保可视化失败不中断训练

---

## 学术价值

这些可视化和功能增强使PEAN训练更适合学术研究：

1. **训练监控**: 全面的损失、指标、学习率曲线
2. **性能对比**: 多数据集PSNR/SSIM/准确率横向对比
3. **特征分析**: 扩散模型去噪过程的定量可视化
4. **定性评估**: 成功/失败案例的直观对比
5. **自适应优化**: 基于验证集的智能学习率调度

可直接将生成的图表用于论文、报告、演示等学术用途。

---

## 常见问题

### Q1: 可视化图表不生成？
**A**: 检查以下几点：
- 确保 matplotlib 已安装（版本 3.3.4）
- 查看训练日志是否有 "Failed to generate plots" 警告
- 确认 `./ckpt` 目录有写入权限

### Q2: 扩散特征图为空？
**A**: 扩散特征仅在 `--pre_training=False` 时生成（默认False）。如果使用了预训练模式（`--pre_training`），则会跳过TPEM扩散模型，无法生成扩散特征可视化。

### Q3: 学习率一直不降？
**A**: 这是正常的！ReduceLROnPlateau仅在准确率停止提升时才降低学习率。如果模型持续改进，学习率会保持不变。

### Q4: 案例可视化只有正确（或错误）案例？
**A**: 这取决于验证集的识别效果：
- 如果准确率很高（>90%），错误案例会很少
- 如果准确率很低（<10%），正确案例会很少
- 代码会展示所有可用的案例（最多各5个）

---

## 文件清单

修改/新增的文件：

✅ **新增**:
- `ADVANCED_FEATURES.md` - 本说明文档

✅ **修改**:
- `interfaces/super_resolution.py` 
  - 添加调度器初始化
  - 修改eval函数支持样本收集
  - 添加可视化生成逻辑
  
- `utils/visualization.py`
  - 新增 `plot_diffusion_features()`
  - 新增 `plot_prediction_examples()`
  - 新增 `tensor_to_image()`

---

## 更新日志

**2025-01-XX**: 初始版本
- ✅ 实现可视化独立文件夹
- ✅ 集成ReduceLROnPlateau调度器
- ✅ 添加扩散模型特征可视化
- ✅ 添加正确/错误案例对比
- ✅ 完整文档和使用说明

---

## 致谢

感谢PEAN原作者提供的优秀框架。本增强功能基于原始代码进行扩展，保持了代码的兼容性和可维护性。
