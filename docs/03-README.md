# VCU GAN模糊测试项目 - 快速开始指南

## 📋 项目简介

基于GAN的汽车VCU（整车控制器）唤醒-休眠场景智能模糊测试项目。通过GAN学习传统模糊测试数据库中的数据，生成能够更容易引发异常的CC2电压序列。

## 🚀 环境配置步骤

### macOS 系统配置（Apple Silicon - M1/M2/M3/M4）

#### 步骤1：检查Python版本

项目需要 **Python 3.10-3.11**（TensorFlow Metal仅支持Python 3.10-3.11）

```bash
# 检查Python版本
python3 --version

# 如果版本不对，使用conda安装Python 3.11
# 安装Anaconda/Miniconda（如果尚未安装）
# 从 https://docs.conda.io/en/latest/miniconda.html 下载安装
```

#### 步骤2：创建conda虚拟环境

**⚠️ 重要：macOS必须使用conda管理环境，不能使用venv！**

```bash
# 创建Python 3.11的conda环境
conda create -n tf_m4 python=3.11

# 激活环境
conda activate tf_m4

# 激活后，终端提示符会显示 (tf_m4)
```

**注意**：每次使用项目前都需要激活conda环境！

#### 步骤3：安装TensorFlow Metal依赖

**Apple Silicon (M1/M2/M3/M4) 专用配置**：

```bash
# 确保conda环境已激活（终端显示 (tf_m4)）
# 安装TensorFlow和Metal支持
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal

# 安装其他依赖
pip install numpy
```

**注意**：
- `tensorflow-macos` 是针对Apple Silicon优化的TensorFlow版本
- `tensorflow-metal` 提供GPU加速支持
- 不要使用标准的 `tensorflow` 包

#### 步骤4：检查环境

```bash
# 进入项目目录
cd /Users/linqi/Desktop/wgan

# 运行环境检查脚本
python scripts/utils/check_environment.py
```

**预期输出**：
```
✓ Python 版本: 3.11.x
✓ TensorFlow: 2.x.x (Metal support enabled)
✓ NumPy: x.x.x
✓ Metal GPU: Available
✓ 数据库: 正常
✓ 配置文件: 正常
```

#### 步骤5：验证GAN架构

```bash
# 测试GAN架构是否正常工作
python scripts/vcu/test_stage2_completion.py
```

**预期输出**：
```
✅ 测试 1/9: 检查核心文件存在性 - 通过
✅ 测试 2/9: 检查配置文件 - 通过
...
🎉 恭喜！所有测试通过 (9/9)
```

#### macOS特有问题

**问题1：Metal性能警告**

某些操作可能触发Metal后端限制（如复杂梯度计算），这是正常的硬件限制，不影响模型使用。

**问题2：`externally-managed-environment` 错误**

**原因**：试图使用pip安装到系统Python

**解决**：必须使用conda环境（见步骤2）

---

### Windows 系统配置

#### 步骤1：检查Python版本

项目需要 **Python 3.10-3.11**（推荐3.11）

**方法1：检查是否已安装Python**

```cmd
# 打开命令提示符（CMD）或PowerShell
python --version
# 或
py --version
```

**方法2：如果没有Python或版本不对，安装Python 3.11**

1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载 Python 3.11.x 安装包（Windows installer）
3. 运行安装程序，**重要**：勾选 "Add Python to PATH"
4. 完成安装

**验证安装**：
```cmd
python --version
# 应该显示 Python 3.11.x
```

#### 步骤2：创建虚拟环境

```cmd
# 打开命令提示符（CMD）或PowerShell
# 进入项目目录（根据你的实际路径修改）
cd C:\Users\YourName\Desktop\wgan

# 使用Python创建虚拟环境
python -m venv venv
# 或如果python命令不可用，使用：
py -3.11 -m venv venv

# 激活虚拟环境
# 在CMD中：
venv\Scripts\activate.bat
# 在PowerShell中：
venv\Scripts\Activate.ps1

# 如果PowerShell执行策略限制，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 激活后，命令提示符会显示 (venv)
```

**注意**：每次使用项目前都需要激活虚拟环境！

#### 步骤3：安装依赖

```cmd
# 确保虚拟环境已激活（命令提示符显示 (venv)）
# 升级pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install tensorflow numpy
```

#### 步骤4：检查环境

```cmd
# 运行环境检查脚本
python scripts\utils\check_environment.py
```

**预期输出**：
```
✓ Python 版本: 3.11.x
✓ TensorFlow: 2.x.x
✓ NumPy: x.x.x
✓ 数据库: 正常
✓ 配置文件: 正常
```

#### 步骤5：验证GAN架构

```cmd
# 测试GAN架构是否正常工作
python scripts\vcu\test_stage2_completion.py
```

**预期输出**：
```
✅ 测试 1/9: 检查核心文件存在性 - 通过
✅ 测试 2/9: 检查配置文件 - 通过
...
🎉 恭喜！所有测试通过 (9/9)
```

---

### 常见问题（Windows）

#### 问题1：`python` 命令不可用

**解决**：
- 使用 `py` 命令：`py -3.11 -m venv venv`
- 或重新安装Python，确保勾选 "Add Python to PATH"

#### 问题2：PowerShell执行策略限制

**错误信息**：`无法加载文件，因为在此系统上禁止运行脚本`

**解决**：
```powershell
# 以管理员身份运行PowerShell，执行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 或使用CMD代替PowerShell
```

#### 问题3：路径分隔符问题

**注意**：Windows使用反斜杠 `\`，但Python代码中可以使用正斜杠 `/`

```cmd
# Windows路径
python scripts\vcu\train_vcu.py

# 或使用正斜杠（Python支持）
python scripts/vcu/train_vcu.py
```

---

## 📖 使用指南

### 当前项目状态

**✅ 第2阶段（GAN架构开发）已完成**

已实现的功能：
- Conv1D生成器和判别器架构
- WGAN-GP训练框架
- 条件生成接口（9维条件向量）
- 完整的模型API

**🚧 第3阶段（训练和生成应用）待开发**

待实现的功能：
- 实际训练脚本（`train_vcu.py`）
- 序列生成脚本（`generate_vcu.py`）
- 模型评估和可视化

### 验证GAN架构

```bash
# macOS
conda activate tf_m4
python scripts/vcu/test_stage2_completion.py

# Windows
venv\Scripts\activate
python scripts\vcu\test_stage2_completion.py
```

### 方式1：使用运行脚本（暂不可用）

**注意**：训练和生成脚本属于第3阶段,尚未实现。

**macOS/Linux**：
```bash
# 激活conda环境
conda activate tf_m4

# 测试GAN架构（可用）
python scripts/vcu/test_stage2_completion.py

# 训练模型（第3阶段待实现）
# ./scripts/utils/run.sh train

# 生成测试序列（第3阶段待实现）
# ./scripts/utils/run.sh generate
```

**Windows**：
```cmd
# 激活虚拟环境
venv\Scripts\activate

# 测试GAN架构（可用）
python scripts\vcu\test_stage2_completion.py

# 训练和生成功能待第3阶段实现
```

### 方式2：直接运行Python脚本

**当前可用的脚本**：

**macOS/Linux**：
```bash
# 激活conda环境
conda activate tf_m4

# 验证GAN架构完整性（推荐首先运行）
python scripts/vcu/test_stage2_completion.py

# 测试数据加载（如果有数据库）
python scripts/vcu/test_vcu_data.py
```

**Windows**：
```cmd
# 激活虚拟环境
venv\Scripts\activate

# 验证GAN架构完整性（推荐首先运行）
python scripts\vcu\test_stage2_completion.py

# 测试数据加载（如果有数据库）
python scripts\vcu\test_vcu_data.py
```

**第3阶段脚本（待实现）**：
- `scripts/vcu/train_vcu.py` - 训练模型
- `scripts/vcu/generate_vcu.py` - 生成测试序列

---

## 🔧 配置说明

### 配置文件位置

配置文件已拆分为多个模块，职责明确：

- `configs/config_vcu_base.py` - 基础配置（FLAG、数据路径等）
- `configs/config_vcu_data.py` - 数据配置（上下文窗口、异常检测等）
- `configs/config_vcu_model.py` - 模型配置（网络维度、架构参数）
- `configs/config_vcu_train.py` - 训练配置（学习率、批次大小、训练轮数）
- `configs/config_vcu.py` - 统一导入接口

### 关键配置项

**基础配置**（`config_vcu_base.py`）：
```python
FLAG = 'LWCO'                       # 条件模式（LWCO=带条件）
MODEL_TYPE = 'conv1d'               # 模型类型（Conv1D架构）
DATA_DIR = 'data/vcu'              # 数据目录
MODEL_WEIGHTS_DIR = 'model_weights/vcu'  # 模型权重目录
```

**模型配置**（`config_vcu_model.py`）：
```python
# 生成器和判别器维度
G_DIM = 32                          # 生成器基础维度
D_DIM = 4                           # 判别器基础维度

# 条件和噪声维度
C_DIM = 9                           # 条件向量维度
Z_DIM = 128                         # 噪声向量维度
EMBEDDING_DIM = 128                 # 条件嵌入维度

# 优化器配置
OPT_TYPE = 'adam'                   # 优化器类型
```

**数据配置**（`config_vcu_data.py`）：
```python
DB_PATH = 'database/db.db'          # 数据库路径
CONTEXT_BEFORE = 4                  # 异常点前的上下文长度
CONTEXT_AFTER = 4                   # 异常点后的上下文长度
```

**训练配置**（`config_vcu_train.py`）：
```python
# 训练参数
MAX_EPOCH = 80                      # 训练轮数
BATCH_SIZE = 32                     # 批次大小

# 学习率
STAGEI_G_LR = 1e-4                 # 生成器学习率
STAGEI_D_LR = 1e-4                 # 判别器学习率

# WGAN-GP参数
LAMBDA_GP = 10                      # 梯度惩罚系数
N_CRITIC = 5                        # 判别器更新次数
```

### 修改配置的建议

- **基础配置**：修改前需团队讨论（影响全局）
- **模型配置**：GAN架构负责人修改
- **训练配置**：GAN应用负责人修改
- **数据配置**：后端负责人修改

---

## 📁 重要目录说明

### 核心目录

- **`configs/`** - 配置文件（所有配置集中管理）
  - `config_vcu_base.py` - 基础配置
  - `config_vcu_data.py` - 数据配置
  - `config_vcu_model.py` - 模型配置
  - `config_vcu_train.py` - 训练配置
  - `config_vcu.py` - 统一导入

- **`nn/`** - 神经网络模块（第2阶段已完成）
  - `model.py` - 基础模型类（NetModel）
  - `conv1d.py` - Conv1D GAN架构实现
  - `scale_model.py` - 模型封装和管理
  - `trainer.py` - 训练器接口
  - `wd_trainer.py` - WGAN-GP训练器实现
  - `optimizer.py` - 优化器配置

- **`sequence/`** - 数据处理模块
  - `db_loader.py` - 数据库加载器
  - `vcu_data_process.py` - VCU数据处理

- **`scripts/vcu/`** - VCU相关脚本
  - `test_stage2_completion.py` - 第2阶段验证脚本（可用）
  - `test_vcu_data.py` - 数据加载测试
  - `train_vcu.py` - 训练脚本（第3阶段待实现）
  - `generate_vcu.py` - 生成脚本（第3阶段待实现）

### 数据目录

- **`database/`** - 数据库文件
- **`data/vcu/`** - 处理后的训练数据和生成结果
  - `train_*.npy` - 训练数据
  - `test_*.npy` - 测试数据
  - `metadata.json` - 元数据信息
  - `generated_sequences_*.txt` - 生成的序列

- **`model_weights/vcu/`** - 训练好的模型权重
  - `{timestamp}_conv1d_LWCO_generator.weights.h5`
  - `{timestamp}_conv1d_LWCO_discriminator.weights.h5`

### 文档目录

- **`docs/`** - 项目文档
  - `01-成员分工.md` - 成员分工和进度
  - `02-项目架构说明.md` - 详细架构说明
  - `03-README.md` - 本文件（快速开始指南）
  - `03-GAN架构开发完成报告.md` - 第2阶段完成报告
  - `04-第3阶段开发指南.md` - 第3阶段开发指南
  - `05-GAN架构开发总结.md` - 架构总结
  - `06-GAN架构测试指南.md` - 测试指南
  - `07-第2阶段与第3阶段关系说明.md` - 阶段关系

---

## 🎯 工作流程

### 当前阶段：第2阶段已完成 ✅

**已完成的工作**：
- ✅ GAN架构设计和实现
- ✅ Conv1D生成器和判别器
- ✅ WGAN-GP训练框架
- ✅ 条件生成接口（9维条件向量）
- ✅ 模型API和配置系统
- ✅ 完整的测试和文档

### 下一阶段：第3阶段待开发 🚧

**第3阶段任务**（由GAN应用负责人实现）：

1. **数据预处理流程**
   - 从数据库读取VCU测试数据
   - 检测异常数据点
   - 提取异常点上下文
   - 构建条件向量
   - 保存预处理数据

2. **模型训练流程**
   - 加载预处理的数据
   - 使用ScaleModel构建模型
   - 使用WDTrainer训练WGAN-GP
   - 保存模型权重

3. **序列生成流程**
   - 加载训练好的模型
   - 根据异常类型和特征生成序列
   - 对生成序列进行评分
   - 输出可用于测试的格式

**参考文档**：
- `docs/04-第3阶段开发指南.md` - 详细的实现指南
- `docs/07-第2阶段与第3阶段关系说明.md` - 接口说明

### 使用已完成的架构

**示例：如何使用GAN架构**

```python
from nn.scale_model import ScaleModel
from configs.config_vcu import *

# 1. 创建模型
model = ScaleModel(
    seq_len=100,        # 序列长度
    c_dim=C_DIM,        # 条件维度=9
    z_dim=Z_DIM,        # 噪声维度=128
    g_dim=G_DIM,        # 生成器维度=32
    d_dim=D_DIM         # 判别器维度=4
)

# 2. 构建模型
model.build()

# 3. 生成序列
import numpy as np
condition = np.random.randn(4, 9)  # 4个样本，9维条件
generated = model.model.generate_wc(condition)
print(generated.shape)  # (4, 100)

# 4. 判别序列
sequences = np.random.randn(4, 100)
logits = model.model.discriminate_wc([condition, sequences])
print(logits.shape)  # (4, 1)
```

---

## ⚠️ 常见问题

### macOS专有问题

#### 问题1：`externally-managed-environment` 错误

**原因**：尝试使用pip安装到系统Python

**解决**：
```bash
# 必须使用conda环境
conda create -n tf_m4 python=3.11
conda activate tf_m4
```

#### 问题2：Metal GPU相关警告

**现象**：看到Metal性能警告或某些操作失败

**原因**：Apple Silicon的Metal后端在某些复杂操作上有限制

**影响**：不影响模型的正常使用（构建、生成、判别）

**解决**：这是正常的硬件限制，可以忽略

#### 问题3：TensorFlow版本不兼容

**错误**：`No module named 'tensorflow'` 或版本错误

**解决**：
```bash
conda activate tf_m4
# 卸载标准TensorFlow
pip uninstall tensorflow

# 安装Apple Silicon专用版本
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

### 通用问题

#### 问题1：`ModuleNotFoundError: No module named 'tensorflow'`

**原因**：未安装TensorFlow或环境未激活

**解决（macOS）**：
```bash
# 激活conda环境
conda activate tf_m4

# 安装TensorFlow Metal
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal
```

**解决（Windows）**：
```cmd
# 激活虚拟环境
venv\Scripts\activate

# 安装TensorFlow
pip install tensorflow
```

#### 问题2：Python版本不兼容

**原因**：Python版本不在支持范围（需要3.10-3.11）

**解决（macOS）**：
```bash
# 创建正确版本的conda环境
conda create -n tf_m4 python=3.11
conda activate tf_m4
```

**解决（Windows）**：
- 从[Python官网](https://www.python.org/downloads/)下载Python 3.11.x
- 安装时勾选 "Add Python to PATH"

#### 问题3：找不到配置文件

**错误**：`ModuleNotFoundError: No module named 'configs'`

**原因**：未在项目根目录运行脚本

**解决**：
```bash
# 确保在项目根目录
cd /path/to/wgan

# 然后运行脚本
python scripts/vcu/test_stage2_completion.py
```

#### 问题4：测试失败

**现象**：`test_stage2_completion.py` 未通过所有测试

**排查步骤**：
1. 检查Python版本：`python --version`（应为3.10-3.11）
2. 检查TensorFlow安装：`python -c "import tensorflow; print(tensorflow.__version__)"`
3. 检查当前目录：`pwd`（应在项目根目录）
4. 查看详细错误信息，根据提示修复

---

## 📚 相关文档

### 必读文档

- **`docs/01-成员分工.md`** - 成员分工说明和各阶段进度
- **`docs/02-项目架构说明.md`** - 详细的项目架构和文件说明
- **`docs/03-README.md`** - 本文件（快速开始指南）

### 第2阶段（已完成）文档

- **`docs/03-GAN架构开发完成报告.md`** - 第2阶段完成报告，包含架构详细说明
- **`docs/05-GAN架构开发总结.md`** - 架构开发总结和设计决策
- **`docs/06-GAN架构测试指南.md`** - 如何测试和验证GAN架构
- **`docs/07-第2阶段与第3阶段关系说明.md`** - 解释两个阶段的职责分工

### 第3阶段（待开发）文档

- **`docs/04-第3阶段开发指南.md`** - 完整的第3阶段实现指南
  - 数据预处理实现
  - 训练脚本实现
  - 生成脚本实现
  - 包含完整代码模板

### 文档阅读建议

**如果你是新成员**：
1. 先读 `01-成员分工.md` 了解项目现状
2. 再读 `02-项目架构说明.md` 理解整体结构
3. 最后读本文件配置环境

**如果你要使用GAN架构**：
1. 读 `03-GAN架构开发完成报告.md` 了解架构API
2. 读 `07-第2阶段与第3阶段关系说明.md` 了解接口
3. 读 `04-第3阶段开发指南.md` 学习如何使用

**如果你要测试验证**：
1. 读 `06-GAN架构测试指南.md` 了解测试方法
2. 运行 `test_stage2_completion.py` 验证环境

---

## ✅ 快速检查清单

### macOS (Apple Silicon)

开始工作前，确保：
- [ ] 已安装Anaconda/Miniconda
- [ ] Python版本正确（3.10或3.11）
- [ ] conda环境已创建（`conda create -n tf_m4 python=3.11`）
- [ ] conda环境已激活（`conda activate tf_m4`，终端显示`(tf_m4)`）
- [ ] TensorFlow Metal已安装（`tensorflow-macos`和`tensorflow-metal`）
- [ ] 依赖已安装（NumPy）
- [ ] 在项目根目录（`cd /Users/linqi/Desktop/wgan`）
- [ ] GAN架构验证通过（`python scripts/vcu/test_stage2_completion.py` 显示9/9通过）

### Windows

开始工作前，确保：
- [ ] Python版本正确（3.10或3.11）
- [ ] Python已添加到PATH
- [ ] 虚拟环境已创建（`python -m venv venv`）
- [ ] 虚拟环境已激活（`venv\Scripts\activate`，终端显示`(venv)`）
- [ ] TensorFlow已安装（`pip install tensorflow`）
- [ ] 依赖已安装（NumPy）
- [ ] 在项目根目录（`cd C:\path\to\wgan`）
- [ ] GAN架构验证通过（`python scripts\vcu\test_stage2_completion.py` 显示9/9通过）

### 环境验证命令

**macOS**：
```bash
# 1. 检查Python版本
python --version  # 应显示 3.10.x 或 3.11.x

# 2. 检查conda环境
conda info --envs  # 应看到 tf_m4

# 3. 激活环境
conda activate tf_m4

# 4. 检查TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# 5. 验证GAN架构
cd /Users/linqi/Desktop/wgan
python scripts/vcu/test_stage2_completion.py
# 应显示：🎉 恭喜！所有测试通过 (9/9)
```

**Windows**：
```cmd
# 1. 检查Python版本
python --version  # 应显示 3.10.x 或 3.11.x

# 2. 激活环境
venv\Scripts\activate

# 3. 检查TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# 4. 验证GAN架构
cd C:\path\to\wgan
python scripts\vcu\test_stage2_completion.py
# 应显示：恭喜！所有测试通过 (9/9)
```

---

## 🆘 需要帮助？

1. **查看文档**：`docs/` 目录下的文档
2. **运行测试**：`python scripts/vcu/test_vcu_data.py`
3. **检查环境**：`python scripts/utils/check_environment.py`
4. **联系团队**：在团队群中提问

---

**祝开发顺利！** 🎉

