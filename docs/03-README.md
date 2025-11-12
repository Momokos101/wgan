# VCU GAN模糊测试项目 - 快速开始指南

## 📋 项目简介

基于GAN的汽车VCU（整车控制器）唤醒-休眠场景智能模糊测试项目。通过GAN学习传统模糊测试数据库中的数据，生成能够更容易引发异常的CC2电压序列。

## 🚀 环境配置步骤

### macOS 系统配置

#### 步骤1：检查Python版本

项目需要 **Python 3.8-3.12**（TensorFlow不支持Python 3.14）

```bash
# 检查Python版本
python3 --version

# 如果版本不对或没有Python 3.12，安装Python 3.12
brew install python@3.12
```

#### 步骤2：创建虚拟环境

```bash
# 进入项目目录
cd /Users/hby/Desktop/WGANGPProject-master

# 使用Python 3.12创建虚拟环境
python3.12 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 激活后，终端提示符会显示 (venv)
```

**注意**：每次使用项目前都需要激活虚拟环境！

#### 步骤3：安装依赖

```bash
# 确保虚拟环境已激活（终端显示 (venv)）
# 升级pip
pip install --upgrade pip

# 安装项目依赖
pip install tensorflow numpy
```

#### 步骤4：检查环境

```bash
# 运行环境检查脚本
python scripts/utils/check_environment.py
```

**预期输出**：
```
✓ Python 版本: 3.12.x
✓ TensorFlow: 2.x.x
✓ NumPy: x.x.x
✓ 数据库: 正常
✓ 配置文件: 正常
```

#### 步骤5：测试数据加载

```bash
# 测试数据加载和异常检测
python scripts/vcu/test_vcu_data.py
```

**预期输出**：
```
✓ 数据库连接成功
✓ 数据加载成功
✓ 异常检测正常
...
```

---

### Windows 系统配置

#### 步骤1：检查Python版本

项目需要 **Python 3.8-3.12**（TensorFlow不支持Python 3.13+）

**方法1：检查是否已安装Python**

```cmd
# 打开命令提示符（CMD）或PowerShell
python --version
# 或
py --version
```

**方法2：如果没有Python或版本不对，安装Python 3.12**

1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载 Python 3.12.x 安装包（Windows installer）
3. 运行安装程序，**重要**：勾选 "Add Python to PATH"
4. 完成安装

**验证安装**：
```cmd
python --version
# 应该显示 Python 3.12.x
```

#### 步骤2：创建虚拟环境

```cmd
# 打开命令提示符（CMD）或PowerShell
# 进入项目目录（根据你的实际路径修改）
cd C:\Users\YourName\Desktop\WGANGPProject-master

# 使用Python创建虚拟环境
python -m venv venv
# 或如果python命令不可用，使用：
py -3.12 -m venv venv

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
✓ Python 版本: 3.12.x
✓ TensorFlow: 2.x.x
✓ NumPy: x.x.x
✓ 数据库: 正常
✓ 配置文件: 正常
```

#### 步骤5：测试数据加载

```cmd
# 测试数据加载和异常检测
python scripts\vcu\test_vcu_data.py
```

**预期输出**：
```
✓ 数据库连接成功
✓ 数据加载成功
✓ 异常检测正常
...
```

---

### 常见问题（Windows）

#### 问题1：`python` 命令不可用

**解决**：
- 使用 `py` 命令：`py -3.12 -m venv venv`
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

### 方式1：使用运行脚本（推荐）

**macOS/Linux**：
```bash
# 激活虚拟环境
source venv/bin/activate

# 测试数据加载
./scripts/utils/run.sh test

# 训练模型
./scripts/utils/run.sh train

# 生成测试序列
./scripts/utils/run.sh generate

# 检查环境
./scripts/utils/run.sh check
```

**Windows**：
```cmd
# 激活虚拟环境
venv\Scripts\activate

# 注意：Windows下run.sh脚本可能不可用，建议直接使用方式2
```

### 方式2：直接运行Python脚本（推荐Windows使用）

**macOS/Linux**：
```bash
# 激活虚拟环境
source venv/bin/activate

# 训练模型
python scripts/vcu/train_vcu.py

# 生成测试序列
python scripts/vcu/generate_vcu.py

# 测试数据加载
python scripts/vcu/test_vcu_data.py
```

**Windows**：
```cmd
# 激活虚拟环境
venv\Scripts\activate

# 训练模型
python scripts\vcu\train_vcu.py
# 或
python scripts/vcu/train_vcu.py

# 生成测试序列
python scripts\vcu\generate_vcu.py

# 测试数据加载
python scripts\vcu\test_vcu_data.py
```

---

## 🔧 配置说明

### 配置文件位置

配置文件已拆分为多个文件，每个人负责自己的配置：

- `configs/config_vcu_data.py` - 数据配置（后端负责人）
- `configs/config_vcu_model.py` - 模型配置（GAN架构负责人）
- `configs/config_vcu_train.py` - 训练配置（GAN应用负责人）
- `configs/config_vcu_base.py` - 基础配置（共享，修改前需讨论）

### 常用配置项

**数据配置**（`config_vcu_data.py`）：
```python
DB_PATH = 'database/db.db'          # 数据库路径
CONTEXT_BEFORE = 4                  # 向前提取的唤醒电压数量
CONTEXT_AFTER = 4                   # 向后提取的唤醒电压数量
```

**训练配置**（`config_vcu_train.py`）：
```python
MAX_EPOCH = 80                      # 训练轮数
BATCH_SIZE = 32                     # 批次大小
STAGEI_G_LR = 1e-4                 # 生成器学习率
STAGEI_D_LR = 1e-4                 # 判别器学习率
```

**模型配置**（`config_vcu_model.py`）：
```python
G_DIM = 32                          # 生成器维度
D_DIM = 4                           # 判别器维度
Z_DIM = 128                         # 噪声向量维度
```

---

## 📁 重要目录说明

- **`database/`** - 数据库文件（`db.db`）
- **`data/vcu/`** - 处理后的数据和生成结果
- **`model_weights/vcu/`** - 训练好的模型权重
- **`logs/vcu/`** - 训练日志
- **`sequence/`** - 数据处理模块（后端负责）
- **`nn/`** - 模型实现（GAN架构负责）
- **`scripts/vcu/`** - 训练和生成脚本（GAN应用负责）

详细说明请查看：`docs/02-项目架构说明.md`

---

## 🎯 工作流程

### 1. 数据预处理（自动）

首次运行训练脚本时，会自动：
- 从数据库读取数据
- 检测异常数据点
- 提取异常点上下文
- 提取电压特征
- 构建条件向量
- 保存到 `data/vcu/` 目录

### 2. 模型训练

```bash
python scripts/vcu/train_vcu.py
```

训练过程：
- 加载预处理的数据
- 构建WGAN-GP模型
- 交替训练生成器和判别器
- 保存模型权重到 `model_weights/vcu/`

### 3. 生成测试序列

```bash
python scripts/vcu/generate_vcu.py
```

生成过程：
- 加载训练好的模型
- 根据异常类型和电压特征生成序列
- 对序列进行评分
- 输出十六进制格式到 `data/vcu/generated_sequences_*.txt`

---

## ⚠️ 常见问题

### 问题1：`ModuleNotFoundError: No module named 'tensorflow'`

**原因**：未安装TensorFlow或虚拟环境未激活

**解决（macOS）**：
```bash
# 激活虚拟环境
source venv/bin/activate

# 安装TensorFlow
pip install tensorflow
```

**解决（Windows）**：
```cmd
# 激活虚拟环境
venv\Scripts\activate

# 安装TensorFlow
pip install tensorflow
```

### 问题2：`externally-managed-environment` 错误（macOS）

**原因**：macOS不允许直接使用pip安装到系统Python

**解决**：使用虚拟环境（见macOS配置步骤2）

### 问题3：Python版本不兼容

**原因**：Python版本过高（如3.13+），TensorFlow不支持

**解决（macOS）**：
```bash
# 安装Python 3.12
brew install python@3.12
```

**解决（Windows）**：
- 从[Python官网](https://www.python.org/downloads/)下载Python 3.12.x
- 安装时勾选 "Add Python to PATH"

### 问题4：数据库文件不存在

**原因**：`database/db.db` 文件不存在

**解决**：确保数据库文件在正确位置，或修改 `configs/config_vcu_data.py` 中的 `DB_PATH`

### 问题5：训练时出现NaN

**原因**：学习率过高或数据有问题

**解决**：
- 降低学习率（修改 `configs/config_vcu_train.py` 中的 `STAGEI_D_LR`）
- 检查数据是否正常（运行 `python scripts/vcu/test_vcu_data.py`）

---

## 📚 相关文档

- **`docs/01-成员分工.md`** - 成员分工说明和工作顺序
- **`docs/02-项目架构说明.md`** - 项目架构和文件说明

---

## ✅ 快速检查清单

开始工作前，确保：

**macOS**：
- [ ] Python版本正确（3.8-3.12）
- [ ] 虚拟环境已创建并激活（`source venv/bin/activate`）
- [ ] 依赖已安装（TensorFlow, NumPy）
- [ ] 环境检查通过（`python scripts/utils/check_environment.py`）
- [ ] 数据库文件存在（`database/db.db`）
- [ ] 数据加载测试通过（`python scripts/vcu/test_vcu_data.py`）

**Windows**：
- [ ] Python版本正确（3.8-3.12）
- [ ] Python已添加到PATH
- [ ] 虚拟环境已创建并激活（`venv\Scripts\activate`）
- [ ] 依赖已安装（TensorFlow, NumPy）
- [ ] 环境检查通过（`python scripts\utils\check_environment.py`）
- [ ] 数据库文件存在（`database\db.db`）
- [ ] 数据加载测试通过（`python scripts\vcu\test_vcu_data.py`）

---

## 🆘 需要帮助？

1. **查看文档**：`docs/` 目录下的文档
2. **运行测试**：`python scripts/vcu/test_vcu_data.py`
3. **检查环境**：`python scripts/utils/check_environment.py`
4. **联系团队**：在团队群中提问

---

**祝开发顺利！** 🎉

