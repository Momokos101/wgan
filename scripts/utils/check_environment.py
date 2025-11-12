"""
环境检查脚本
检查运行所需的所有依赖和配置
"""
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("1. 检查 Python 版本")
    print("=" * 60)
    version = sys.version_info
    print(f"  Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ⚠️  警告: 建议使用 Python 3.7 或更高版本")
        return False
    print("  ✓ Python 版本符合要求")
    return True

def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("2. 检查依赖包")
    print("=" * 60)
    
    dependencies = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'sqlite3': 'SQLite3 (内置)'
    }
    
    missing = []
    installed = []
    
    for module, name in dependencies.items():
        try:
            if module == 'tensorflow':
                import tensorflow as tf
                print(f"  ✓ {name}: {tf.__version__}")
                installed.append(name)
            elif module == 'numpy':
                import numpy as np
                print(f"  ✓ {name}: {np.__version__}")
                installed.append(name)
            elif module == 'sqlite3':
                import sqlite3
                print(f"  ✓ {name}: {sqlite3.sqlite_version}")
                installed.append(name)
        except ImportError:
            print(f"  ✗ {name}: 未安装")
            missing.append(name)
    
    return missing, installed

def check_database():
    """检查数据库文件"""
    print("\n" + "=" * 60)
    print("3. 检查数据库文件")
    print("=" * 60)
    
    from configs.config_vcu import DB_PATH
    
    if os.path.exists(DB_PATH):
        size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
        print(f"  ✓ 数据库文件存在: {DB_PATH}")
        print(f"    文件大小: {size:.2f} MB")
        return True
    else:
        print(f"  ✗ 数据库文件不存在: {DB_PATH}")
        print(f"    请检查 config_vcu.py 中的 DB_PATH 配置")
        return False

def check_directories():
    """检查输出目录"""
    print("\n" + "=" * 60)
    print("4. 检查输出目录")
    print("=" * 60)
    
    from configs.config_vcu import OUTPUT_DIR, LOG_DIR, MODEL_PATH
    
    directories = {
        '数据输出目录': OUTPUT_DIR,
        '日志目录': LOG_DIR,
        '模型目录': MODEL_PATH
    }
    
    for name, path in directories.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path} (已存在)")
        else:
            print(f"  - {name}: {path} (将自动创建)")
    
    return True

def check_config():
    """检查配置文件"""
    print("\n" + "=" * 60)
    print("5. 检查配置文件")
    print("=" * 60)
    
    try:
        from configs.config_vcu import (
            DB_PATH, BATCH_SIZE, MAX_EPOCH, 
            C_DIM, Z_DIM, FLAG, PROGRAM
        )
        
        print(f"  ✓ config_vcu.py 加载成功")
        print(f"    数据库路径: {DB_PATH}")
        print(f"    批次大小: {BATCH_SIZE}")
        print(f"    最大轮数: {MAX_EPOCH}")
        print(f"    条件维度: {C_DIM}")
        print(f"    噪声维度: {Z_DIM}")
        print(f"    模式: {FLAG}")
        print(f"    程序: {PROGRAM}")
        return True
    except Exception as e:
        print(f"  ✗ 配置文件加载失败: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("VCU GAN 模糊测试环境检查")
    print("=" * 60)
    
    results = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'database': check_database(),
        'directories': check_directories(),
        'config': check_config()
    }
    
    missing_deps, installed_deps = results['dependencies']
    
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    
    all_ok = True
    
    if results['python']:
        print("  ✓ Python 版本: 正常")
    else:
        print("  ✗ Python 版本: 需要更新")
        all_ok = False
    
    if not missing_deps:
        print("  ✓ 依赖包: 全部已安装")
    else:
        print(f"  ✗ 依赖包: 缺少 {', '.join(missing_deps)}")
        all_ok = False
    
    if results['database']:
        print("  ✓ 数据库文件: 存在")
    else:
        print("  ✗ 数据库文件: 不存在")
        all_ok = False
    
    if results['config']:
        print("  ✓ 配置文件: 正常")
    else:
        print("  ✗ 配置文件: 异常")
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ 环境检查通过！可以开始运行训练和生成")
        print("=" * 60)
        print("\n下一步:")
        print("  1. python test_vcu_data.py  # 测试数据加载")
        print("  2. python train_vcu.py      # 训练模型")
        print("  3. python generate_vcu.py   # 生成序列")
    else:
        print("✗ 环境检查未通过，请先解决上述问题")
        print("=" * 60)
        if missing_deps:
            print("\n安装缺失的依赖:")
            if 'TensorFlow' in missing_deps:
                print("  pip install tensorflow")
                print("  或")
                print("  conda install tensorflow")
    
    return all_ok

if __name__ == '__main__':
    main()

