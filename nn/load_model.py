import os
import sys


def load_model(model, model_path, flag, model_type):
    directories = os.listdir(model_path)
    submodel_names = ['generator', 'discriminator']
    submodels = [model.generator, model.discriminator]
    for i in range(2):
        submodel_name = submodel_names[i]
        submodel = submodels[i]

        filter_directories = list(filter(lambda x: x.find(flag) >= 0 and x.find(model_type) >= 0 and x.find(submodel_name) >= 0, directories))

        filter_directories.sort(reverse=False)
        # 支持 .weights.h5 和旧格式
        load_model_name = filter_directories[-1]
        if load_model_name.endswith('.weights.h5'):
            load_model_name = load_model_name  # 直接使用
        else:
            load_model_name = load_model_name[:-6] if len(load_model_name) > 6 else load_model_name  # 旧格式
        sys.stdout.write('load_model_name: ')
        sys.stdout.write(load_model_name)
        sys.stdout.write('\n')
        model_file_path = os.path.join(model_path, load_model_name)
        if not model_file_path.endswith('.weights.h5') and not os.path.exists(model_file_path):
            # 尝试添加 .weights.h5 后缀
            model_file_path = model_file_path + '.weights.h5'
        
        # 检查文件是否存在
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"模型文件不存在: {model_file_path}")
        
        # TensorFlow 2.x 中 load_weights 返回 None，不能链式调用
        try:
            submodel.load_weights(model_file_path)
            sys.stdout.write(f'成功加载: {load_model_name}\n')
        except Exception as e:
            sys.stdout.write(f'加载模型失败: {load_model_name}, 错误: {e}\n')
            raise

    return model

