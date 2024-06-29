import os
import subprocess
import shutil

# 步骤1：如果存在现有的nerf-pytorch目录，则将其删除
if os.path.exists('nerf-pytorch'):
    try:
        shutil.rmtree('nerf-pytorch')
    except Exception as e:
        print(f"无法删除目录 'nerf-pytorch'：{e}")
        exit(1)


# 步骤2：安装支持CUDA的PyTorch
subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--extra-index-url', 'https://download.pytorch.org/whl/cu116'])

# 步骤3：克隆nerf-pytorch仓库
try:
    result = subprocess.run(['git', 'clone', 'https://github.com/yenchenlin/nerf-pytorch.git'], check=True)
    os.chdir('nerf-pytorch')
except subprocess.CalledProcessError:
    print("GitHub克隆失败。请确保您有互联网连接并重试。")
    exit(1)

# 步骤4：更新requirements.txt
with open('requirements.txt', 'w') as f:
    f.write("""
torch
torchvision
torchaudio
tensorboard
numpy
opencv-python
matplotlib
imageio
imageio[ffmpeg]
scipy
configargparse
""")

# 步骤5：安装依赖项
subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

# 步骤6：创建训练配置文件
config_content = """
expname = 'colmap_experiment'
basedir = './logs'
datadir = r'D:\python-learn\renwu333\colmap_output\sparse'

# training options
N_rand = 1024
lrate = 5e-4
lrate_decay = 100
netdepth = 8
netwidth = 256
use_viewdirs = True

# rendering options
N_samples = 64
N_importance = 128
"""

config_path = 'configs/colmap.txt'
with open(config_path, 'w') as f:
    f.write(config_content)

# 步骤7：训练模型
subprocess.run(['python', 'run_nerf.py', '--config', config_path])
