import os
import sys
sys.path.append('/path/to/colmap/scripts/python')
from read_write_model import read_model

# 运行生成脚本
sparse_dir = './sparse/0'
poses_bounds = get_poses_bounds(sparse_dir)
np.save(os.path.join(sparse_dir, 'poses_bounds.npy'), poses_bounds)
