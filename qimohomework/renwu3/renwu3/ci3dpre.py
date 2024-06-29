import os
import numpy as np
from read_write_model import read_model

def get_poses_bounds(sparse_dir):
    cameras, images, points3D = read_model(path=sparse_dir, ext='.bin')

    w2c_mats = []
    bds = []

    for image_id in images:
        image = images[image_id]
        cam = cameras[image.camera_id]

        R = image.qvec2rotmat()
        t = image.tvec.reshape(3, 1)
        m = np.concatenate([np.concatenate([R, t], 1), np.array([[0, 0, 0, 1]])], 0)
        w2c_mats.append(m)

        pts = np.array([points3D[pt_id].xyz for pt_id in image.point3D_ids if pt_id != -1])
        if len(pts) == 0:
            continue
        min_bounds = np.min(pts, 0)
        max_bounds = np.max(pts, 0)
        bounds = np.concatenate([min_bounds, max_bounds], 0)
        bds.append(bounds)

    w2c_mats = np.stack(w2c_mats, 0)
    bds = np.stack(bds, 0)

    # 打印 w2c_mats 和 bds 的形状进行调试
    print(f"w2c_mats shape: {w2c_mats.shape}")
    print(f"bds shape: {bds.shape}")

    # 将 bds 转换为 (N, 4, 4)
    bds_expanded = np.zeros((bds.shape[0], 4, 4))
    bds_expanded[:, :3, 0] = bds[:, :3]
    bds_expanded[:, :3, 1] = bds[:, 3:]
    bds_expanded[:, 3, 3] = 1

    # 打印 bds_expanded 的形状进行调试
    print(f"bds_expanded shape: {bds_expanded.shape}")

    # 连接 w2c_mats 和 bds
    poses_bounds = np.concatenate([w2c_mats, bds_expanded], axis=2)
    return poses_bounds

sparse_dir = r'D:\python-learn\renwu333\colmap_output\sparse\0'
poses_bounds = get_poses_bounds(sparse_dir)
np.save(os.path.join(sparse_dir, 'poses_bounds.npy'), poses_bounds)
