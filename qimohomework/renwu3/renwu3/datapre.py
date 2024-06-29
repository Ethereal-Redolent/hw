import os
import json
import struct
import numpy as np
from glob import glob

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("I", f.read(4))[0]
            model_id = struct.unpack("I", f.read(4))[0]
            width = struct.unpack("Q", f.read(8))[0]
            height = struct.unpack("Q", f.read(8))[0]
            params = struct.unpack("d" * 4, f.read(32))
            cameras[camera_id] = {"model": model_id, "width": width, "height": height, "params": params}
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("I", f.read(4))[0]
            qvec = struct.unpack("d" * 4, f.read(32))
            tvec = struct.unpack("d" * 3, f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]
            name = ""
            while True:
                char = struct.unpack("c", f.read(1))[0].decode("utf-8")
                if char == "\x00":
                    break
                name += char
            images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name}
            num_points2d = struct.unpack("Q", f.read(8))[0]
            points2d = []
            for _ in range(num_points2d):
                x = struct.unpack("d", f.read(8))[0]
                y = struct.unpack("d", f.read(8))[0]
                point3d_id = struct.unpack("Q", f.read(8))[0]
                points2d.append((x, y, point3d_id))
            images[image_id]["points2d"] = points2d
    return images

def read_points3d_binary(path):
    points3d = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack("Q", f.read(8))[0]
            xyz = struct.unpack("d" * 3, f.read(24))
            rgb = struct.unpack("B" * 3, f.read(3))
            error = struct.unpack("d", f.read(8))[0]
            track_length = struct.unpack("Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("I", f.read(4))[0]
                point2d_idx = struct.unpack("I", f.read(4))[0]
                track.append((image_id, point2d_idx))
            points3d[point3d_id] = {"xyz": xyz, "rgb": rgb, "error": error, "track": track}
    return points3d

def load_colmap_data(colmap_output_dir):
    camerasfile = os.path.join(colmap_output_dir, 'sparse/0/cameras.bin')
    imagesfile = os.path.join(colmap_output_dir, 'sparse/0/images.bin')
    pointsfile = os.path.join(colmap_output_dir, 'sparse/0/points3D.bin')

    cameras = read_cameras_binary(camerasfile)
    images = read_images_binary(imagesfile)
    points = read_points3d_binary(pointsfile)

    return cameras, images, points

def save_nerf_format(cameras, images, images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for image_id in images:
        image = images[image_id]
        qvec = image["qvec"]
        tvec = image["tvec"]
        image_path = os.path.join(images_dir, image["name"])

        frame = {
            "file_path": image_path,
            "rotation": list(qvec),
            "translation": list(tvec)
        }
        frames.append(frame)

    transform_dict = {
        "camera_angle_x": float(cameras[1]["params"][0]),
        "frames": frames
    }

    with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
        json.dump(transform_dict, f, indent=4)

# Example usage
colmap_output_dir = 'D:/python-learn/renwu333/colmap_output'
images_dir = 'D:/python-learn/renwu333/images'
output_dir = 'D:/python-learn/renwu333/nerf_format'

cameras, images, points = load_colmap_data(colmap_output_dir)
save_nerf_format(cameras, images, images_dir, output_dir)
print("Data pre-processing completed successfully.")
