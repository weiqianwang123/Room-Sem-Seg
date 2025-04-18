import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import re

# ---------- 解析 wall 信息 ----------
def parse_wall_line(line):
    pattern = r'Wall\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'
    match = re.search(pattern, line)
    if match:
         x1 = float(match.group(1))
         y1 = float(match.group(2))
         z1 = float(match.group(3))
         x2 = float(match.group(4))
         y2 = float(match.group(5))
         z2 = float(match.group(6))
         height = float(match.group(7))
         angle = float(match.group(8))
         return np.array([x1, y1, z1]), np.array([x2, y2, z2]), height, angle
    return None

# ---------- 获取 wall mask ----------
import cv2  # 用于画线

def get_wall_mask(ps, min_coords, max_coords, width, height, wall_lines):

    mask = np.zeros((height, width), dtype=np.uint8)

    for line in wall_lines:
        result = parse_wall_line(line)
        if result:
            p1, p2, _, _ = result

            # 坐标归一化并映射到图像尺寸
            norm_p1 = ((p1[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * [width, height]).astype(int)
            norm_p2 = ((p2[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * [width, height]).astype(int)

            # 注意 OpenCV 是 (x, y)，但图像是 [height, width] 的格式
            norm_p1 = np.clip(norm_p1, 0, [width - 1, height - 1])
            norm_p2 = np.clip(norm_p2, 0, [width - 1, height - 1])

            # 用 OpenCV 画线（墙壁粗一点）
            cv2.line(mask, tuple(norm_p1), tuple(norm_p2), color=255, thickness=1)

    # 归一化到 0.0 ~ 1.0 的 float 类型
    return mask.astype(np.float32) / 255.0


# ---------- 密度图生成 ----------
def generate_density(point_cloud, wall_lines, width=256, height=256):
    ps = point_cloud
    image_res = np.array((width, height))

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords

    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    wall_mask = get_wall_mask(ps, min_coords, max_coords, width, height, wall_lines)

    normalization_dict = {
        "min_coords": min_coords,
        "max_coords": max_coords,
        "image_res": image_res,
    }

    coordinates = np.round(
        (ps[:, :2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * image_res
    ).astype(int)

    coordinates = np.clip(coordinates, 0, image_res - 1)

    density = np.zeros((height, width), dtype=np.float32)
    unique_coords, counts = np.unique(coordinates, axis=0, return_counts=True)
    density[unique_coords[:, 1], unique_coords[:, 0]] = counts
    density /= np.max(density)  # 归一化

    # ✅ 添加墙壁信息
    density += (wall_mask * np.max(density) / 10).astype(np.float32)

    return density, wall_mask, normalization_dict

# ---------- 保存 PNG ----------
def save_density_as_png(density_map, output_path, cmap="hot"):
    plt.imsave(output_path, density_map, cmap=cmap)
    print(f"Density map saved as PNG at {output_path}")

    density_image = (density_map * 255).astype(np.uint8)
    Image.fromarray(density_image).save(output_path.replace(".png", "_gray.png"))
    print(f"Density map (grayscale) saved at {output_path.replace('.png', '_gray.png')}")

# ---------- 主程序 ----------
def main():
    ply_file = "/home/qianwei/SpatialLM/pcd/test_set/17DRP5sb8fy/house_segmentations/17DRP5sb8fy.ply"
    txt_file = "/home/qianwei/SpatialLM/test.txt"

    pcd = o3d.io.read_point_cloud(ply_file)
    point_cloud = np.asarray(pcd.points)

    # 读取墙壁定义
    with open(txt_file, "r") as f:
        wall_lines = [line for line in f if "Wall(" in line]

    # 生成密度图
    density, wall_mask, normalization_dict = generate_density(point_cloud, wall_lines, width=256, height=256)

    # 保存图像
    output_path = "/home/qianwei/SpatialLM/density.png"
    save_density_as_png(density, output_path)

if __name__ == "__main__":
    main()
