import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon

def polygon_to_mask(polygon, image_shape):
    """
    将单个多边形 rasterize 成二值 mask。
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon).reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def visualize_polygons(polys, image_shape, save_path):
    """
    将一组多边形渲染到空白图像上，每个多边形用随机颜色填充，并保存。
    """
    canvas = np.ones(image_shape, dtype=np.uint8) * 255
    for poly in polys:
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], color)
    cv2.imwrite(save_path, canvas)
    print(f"Saved visualization to {save_path}")

def calculate_ap50_and_iou(gt_polys, pred_polys, image_shape, iou_thresh=0.5):
    """
    对一张图的所有 GT 多边形和预测多边形计算 AP50 和平均 IoU。
    """
    gt_masks   = [polygon_to_mask(p, image_shape) for p in gt_polys]
    pred_masks = [polygon_to_mask(p, image_shape) for p in pred_polys]

    tp = 0
    total_iou = 0.0

    # 对每个 GT，找最佳匹配
    for gm in gt_masks:
        best_iou = 0.0
        for pm in pred_masks:
            inter = np.logical_and(gm, pm).sum()
            union = np.logical_or(gm, pm).sum()
            iou = inter / (union + 1e-7)
            best_iou = max(best_iou, iou)
        if best_iou >= iou_thresh:
            tp += 1
        total_iou += best_iou

    fn        = len(gt_masks) - tp
    fp        = len(pred_masks) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0
    ap50      = precision * recall
    avg_iou   = total_iou / len(gt_masks) if gt_masks else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return ap50, avg_iou, tp, fp, fn,f1, precision,recall 

if __name__ == "__main__":
    # --- 配置部分 ---
    image_shape = (256, 256, 3)      # 根据你的实际分辨率修改
    output_dir  = "./vis_and_eval"
    os.makedirs(output_dir, exist_ok=True)

    GT_JSON_PATH   = "/home/qianwei/SpatialLM/pcd/test_set/q9vSo1VnCiC/floor_0_annotations.json"
    PRED_JSON_PATH = "/home/qianwei/RoomFormer-main/my_method/data/room_polygons.json"

    # --- 加载 GT 多边形 ---
    with open(GT_JSON_PATH, 'r') as f:
        gt_entries = json.load(f)
    gt_polys = []
    for entry in gt_entries:
        for poly in entry["contours"]:
            gt_polys.append(poly)

    # --- 加载预测多边形 ---
    with open(PRED_JSON_PATH, 'r') as f:
        pred_data = json.load(f)
    pred_polys = pred_data["room_polys"]

    # --- 可视化 GT & 预测 ---
    gt_vis_path   = os.path.join(output_dir, "gt_vis.png")
    pred_vis_path = os.path.join(output_dir, "pred_vis.png")
    visualize_polygons(gt_polys,   image_shape, gt_vis_path)
    visualize_polygons(pred_polys, image_shape, pred_vis_path)

    # --- 计算评估指标 ---
    ap50, avg_iou, tp, fp, fn ,f1,precision,recall = calculate_ap50_and_iou(
        gt_polys, pred_polys, image_shape
    )
    print("=== Evaluation Results ===")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}")
    print(f"AP50       = {ap50:.3f}")
    print(f"Mean IoU   = {avg_iou:.3f}")
    print(f"Precision = {precision:.3f}")
    print(f"Recall    = {recall:.3f}")
    print(f"F1 Score  = {f1:.3f}")
