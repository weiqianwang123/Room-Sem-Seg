
# README

This project has three main steps:

1. **Wall Detection**
2. **Density Map Generation**
3. **Room Segmentation**

---

## 1. Wall Detection

- **Repository:** [SpatialM](https://github.com/manycore-research/SpatialLM.git)
- Clone or download SpatialM, run its wall‑detection tool on your point‑cloud file
- Output: a text file (e.g. `a.txt`) listing detected wall segments

---

## 2. Density Map Generation

- **This Repository:** contains `generate_density.py`
- Configure `pointcloud_path` and `wall_txt_path` inside the script
- Run it to produce a density map image (e.g. `Results/demo/density.png`)

---

## 3. Room Segmentation

- **Repository:** [RoomSeg](https://github.com/weiqianwang123/RoomSegmentation.git)
- Clone or download RoomSeg, install dependencies, place your model checkpoint in `Weights/`
- Run the demo on your density map (`Results/demo/density.png`)
- Output: room‑segmentation results in your chosen output directory

---
