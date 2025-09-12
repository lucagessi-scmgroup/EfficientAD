import cv2
import numpy as np
import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='datasets/spesso/test/bad',
                        help="Select dataset input images folder")
    parser.add_argument('-m', '--mask', default='output/10/anomaly_maps/spesso/test/bad',
                    help="Select EfficientAD output folder with anomaly score map")
    parser.add_argument('-d', '--demo', default='',
                    help="Demo image for test. Default empty")
    return parser.parse_args()

def overlay_heatmap(img, heat, alpha=0.4, cmap=cv2.COLORMAP_TURBO, thr=None):
    """
    img: HxW (grigio) o HxWx3 (uint8 0..255, BGR o RGB)
    heat: HxW o h×w (float) – verrà ridimensionata a img
    alpha: opacità della heatmap [0..1]
    cmap: mappa colori OpenCV (es. COLORMAP_TURBO/JET/HOT)
    thr: se dato, applica la heatmap solo dove heat >= thr
    """
    # porta img a BGR uint8
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()
        # se è RGB e vuoi BGR per OpenCV:
        # base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    H, W = base.shape[:2]
    heat = cv2.resize(heat.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # normalizza 0..1 ignorando NaN
    m, M = np.nanmin(heat), np.nanmax(heat)
    denom = (M - m) if (M - m) > 1e-12 else 1.0
    h01 = (heat - m) / denom
    h01 = np.clip(h01, 0, 1)
    h01[np.isnan(h01)] = 0.0

    # colormap → uint8 BGR
    h255 = (h01 * 255).astype(np.uint8)
    hclr = cv2.applyColorMap(h255, cmap)  # BGR

    # maschera opzionale per soglia
    if thr is not None:
        mask = (h01 >= thr).astype(np.float32)[..., None]  # H×W×1
        hclr = (hclr * mask).astype(np.uint8)

    # alpha blend
    overlay = cv2.addWeighted(base, 1 - alpha, hclr, alpha, 0)

    return overlay  # BGR

# esempio d’uso
# img = cv2.imread("input.png")      # uint8
# heat = np.load("heat.npy")         # float HxW
# out = overlay_heatmap(img, heat, alpha=0.45, cmap=cv2.COLORMAP_TURBO, thr=None)
# cv2.imwrite("overlay.png", out)

import os

def main():
    config = get_argparse()
    print(f"Input dataset images folder: {config.input}")
    print(f"Anomaly map folder: {config.mask}")
    config.demo = "cropped_corr_0_sub_9"
    img_path = os.path.join(config.input, config.demo + ".png")
    img =  img = cv2.imread( img_path )
    mask_path = os.path.join(config.mask, config.demo + ".tiff")
    mask = cv2.imread( mask_path, cv2.IMREAD_UNCHANGED )
    out = overlay_heatmap( img, mask )
    cv2.imwrite("overlay.png", out)

if __name__ == '__main__':
    main()