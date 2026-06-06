import cv2
import numpy as np
from PIL import Image
import sys

name = sys.argv[1] if len(sys.argv) > 1 else 'g0001'
base = '/mnt/d/IRISCIENCE/Datasets/REFUGE/train'

img = np.array(Image.open(f'{base}/Images_Cropped/{name}.jpg'))
mask = np.array(Image.open(f'{base}/Masks_Cropped/{name}.png'))

unique = np.unique(mask)
if set(unique).issubset({0, 128, 255}):
    new_mask = np.zeros_like(mask)
    new_mask[mask == 128] = 1
    new_mask[mask == 0]   = 2
    mask = new_mask
elif not set(unique).issubset({0, 1, 2}):
    raise ValueError(f"Formato de mascara no reconocido: {unique}")

disc_mask = (mask >= 1).astype(np.uint8)
cup_mask  = (mask == 2).astype(np.uint8)

contours_disc, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_cup,  _ = cv2.findContours(cup_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours_disc or not contours_cup:
    raise ValueError("No se encontraron contornos de disco o copa")

disc_cnt = max(contours_disc, key=cv2.contourArea)
cup_cnt  = max(contours_cup,  key=cv2.contourArea)

def vertical_extent(contour):
    ys = contour[:, 0, 1]
    return ys.max() - ys.min()

disc_height = vertical_extent(disc_cnt)
cup_height  = vertical_extent(cup_cnt)
vcdr = cup_height / disc_height if disc_height > 0 else 0

M = cv2.moments(cup_cnt)
if M["m00"] == 0:
    x, y, w, h = cv2.boundingRect(cup_cnt)
    cx, cy = x + w//2, y + h//2
else:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

disc_pts = disc_cnt.reshape(-1, 2).astype(np.float32)
cup_pts  = cup_cnt.reshape(-1, 2).astype(np.float32)

disc_vectors = disc_pts - np.array([cx, cy])
disc_radii   = np.linalg.norm(disc_vectors, axis=1)
max_disc_radius = disc_radii.max()

N_ANGLES = 360
angles = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
rim_widths = []

for theta in angles:
    dx = np.cos(theta)
    dy = np.sin(theta)
    disc_proj = (disc_pts[:,0] - cx) * dx + (disc_pts[:,1] - cy) * dy
    disc_proj = disc_proj[disc_proj > 0]
    cup_proj = (cup_pts[:,0] - cx) * dx + (cup_pts[:,1] - cy) * dy
    cup_proj = cup_proj[cup_proj > 0]
    if len(disc_proj) > 0 and len(cup_proj) > 0:
        r_disc = disc_proj.max()
        r_cup  = cup_proj.max()
        width = max(0, r_disc - r_cup)
        rim_widths.append(width)

rim_widths = np.array(rim_widths)
min_rim_width = rim_widths.min() if len(rim_widths) > 0 else 0
rdr_real = min_rim_width / max_disc_radius if max_disc_radius > 0 else 0

cup_area   = (mask == 2).sum()
disc_area  = (mask >= 1).sum()
rim_area   = disc_area - cup_area
cdr_area   = cup_area / disc_area if disc_area > 0 else 0
rim_ratio  = rim_area / disc_area if disc_area > 0 else 0

overlay = img.copy().astype(np.uint8)
overlay[mask == 2] = [255, 0, 0]
overlay[mask == 1] = [0, 255, 0]

disc_ys = disc_cnt[:, 0, 1]
disc_top    = disc_cnt[disc_ys.argmin()][0]
disc_bottom = disc_cnt[disc_ys.argmax()][0]

cup_ys = cup_cnt[:, 0, 1]
cup_top    = cup_cnt[cup_ys.argmin()][0]
cup_bottom = cup_cnt[cup_ys.argmax()][0]

cv2.line(overlay, tuple(disc_top),    tuple(disc_bottom),    (255, 255, 0), 2)
cv2.line(overlay, tuple(cup_top),     tuple(cup_bottom),     (0, 255, 255), 2)

if len(rim_widths) > 0:
    min_idx = rim_widths.argmin()
    min_theta = angles[min_idx]
    r_disc_at_min = max_disc_radius
    end_x = int(cx + r_disc_at_min * np.cos(min_theta))
    end_y = int(cy + r_disc_at_min * np.sin(min_theta))
    cv2.line(overlay, (cx, cy), (end_x, end_y), (255, 0, 255), 2)

out = f'overlay_full_{name}.png'
Image.fromarray(overlay).save(out)

print(f"{'='*60}")
print(f"Archivo: {out}")
print(f"{'='*60}")
print(f"vCDR (vertical)      = {vcdr:.4f}   <- Estandar clinico")
print(f"CDR_area             = {cdr_area:.4f}   <- Proporcion de areas (no clinico)")
print(f"{'-'*60}")
print(f"RDR REAL (DDLS)      = {rdr_real:.4f}   <- Ancho minimo del rim / radio disco")
print(f"RIM_ratio (1-CDR)    = {rim_ratio:.4f}   <- NO es RDR real, solo area")
print(f"{'-'*60}")
print(f"Altura disco (px)    = {disc_height}")
print(f"Altura copa (px)     = {cup_height}")
print(f"Radio disco max (px) = {max_disc_radius:.1f}")
print(f"Ancho rim min (px)   = {min_rim_width:.1f}")
print(f"{'='*60}")
