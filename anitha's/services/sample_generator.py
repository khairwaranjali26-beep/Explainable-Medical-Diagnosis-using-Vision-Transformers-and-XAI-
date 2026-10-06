import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_base_chest_anatomy(width=512, height=512):
    """
    Synthesizes an anatomically structured chest radiograph base with:
    - Mediastinum & Cardiac silhouette
    - Bilateral lung fields (radiolucent/dark)
    - Thoracic ribs & clavicles
    - Diaphragmatic domes
    - Spine & vascular arborization
    """
    img = np.zeros((height, width), dtype=np.float32)
    
    # Background soft tissue / thoracic wall
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Thoracic cage ellipse
    dx = (x_coords - width / 2) / (width * 0.42)
    dy = (y_coords - height * 0.52) / (height * 0.46)
    cage_mask = (dx**2 + dy**2) < 1.0
    img[cage_mask] = 140.0 + np.random.normal(0, 4, img[cage_mask].shape)
    
    # Bilateral Lungs (darker radiolucent zones)
    # Right Lung (viewer's left: x ~ 100 to 220, y ~ 120 to 410)
    rx = (x_coords - 165) / 65.0
    ry = (y_coords - 265) / 135.0
    right_lung_mask = (rx**2 + ry**2) < 1.0
    # Left Lung (viewer's right: x ~ 290 to 410, y ~ 120 to 410)
    lx = (x_coords - 340) / 60.0
    ly = (y_coords - 265) / 130.0
    left_lung_mask = (lx**2 + ly**2) < 1.0
    
    # Soft lung tissue absorption (healthy lungs are dark in X-ray)
    img[right_lung_mask] = 45.0 + np.random.normal(0, 6, img[right_lung_mask].shape)
    img[left_lung_mask] = 45.0 + np.random.normal(0, 6, img[left_lung_mask].shape)
    
    # Cardiac silhouette (dense white/gray in left chest)
    cx = (x_coords - 290) / 75.0
    cy = (y_coords - 345) / 60.0
    heart_mask = (cx**2 + cy**2) < 1.0
    img[heart_mask] = 175.0 + np.random.normal(0, 5, img[heart_mask].shape)
    
    # Aortic knob and mediastinum
    mediastinum_mask = (np.abs(x_coords - width / 2) < 32) & (y_coords > 80) & (y_coords < 380)
    img[mediastinum_mask] = 185.0 + np.random.normal(0, 6, img[mediastinum_mask].shape)
    
    # Diaphragmatic domes
    r_diaphragm = (x_coords > 80) & (x_coords < 240) & (y_coords > 400 - (1 - ((x_coords - 165)/80)**2)*40)
    l_diaphragm = (x_coords > 260) & (x_coords < 430) & (y_coords > 415 - (1 - ((x_coords - 345)/85)**2)*35)
    img[r_diaphragm] = 200.0
    img[l_diaphragm] = 190.0
    
    # Convert to PIL for drawing fine anatomical details (ribs, clavicles, vascular markings)
    pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    
    # Draw Clavicles
    draw.line([(110, 115), (240, 135)], fill=210, width=8)
    draw.line([(402, 115), (272, 135)], fill=210, width=8)
    
    # Draw 7 pairs of posterior thoracic ribs
    for i in range(7):
        y_pos = 150 + i * 36
        # Right ribs
        draw.arc([75, y_pos - 15, 255, y_pos + 45], start=180, end=350, fill=135, width=6)
        # Left ribs
        draw.arc([255, y_pos - 15, 435, y_pos + 45], start=190, end=360, fill=135, width=6)
        
    # Pulmonary vascular branch markings
    for dy in range(-40, 45, 12):
        draw.line([(245, 270), (180, 260 + dy)], fill=100, width=2)
        draw.line([(265, 275), (325, 270 + dy)], fill=95, width=2)

    # Spine column
    draw.rectangle([width//2 - 12, 60, width//2 + 12, 480], fill=165)
    
    # Smooth with radiologic blur to simulate scatter & tissue attenuation
    smoothed = pil_img.filter(ImageFilter.GaussianBlur(radius=3.5))
    return smoothed

def generate_sample_xrays():
    """
    Generates and saves:
    - Normal Chest X-ray
    - Pneumonia Chest X-ray (with alveolar infiltrates & lobar consolidation)
    """
    os.makedirs(Config.SAMPLE_FOLDER, exist_ok=True)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # 1. Normal Chest Radiograph
    normal_pil = create_base_chest_anatomy()
    # Add subtle chest X-ray noise and contrast
    normal_arr = np.array(normal_pil, dtype=np.float32)
    normal_arr += np.random.normal(0, 3.5, normal_arr.shape)
    normal_arr = np.clip(normal_arr, 0, 255).astype(np.uint8)
    normal_final = Image.fromarray(normal_arr).convert('RGB')
    
    # Add orientation marker "R" (radiology standard)
    draw_n = ImageDraw.Draw(normal_final)
    draw_n.text((25, 25), "R", fill=(240, 240, 240))
    draw_n.text((25, 45), "PA ERECT", fill=(180, 180, 180))
    
    normal_path = os.path.join(Config.SAMPLE_FOLDER, 'sample_normal_chest_xray.jpg')
    normal_final.save(normal_path, quality=95)
    print(f"[Sample Generator] Created: {normal_path}")
    
    # 2. Pneumonia Chest Radiograph
    # Has dense focal opacification in Right Middle & Lower Lobe with patchy consolidation
    pneu_pil = create_base_chest_anatomy()
    pneu_arr = np.array(pneu_pil, dtype=np.float32)
    
    y_grid, x_grid = np.mgrid[0:512, 0:512]
    # Right lower lobe consolidation hotspot (viewer's left side: x ~ 130-220, y ~ 260-390)
    dist_r = np.sqrt(((x_grid - 175) / 52.0)**2 + ((y_grid - 325) / 60.0)**2)
    consolidation_mask = dist_r < 1.0
    
    # Apply patchy alveolar dense infiltration
    infiltrate_noise = np.random.normal(120, 15, (512, 512))
    pneu_arr[consolidation_mask] += (1.0 - dist_r[consolidation_mask]) * infiltrate_noise[consolidation_mask] * 0.95
    
    # Add second smaller patchy nodular opacity in right mid zone
    dist_mid = np.sqrt(((x_grid - 190) / 35.0)**2 + ((y_grid - 250) / 35.0)**2)
    mid_mask = dist_mid < 1.0
    pneu_arr[mid_mask] += (1.0 - dist_mid[mid_mask]) * 95.0
    
    # Clip and convert
    pneu_arr = np.clip(pneu_arr, 0, 255).astype(np.uint8)
    pneu_pil = Image.fromarray(pneu_arr).filter(ImageFilter.GaussianBlur(radius=2.5))
    pneu_final = pneu_pil.convert('RGB')
    
    draw_p = ImageDraw.Draw(pneu_final)
    draw_p.text((25, 25), "R", fill=(240, 240, 240))
    draw_p.text((25, 45), "PA ERECT", fill=(180, 180, 180))
    
    pneu_path = os.path.join(Config.SAMPLE_FOLDER, 'sample_pneumonia_chest_xray.jpg')
    pneu_final.save(pneu_path, quality=95)
    print(f"[Sample Generator] Created: {pneu_path}")
    
    return normal_path, pneu_path

if __name__ == '__main__':
    generate_sample_xrays()
