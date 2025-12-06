#!/usr/bin/env python3
"""
Simple script to create occluded images for Kaggle environment
"""
import os
import sys
import json
from PIL import Image, ImageDraw
import random
import glob

# Add current directory to path for imports
sys.path.append('.')

# Simple occlusion adding class
class SimpleOcclusion():
    def __init__(self, occlusion_dir='./occlusion_img'):
        self.files = []
        if os.path.exists(occlusion_dir):
            for file in os.listdir(occlusion_dir):
                if file.endswith('.png'):
                    self.files.append(os.path.join(occlusion_dir, file))
        print(f"Found {len(self.files)} occlusion objects")
    
    def add_occlusion(self, img_path, save_path):
        """Add random occlusion to image"""
        try:
            img = Image.open(img_path)
            width, height = img.size
            
            # Create simple rectangle occlusion if no objects available
            if len(self.files) == 0:
                # Random rectangle
                x = random.randint(0, width//2)
                y = random.randint(0, height//2)
                w = random.randint(width//8, width//4)
                h = random.randint(height//8, height//4)
                
                draw = ImageDraw.Draw(img)
                draw.rectangle([x, y, x+w, y+h], fill=(0, 0, 0))
            else:
                # Use occlusion object
                occlusion_file = random.choice(self.files)
                occlusion_img = Image.open(occlusion_file).convert('RGBA')
                
                # Resize and position (robust to very small images)
                max_size = max(1, min(width, height) // 3)
                min_size = 20

                # If image is too small, skip occlusion but still save
                if max_size <= min_size:
                    img.save(save_path)
                    return True

                occ_w = random.randint(min_size, max_size)
                occ_h = random.randint(min_size, max_size)
                occlusion_img = occlusion_img.resize((occ_w, occ_h))
                
                x = random.randint(0, max(1, width - occlusion_img.width))
                y = random.randint(0, max(1, height - occlusion_img.height))
                
                img.paste(occlusion_img, (x, y), occlusion_img)
            
            img.save(save_path)
            return True
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False

def main():
    """Generate occluded images for all RSTPReid imgs on Kaggle.

    Reads images from:
        /kaggle/working/RDE_Improve/data/RSTPReid/imgs/*.jpg
    Writes occluded images to:
        ./occluded_data/RSTPReid/imgs_occlusion_new/<filename>.jpg
    This matches the fallback path used by change_path in datasets/build.py.
    """

    # Fixed paths for Kaggle RSTPReid
    data_root = '/kaggle/working/RDE_Improve/data/RSTPReid/'
    img_dir = os.path.join(data_root, 'imgs')
    output_root = './occluded_data/RSTPReid'
    out_dir = os.path.join(output_root, 'imgs_occlusion_new')

    print("Creating occluded images for RSTPReid...")
    print(f"Image dir: {img_dir}")
    print(f"Output dir: {out_dir}")

    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Collect all jpg images
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    print(f"Found {len(img_paths)} images")

    occluder = SimpleOcclusion()

    count = 0
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        save_path = os.path.join(out_dir, filename)

        # Allow re-run: skip if occluded version already exists
        if os.path.exists(save_path):
            continue

        if occluder.add_occlusion(img_path, save_path):
            count += 1
            if count % 500 == 0:
                print(f"Processed {count} images...")

    print(f"Completed! Generated {count} occluded images")
    print(f"Occluded images saved in: {out_dir}")

if __name__ == '__main__':
    main()
