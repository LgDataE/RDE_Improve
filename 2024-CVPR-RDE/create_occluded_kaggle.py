#!/usr/bin/env python3
"""
Simple script to create occluded images for Kaggle environment
"""
import os
import sys
import json
from PIL import Image, ImageDraw
import random

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
                
                # Resize and position
                max_size = min(width, height) // 3
                occlusion_img = occlusion_img.resize((random.randint(20, max_size), 
                                                    random.randint(20, max_size)))
                
                x = random.randint(0, max(1, width - occlusion_img.width))
                y = random.randint(0, max(1, height - occlusion_img.height))
                
                img.paste(occlusion_img, (x, y), occlusion_img)
            
            img.save(save_path)
            return True
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False

def main():
    # Configuration for RSTPReid
    data_root = '/kaggle/working/RDE_Improve/data/RSTPReid/'
    json_file = '/kaggle/working/RDE_Improve/data/RSTPReid/data_captions.json'
    output_dir = './occluded_data/RSTPReid'
    
    print("Creating occluded images for RSTPReid...")
    print(f"Data root: {data_root}")
    print(f"JSON file: {json_file}")
    print(f"Output dir: {output_dir}")
    
    # Check paths
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return
    
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize occlusion adder
    occluder = SimpleOcclusion()
    
    # Process images
    processed_ids = set()
    count = 0
    
    for item in data:
        img_path = item['img_path']
        img_id = item['id']
        
        # Only process first image per ID
        if img_id in processed_ids:
            continue
            
        processed_ids.add(img_id)
        
        # Paths - fix the path construction
        holistic_path = os.path.join(data_root, 'imgs', os.path.basename(img_path))
        occlusion_path = os.path.join(output_dir, 'imgs_occlusion_new', os.path.basename(img_path))
        
        # Create directory
        os.makedirs(os.path.dirname(occlusion_path), exist_ok=True)
        
        # Process
        if os.path.exists(holistic_path):
            if occluder.add_occlusion(holistic_path, occlusion_path):
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} images...")
        else:
            print(f"Image not found: {holistic_path}")
    
    print(f"Completed! Generated {count} occluded images")
    print(f"Occluded images saved in: {output_dir}")

if __name__ == '__main__':
    main()
