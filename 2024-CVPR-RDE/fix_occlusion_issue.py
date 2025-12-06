#!/usr/bin/env python3
"""
Fix occlusion issue by generating missing occluded images or using holistic dataset
"""
import os
import sys
import json
import shutil
from pathlib import Path

def fix_occlusion_paths():
    """Fix the occlusion path issue by copying holistic images to occlusion paths"""
    
    # Configuration
    data_root = '/kaggle/working/RDE_Improve/data/RSTPReid/'
    json_file = '/kaggle/working/RDE_Improve/data/RSTPReid/data_captions.json'
    occlusion_dir = './occluded_data/RSTPReid/imgs_occlusion_new/'
    
    print("Fixing occlusion path issues...")
    print(f"Data root: {data_root}")
    print(f"Occlusion dir: {occlusion_dir}")
    
    # Check if data exists
    if not os.path.exists(json_file):
        print(f"ERROR: JSON file not found: {json_file}")
        return False
        
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Create occlusion directory
    os.makedirs(occlusion_dir, exist_ok=True)
    
    # Process images - copy holistic to occlusion paths
    processed_count = 0
    missing_count = 0
    
    for item in data:
        img_path = item['img_path']
        filename = os.path.basename(img_path)
        
        holistic_path = os.path.join(data_root, 'imgs', filename)
        occlusion_path = os.path.join(occlusion_dir, filename)
        
        if os.path.exists(holistic_path):
            # Copy holistic image to occlusion path (as fallback)
            if not os.path.exists(occlusion_path):
                shutil.copy2(holistic_path, occlusion_path)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")
        else:
            missing_count += 1
            print(f"Missing holistic image: {holistic_path}")
    
    print(f"Completed! Processed {processed_count} images")
    if missing_count > 0:
        print(f"Warning: {missing_count} holistic images were missing")
    
    return True

def create_simple_occlusions():
    """Create simple occluded images if needed"""
    try:
        from create_occluded_kaggle import main as create_main
        print("Running occlusion generation...")
        create_main()
        return True
    except Exception as e:
        print(f"Occlusion generation failed: {e}")
        return False

def main():
    print("=== RDE Occlusion Fix Script ===")
    print("This script fixes the hanging issue by ensuring all required images exist")
    
    # Try to generate proper occlusions first
    if not create_simple_occlusions():
        print("Falling back to copying holistic images...")
        if not fix_occlusion_paths():
            print("Failed to fix occlusion paths")
            sys.exit(1)
    
    print("Occlusion fix completed successfully!")
    print("You can now run: !sh run_rde.sh")

if __name__ == '__main__':
    main()
