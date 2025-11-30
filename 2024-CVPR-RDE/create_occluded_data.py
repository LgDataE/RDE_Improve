#!/usr/bin/env python3
"""
Script to generate occluded images for RDE dataset
Adapted from MGCC process_data.py
"""
import numpy as np
import math
from PIL import Image, ImageDraw
import random
import os
import argparse
from utils.read_write_data import read_json

class Occlusion_Adding():
    def __init__(self):
        self.path = './occlusion_img/'
        self.files = []
        for file in os.listdir(self.path):
            if file.endswith('.png'):
                self.files.append(os.path.join(self.path, file))
    
    def __call__(self, holistic_path, occlusion_path, save_path):
        """
        Add occlusion to holistic image and save
        """
        holistic_img = Image.open(holistic_path)
        width, height = holistic_img.size
        
        # Random occlusion parameters
        occlusion_file = random.choice(self.files)
        occlusion_img = Image.open(occlusion_file).convert('RGBA')
        
        # Random position and size
        max_occlusion_size = min(width, height) // 3
        occlusion_img = occlusion_img.resize((random.randint(20, max_occlusion_size), 
                                            random.randint(20, max_occlusion_size)))
        
        # Random position
        x = random.randint(0, width - occlusion_img.width)
        y = random.randint(0, height - occlusion_img.height)
        
        # Paste occlusion
        holistic_img.paste(occlusion_img, (x, y), occlusion_img)
        holistic_img.save(save_path)
        
        return 0

def generate_occlusion(args):
    json_dir = args.json_root
    reid_raw_data = read_json(json_dir)
    Occlusion = Occlusion_Adding()
    name = args.data_name

    occlusion_path = args.occlusion_path
    
    if name == 'CUHK-PEDES':
        last_id = -1
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['file_path'])
            
            # Determine save path
            if "cam_a" in holistic_path:
                save_path = holistic_path.replace('cam_a', 'cam_a_occlusion_new')
            elif "cam_b" in holistic_path:
                save_path = holistic_path.replace('cam_b', 'cam_b_occlusion_new')
            elif "CUHK01" in holistic_path:
                save_path = holistic_path.replace('CUHK01', 'CUHK01_occlusion_new')     
            elif "CUHK03" in holistic_path:
                save_path = holistic_path.replace('CUHK03', 'CUHK03_occlusion_new') 
            elif "Market" in holistic_path:
                save_path = holistic_path.replace('Market', 'Market_occlusion_new')       
            elif "test_query" in holistic_path:
                save_path = holistic_path.replace('test_query', 'test_query_occlusion_new')
            elif "train_query" in holistic_path:
                save_path = holistic_path.replace('train_query', 'train_query_occlusion_new')
            
            # Create directory
            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)   
            
            current_id = data['id']
            if last_id != current_id:  # Generate occlusion for first image of each ID
                Occlusion(holistic_path, occlusion_path, save_path)
                last_id = current_id
                print(f"Generated occlusion for ID {current_id}: {save_path}")
            else:   # Copy holistic image for other images
                holistic_img = Image.open(holistic_path)
                holistic_img.save(save_path)
                
    elif name == 'ICFG-PEDES':
        last_id = -1
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['file_path'])
            
            if "test" in holistic_path:
                save_path = holistic_path.replace('test', 'test_occlusion_new')
            elif "train" in holistic_path:
                save_path = holistic_path.replace('train', 'train_occlusion_new')
                
            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)
            
            current_id = data['id']
            if last_id != current_id:  # Generate occlusion for first image of each ID
                Occlusion(holistic_path, occlusion_path, save_path)
                last_id = current_id
                print(f"Generated occlusion for ID {current_id}: {save_path}")
            else:   # Copy holistic image for other images
                holistic_img = Image.open(holistic_path)
                holistic_img.save(save_path)
                
    elif name == 'RSTPReid':
        last_id = -1
        for data in reid_raw_data:
            holistic_path = os.path.join(args.data_root, 'imgs', data['img_path'])
            
            if "imgs" in holistic_path:
                save_path = holistic_path.replace('imgs', 'imgs_occlusion_new')
                
            save_path2 = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_path2, exist_ok=True)
            
            current_id = data['id']
            if last_id != current_id:  
                Occlusion(holistic_path, occlusion_path, save_path)
                last_id = current_id
                print(f"Generated occlusion for ID {current_id}: {save_path}")
            else:   
                holistic_img = Image.open(holistic_path)
                holistic_img.save(save_path)
    
    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='Generate occluded images for RDE dataset')
    parser.add_argument('--data_name', default='CUHK-PEDES', type=str, 
                       choices=['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'])    
    parser.add_argument('--data_root', default='./data/CUHK-PEDES/', type=str)       
    parser.add_argument('--json_root', default='./data/CUHK-PEDES/reid_raw.json', type=str)
    parser.add_argument('--occlusion_path', default='./occlusion_img', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    print(f"Generating occluded images for {args.data_name}")
    print(f"Data root: {args.data_root}")
    print(f"JSON file: {args.json_root}")
    print(f"Occlusion objects: {args.occlusion_path}")
    
    # Check if paths exist
    if not os.path.exists(args.data_root):
        print(f"Error: Data root {args.data_root} does not exist!")
        exit(1)
    if not os.path.exists(args.json_root):
        print(f"Error: JSON file {args.json_root} does not exist!")
        exit(1)
    if not os.path.exists(args.occlusion_path):
        print(f"Error: Occlusion path {args.occlusion_path} does not exist!")
        exit(1)
    
    # Generate occluded images
    generate_occlusion(args)
    print("Occluded image generation completed!")
