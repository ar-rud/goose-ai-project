import os
import numpy as np
import pandas as pd
from PIL import Image

SUPER_CATEGORIES = [
    'vegetation', 'terrain', 'sky', 'construction', 
    'vehicle', 'road', 'object', 'void', 'sign', 
    'human', 'water', 'animal'
]
cat_to_id = {cat: idx for idx, cat in enumerate(SUPER_CATEGORIES)}

CLASS_TO_CATEGORY = {
    # vegetation
    'bush': 'vegetation', 'crops': 'vegetation', 'forest': 'vegetation', 
    'hedge': 'vegetation', 'high_grass': 'vegetation', 'leaves': 'vegetation', 
    'low_grass': 'vegetation', 'moss': 'vegetation', 'scenery_vegetation': 'vegetation', 
    'tree_crown': 'vegetation', 'tree_root': 'vegetation', 'tree_trunk': 'vegetation',
    
    # terrain
    'asphalt': 'terrain', 'cobble': 'terrain', 'gravel': 'terrain', 
    'snow': 'terrain', 'soil': 'terrain',
    
    # sky
    'sky': 'sky',
    
    # construction
    'bridge': 'construction', 'building': 'construction', 'container': 'construction', 
    'debris': 'construction', 'fence': 'construction', 'guard_rail': 'construction', 
    'tunnel': 'construction', 'wall': 'construction', 'wire': 'construction',
    
    # vehicle
    'bicycle': 'vehicle', 'bus': 'vehicle', 'car': 'vehicle', 'caravan': 'vehicle', 
    'heavy_machinery': 'vehicle', 'kick_scooter': 'vehicle', 'motorcycle': 'vehicle', 
    'on_rails': 'vehicle', 'trailer': 'vehicle', 'truck': 'vehicle', 
    'military_vehicle': 'vehicle',
    
    # road
    'bikeway': 'road', 'curb': 'road', 'pedestrian_crossing': 'road', 
    'rail_track': 'road', 'road_marking': 'road', 'sidewalk': 'road',
    
    # object
    'barrel': 'object', 'obstacle': 'object', 'pipe': 'object', 
    'pole': 'object', 'rock': 'object', 'street_light': 'object',
    
    # void
    'ego_vehicle': 'void', 'outlier': 'void', 'undefined': 'void',
    
    # sign
    'barrier_tape': 'sign', 'boom_barrier': 'sign', 'misc_sign': 'sign', 
    'road_block': 'sign', 'traffic_cone': 'sign', 'traffic_light': 'sign', 
    'traffic_sign': 'sign',
    
    # human
    'person': 'human', 'rider': 'human',
    
    # water
    'water': 'water',
    
    # animal
    'animal': 'animal'
}

df = pd.read_csv('challenge_label_mapping.csv')
df['supercategory'] = df['class_name'].map(CLASS_TO_CATEGORY)
df['supercategory_id'] = df['supercategory'].map(cat_to_id)
mapping_array = df.sort_values('label_key')['supercategory_id'].to_numpy(dtype=np.int64)

source_labels_dir = "small-goose/labels"
target_labels12_dir = "small-goose/labels12"

if not os.path.exists(source_labels_dir):
    print(f"Помилка: папка '{source_labels_dir}' не знайдена! Спершу підготуйте маски.")
else:
    print("Початок пакетного перекодування масок у small-goose/labels12...")
    
    # (train, val) split
    for split in os.listdir(source_labels_dir):
        split_path = os.path.join(source_labels_dir, split)
        if not os.path.isdir(split_path):
            continue
        
        # viewing scenarios
        for scenario in os.listdir(split_path):
            scenario_path = os.path.join(split_path, scenario)
            if not os.path.isdir(scenario_path):
                continue
            
            # creating labels12 for 12 class mask
            target_scenario_dir = os.path.join(target_labels12_dir, split, scenario)
            os.makedirs(target_scenario_dir, exist_ok=True)
            
            # processing each mask
            for file in os.listdir(scenario_path):
                if not file.endswith('_labelids.png'):
                    continue
                
                src_file_path = os.path.join(scenario_path, file)
                dst_file_path = os.path.join(target_scenario_dir, file)
                
                # loading masks
                raw_mask = np.array(Image.open(src_file_path))
                
                # vectorized encoding to 12 classes
                mapped_mask = mapping_array[raw_mask].astype(np.uint8)
                
                # saving results
                Image.fromarray(mapped_mask, mode='L').save(dst_file_path)

    print("Готово! Усі зменшені маски успішно перекодовані та збережені у папці 'small-goose/labels12'.")