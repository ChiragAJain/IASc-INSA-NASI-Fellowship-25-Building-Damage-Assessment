import os
import json
from PIL import Image
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

IMAGE_DIR = "./holdout/images"
LABEL_DIR = "./holdout/labels"
OUTPUT_DIR = "postdisaster-patches_holdout"
CROP_SIZE = 128
DAMAGE_CLASSES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

for damage_class in DAMAGE_CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, damage_class), exist_ok=True)

def parse_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_name = data["metadata"]["img_name"]
    buildings = []
    for i, feat in enumerate(data["features"]["xy"]):
        dmg = feat["properties"].get("subtype", "un-classified")
        if dmg in DAMAGE_CLASSES:
            wkt = feat["wkt"]
            buildings.append((img_name, dmg, wkt, i))
    return buildings

def wkt_to_clipped_box(wkt_string, crop_size, img_width, img_height):
    polygon = wkt_loads(wkt_string)
    minx, miny, maxx, maxy = polygon.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    half = crop_size // 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(img_width, int(cx + half))
    y2 = min(img_height, int(cy + half))
    return x1, y1, x2, y2

def process_patch(task):
    img_name, label, wkt, count = task
    try:
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        box = wkt_to_clipped_box(wkt, CROP_SIZE, img.width, img.height)
        patch = img.crop(box)
        out_name = f"{img_name[:-4]}_{count}.png"
        patch.save(os.path.join(OUTPUT_DIR, label, out_name))
        return 1
    except Exception as e:
        print(f"Failed on {img_name}: {e}")
        return 0

def main():
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith("post_disaster.json")]
    print(f"Found {len(label_files)} post-disaster annotations.")
    
    tasks = []
    for lbl_file in tqdm(label_files, desc="🔍 Parsing JSON"):
        buildings = parse_json(os.path.join(LABEL_DIR, lbl_file))
        tasks.extend(buildings)

    print(f"Total building patches to process: {len(tasks)}")
    total_saved = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_patch, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing patches"):
            total_saved += future.result()

    print(f"\nTotal patches saved: {total_saved}")

if __name__ == "__main__":
    main()
