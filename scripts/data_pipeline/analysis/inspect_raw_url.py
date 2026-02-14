from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO

def inspect():
    print("Loading dataset stream...")
    ds = load_dataset("allenai/pixmo-points", split="train", streaming=True)
    
    iterator = iter(ds)
    
    print("\n--- Inspecting First Valid Example ---")
    for i in range(10): # Try up to 10 to find one with a working URL
        try:
            item = next(iterator)
            url = item['image_url']
            points = item['points']
            print(f"Checking URL: {url}")
            
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content))
                width, height = img.size
                print(f"Image downloaded. Size: {width}x{height}")
                print(f"Points sample (Raw): {points[:2]}")
                print(f"Label: {item['label']}")
                
                # Check normalization
                p1 = points[0]
                is_normalized = (p1[0] <= 1.0 and p1[1] <= 1.0) and (max([p[0] for p in points]) <= 1.0)
                print(f"Are points normalized (0-1)? {is_normalized}")
                
                # Simulate formatting
                formatted_points = []
                for idx, (x, y) in enumerate(points, 1):
                    # Normalize to 0-100 if they are absolute pixels
                    if not is_normalized:
                        nx = (x / width) * 100
                        ny = (y / height) * 100
                    else:
                        nx = x * 100
                        ny = y * 100
                    formatted_points.append(f'x{idx}="{nx:.1f}" y{idx}="{ny:.1f}"')
                
                pts_str = " ".join(formatted_points)
                label = item.get('label', 'object')
                final_str = f'<points {pts_str} alt={label}>{label}</points>'
                print(f"\nGenerated Answer String (Preview): {final_str[:200]}...")
                break
            else:
                print(f"Failed to download: {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
