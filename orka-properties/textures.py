import os
import yaml

def extract_textures_from_dtd(file_path):
    textures = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                labels = parts[1:]
                textures.update(label.strip() for label in labels if label.strip())
    return {label: label for label in sorted(textures)}

def save_to_yaml(data, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=True, allow_unicode=True)
    print(f"Saved {len(data)} texture entries to {full_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dtd_path = os.path.join(script_dir, 'source', 'dtd.txt')

    if not os.path.exists(dtd_path):
        raise FileNotFoundError(f"No file found at {dtd_path}")

    texture_dict = extract_textures_from_dtd(dtd_path)
    save_to_yaml(texture_dict, "textures.yaml")
