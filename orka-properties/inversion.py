import yaml
import os

def invert_alias_yaml(input_filename, output_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_filename)
    output_path = os.path.join(script_dir, output_filename)

    with open(input_path, 'r', encoding='utf-8') as f:
        original = yaml.safe_load(f)

    inverted = {}
    for canonical, aliases in original.items():
        for alias in aliases:
            normalized = alias.strip().lower()
            inverted[normalized] = canonical.strip().lower()

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(inverted, f, sort_keys=True, allow_unicode=True)

    print(f"Inverted YAML saved to {output_path} with {len(inverted)} entries.")

if __name__ == "__main__":
    invert_alias_yaml("colours.yaml", "inverted_colours.yaml")
