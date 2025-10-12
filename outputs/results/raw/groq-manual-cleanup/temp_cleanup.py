#!/usr/bin/env python3
import json
import re
import glob
import os

def extract_emotion(prompt: str) -> str:
    """Extracts the emotion from the user prompt (text between 'concept of' and '(which')."""
    match = re.search(r"concept of ([^\(]+)\s*\(which", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace(" ", "_").lower()
    return "unknown"

def process_jsonl_file(input_path: str):
    print(f"[INFO] Processing: {input_path}")
    modified_data = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            body = entry.get("body", {})
            model = body.get("model", "unknown-model").split("/")[-1]
            messages = body.get("messages", [])
            source = custom_id.split("_")[0] if "_" in custom_id else "unknown"

            # Extract emotion name from prompt
            emotion = "unknown"
            for msg in messages:
                if msg.get("role") == "user":
                    emotion = extract_emotion(msg.get("content", ""))
                    break

            # Rebuild new custom_id
            parts = custom_id.split("_")
            if len(parts) >= 3:
                id1, id2 = parts[-2], parts[-1]
                new_custom_id = f"{source}_{emotion}_{id1}_{id2}"
                entry["custom_id"] = new_custom_id
            else:
                new_custom_id = custom_id

            modified_data.append(entry)

    # Save modified input file
    out_input_path = f"results_groq_{model}-{source}-input.jsonl"
    with open(out_input_path, "w", encoding="utf-8") as outfile:
        for entry in modified_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved modified input: {out_input_path}")

    # Modify output file if present
    orig_output_path = input_path.replace("-input", "-output")
    if os.path.exists(orig_output_path):
        print(f"[INFO] Found output file: {orig_output_path}")
        output_modified = []
        with open(orig_output_path, "r", encoding="utf-8") as outfile_in:
            for line in outfile_in:
                if not line.strip():
                    continue
                entry = json.loads(line)
                old_id = entry.get("custom_id", "")
                for input_entry in modified_data:
                    if old_id.split("_")[-2:] == input_entry["custom_id"].split("_")[-2:]:
                        entry["custom_id"] = input_entry["custom_id"]
                        break
                output_modified.append(entry)

        out_output_path = f"results_groq_{model}-{source}.jsonl"
        with open(out_output_path, "w", encoding="utf-8") as outfile_out:
            for entry in output_modified:
                outfile_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Saved modified output: {out_output_path}")
    else:
        print(f"[WARN] No matching output file found for {input_path}")

def main():
    for file_path in glob.glob("*input*.jsonl"):
        process_jsonl_file(file_path)

if __name__ == "__main__":
    main()
