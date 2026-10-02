import json

file_path = "colab/ASTRA_REAL_CAMERA_ADAPTATION.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 4 to inject session_id extraction
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "out_json = EXTRACT_DIR /" in "".join(cell["source"]):
        source = "".join(cell["source"])
        new_source = source.replace(
            'out_json = EXTRACT_DIR / f"{f.stem}_poses.json"\n        with open(out_json, "w") as jf:\n            json.dump({"action": action, "clip_id": f.stem, "poses": pose_sequence}, jf)',
            '''# Read session_id from metadata if available
        session_id = f.stem
        meta_file = f.with_suffix(".json")
        if meta_file.exists():
            with open(meta_file, "r") as mf:
                try:
                    session_id = json.load(mf).get("session_id", f.stem)
                except: pass
                
        out_json = EXTRACT_DIR / f"{f.stem}_poses.json"
        with open(out_json, "w") as jf:
            json.dump({"action": action, "clip_id": f.stem, "session_id": session_id, "poses": pose_sequence}, jf)'''
        )
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_source.splitlines()]
        
# Find cell 5 to use GroupShuffleSplit and print the groups
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "from sklearn.model_selection import train_test_split" in "".join(cell["source"]):
        source = "".join(cell["source"])
        new_source = source.replace(
            "from sklearn.model_selection import train_test_split",
            "from sklearn.model_selection import GroupShuffleSplit"
        )
        
        new_source = new_source.replace(
            'data_list.append({"clip_id": d["clip_id"], "poses": d["poses"], "label": ACTION2ID[d["action"]]})',
            'data_list.append({"clip_id": d["clip_id"], "session_id": d.get("session_id", d["clip_id"]), "poses": d["poses"], "label": ACTION2ID[d["action"]]})'
        )
        
        # Replace the train_test_split logic with GroupShuffleSplit
        old_split = """# SPLIT BY CLIP/SESSION (70/15/15)
train_data, temp_data = train_test_split(data_list, test_size=0.3, stratify=[d["label"] for d in data_list], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=[d["label"] for d in temp_data], random_state=42)"""

        new_split = """# SPLIT BY SESSION (GroupShuffleSplit to ensure no overlap)
import numpy as np
groups = [d["session_id"] for d in data_list]

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss1.split(data_list, groups=groups))
train_data = [data_list[i] for i in train_idx]
temp_data = [data_list[i] for i in temp_idx]
temp_groups = [groups[i] for i in temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_data, groups=temp_groups))
val_data = [temp_data[i] for i in val_idx]
test_data = [temp_data[i] for i in test_idx]

train_groups = set([d["session_id"] for d in train_data])
val_groups = set([d["session_id"] for d in val_data])
test_groups = set([d["session_id"] for d in test_data])

print("\\n--- SPLIT VERIFICATION ---")
print(f"Train groups: {list(train_groups)}")
print(f"Validation groups: {list(val_groups)}")
print(f"Test groups: {list(test_groups)}")

overlap_val = train_groups.intersection(val_groups)
overlap_test = train_groups.intersection(test_groups)
overlap_val_test = val_groups.intersection(test_groups)
assert len(overlap_val) == 0, f"Overlap in Train/Val: {overlap_val}"
assert len(overlap_test) == 0, f"Overlap in Train/Test: {overlap_test}"
assert len(overlap_val_test) == 0, f"Overlap in Val/Test: {overlap_val_test}"
print("Zero overlap verified.")
"""
        new_source = new_source.replace(old_split, new_split)
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_source.splitlines()]

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook updated.")
