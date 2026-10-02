import json

with open('colab/ASTRA_STGCN_training.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if 'source' in cell:
        for i, line in enumerate(cell['source']):
            # Replace v2 with v3 in paths, names, etc.
            if 'action_v2' in line:
                cell['source'][i] = line.replace('action_v2', 'action_v3')
            if 'synthetic_v2' in line:
                cell['source'][i] = line.replace('synthetic_v2', 'synthetic_v3')
            if 'ASTRA-v2' in line:
                cell['source'][i] = line.replace('ASTRA-v2', 'ASTRA-v3')
            if 'astra_dataset_v2.zip' in line:
                cell['source'][i] = line.replace('astra_dataset_v2.zip', 'astra_dataset_v3.zip')

            # Update the forward pass of STGCNPlusPlus
            if 'x = x.mean(dim=-1).mean(dim=-1)' in line:
                cell['source'][i] = (
                    "        # Spatial pooling\n"
                    "        x = x.mean(dim=-1)  # (N, C, T)\n"
                    "        # Temporal pooling over the final 15 positions\n"
                    "        pool_frames = min(15, x.size(-1))\n"
                    "        x = x[:, :, -pool_frames:].mean(dim=-1)  # (N, C)\n"
                )

with open('colab/ASTRA_STGCN_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
