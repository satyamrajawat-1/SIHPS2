import json

with open('colab/ASTRA_STGCN_training.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. Modify the training config cell (Part 5) to use WeightedRandomSampler
# Find Part 5 cell
part5_idx = None
for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'code' and 'EPOCHS = 80' in cell['source'][1]:
        part5_idx = i
        break

if part5_idx is not None:
    source = data['cells'][part5_idx]['source']
    new_source = []
    for line in source:
        if line.startswith('train_loader = DataLoader'):
            # Replace train_loader with sampler logic
            new_source.extend([
                "# Calculate class weights for WeightedRandomSampler\n",
                "train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]\n",
                "class_counts = np.bincount(train_labels)\n",
                "class_weights = 1.0 / np.maximum(class_counts, 1)  # avoid div by zero\n",
                "sample_weights = [class_weights[label] for label in train_labels]\n",
                "sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)\n",
                "\n",
                "print(f\"Class counts in train: {class_counts}\")\n",
                "print(f\"Class weights: {[round(w, 4) for w in class_weights]}\")\n",
                "\n",
                "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, drop_last=True, pin_memory=True)\n"
            ])
        elif 'train_loader' in line and 'shuffle=True' in line:
             pass # should be caught by above
        else:
            new_source.append(line)
    
    data['cells'][part5_idx]['source'] = new_source

    # 2. Add sanity check cell after Part 5
    sanity_check_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Sanity check: verify balanced sampling\n",
            "print(\"Running sanity check on one batch...\")\n",
            "batch_x, batch_y = next(iter(train_loader))\n",
            "batch_counts = np.bincount(batch_y.numpy(), minlength=NUM_CLASSES)\n",
            "print(f\"Batch size: {batch_x.size(0)}\")\n",
            "print(f\"Class counts in batch: {batch_counts}\")\n",
            "for i, count in enumerate(batch_counts):\n",
            "    print(f\"  {CLASS_NAMES[i]:7s}: {count:2d} ({(count/batch_x.size(0))*100:.1f}%)\")\n",
            "print(\"Sampling is balanced if distribution is roughly equal (~16.6% each).\")"
        ]
    }
    data['cells'].insert(part5_idx + 1, sanity_check_cell)

with open('colab/ASTRA_STGCN_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
