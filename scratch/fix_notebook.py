import json

with open('colab/ASTRA_STGCN_training.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

source = data['cells'][5]['source']
for i, line in enumerate(source):
    if '!unzip' in line:
        source[i] = line.replace('{fname}', '"{fname}"')

with open('colab/ASTRA_STGCN_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
