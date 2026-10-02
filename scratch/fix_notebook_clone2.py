import json

file_path = "colab/ASTRA_REAL_CAMERA_ADAPTATION.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "!unzip -q" in "".join(cell["source"]):
        source = "".join(cell["source"])
        new_source = source.replace(
            '''# Extract the codebase from Google Drive\n!unzip -q /content/drive/MyDrive/real_camera_astra/sih_codebase.zip -d /content/astra || echo "Already extracted"\n%cd /content/astra\n''',
            '''!git clone https://github.com/satyamrajawat-1/SIHPS2.git astra || echo "ASTRA already cloned"\n%cd astra\n'''
        )
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_source.splitlines()]

    if cell["cell_type"] == "markdown" and "2. Extract ASTRA" in "".join(cell["source"]):
        cell["source"] = ["## 2. Clone ASTRA & Install Dependencies\n"]

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook clone step updated back to git clone.")
