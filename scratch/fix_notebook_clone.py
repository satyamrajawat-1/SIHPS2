import json

file_path = "colab/ASTRA_REAL_CAMERA_ADAPTATION.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "!git clone" in "".join(cell["source"]):
        source = "".join(cell["source"])
        new_source = source.replace(
            "!git clone https://github.com/google-deepmind/astra.git || echo \"ASTRA already cloned\"\n%cd astra\n",
            '''# Extract the codebase from Google Drive
!unzip -q /content/drive/MyDrive/real_camera_astra/sih_codebase.zip -d /content/astra || echo "Already extracted"
%cd /content/astra
'''
        )
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_source.splitlines()]

    if cell["cell_type"] == "markdown" and "Clone ASTRA" in "".join(cell["source"]):
        cell["source"] = ["## 2. Extract ASTRA & Install Dependencies\n", "Ensure you have zipped your entire local `SIH` folder into `sih_codebase.zip` and uploaded it to your `real_camera_astra` folder in Drive."]

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook clone step updated.")
