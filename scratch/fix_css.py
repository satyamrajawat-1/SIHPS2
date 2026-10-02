import re

file_path = "scratch/astra_web/index.html"
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# Fix camera-container CSS
old_css = """  .camera-container {
    position: relative;
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    background: #000;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    display: flex;
    justify-content: center;
    align-items: center;
    aspect-ratio: 16/9;
  }"""

new_css = """  .camera-container {
    position: relative;
    width: 100%;
    max-width: 1400px;
    height: calc(100vh - 300px); /* Leave room for bottom bar */
    margin: 0 auto;
    background: #000;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    display: flex;
    justify-content: center;
    align-items: center;
  }"""

html = html.replace(old_css, new_css)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(html)
print("CSS fixed.")
