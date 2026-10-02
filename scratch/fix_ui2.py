import re

file_path = "scratch/astra_web/index.html"
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# Make all document.getElementById('.').textContent = ... safe
html = re.sub(
    r"document\.getElementById\('([^']+)'\)\.textContent\s*=\s*(.+?);",
    r"safeSetText('\1', \2);",
    html
)

# Make all document.getElementById('.').style.color = ... safe
html = re.sub(
    r"document\.getElementById\('([^']+)'\)\.style\.color\s*=\s*(.+?);",
    r"safeSetStyle('\1', 'color', \2);",
    html
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(html)
print("DOM references aggressively patched.")
