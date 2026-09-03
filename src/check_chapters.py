import re
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Path to the original novel
input_file = ROOT / "data" / "raw" / "three_kingdoms.txt"

# Read the text file
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Split the novel by chapter titles
chapters = re.split(r"(第[一二三四五六七八九十百零〇]+回.*)", text)

# Extract chapter titles
titles = chapters[1::2]

# Print summary
print("=" * 50)
print(f"Total chapters found: {len(titles)}")
print("=" * 50)

print("\nFirst 5 chapter titles:\n")

for title in titles[:5]:
    print(title.strip())