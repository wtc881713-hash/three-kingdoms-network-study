from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Original novel path
input_file = ROOT / "data" / "raw" / "three_kingdoms.txt"

# Read the text
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

print("========== File Information ==========")
print(f"Characters: {len(text):,}")
print(f"Lines: {len(text.splitlines()):,}")
print()
print("First 500 characters:")
print("-" * 50)
print(text[:500])
print("-" * 50)