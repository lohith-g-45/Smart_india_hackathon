from pathlib import Path

dataset_root = Path(
    r"C:\Users\tuala\Downloads\Synchronised V abd S datasets"
)

print("=" * 80)
print("IO-VNBD FILE STRUCTURE")
print("=" * 80)

s_files = []
v_files = []

for path in dataset_root.rglob("*.csv"):

    name = path.name.strip()

    if name.lower().startswith("s-"):
        s_files.append(path)

    elif name.lower().startswith("v-"):
        v_files.append(path)


print("\nS FILES:", len(s_files))
print("=" * 80)

for path in sorted(s_files, key=lambda x: str(x).lower()):
    print(path.relative_to(dataset_root))


print("\nV FILES:", len(v_files))
print("=" * 80)

for path in sorted(v_files, key=lambda x: str(x).lower()):
    print(path.relative_to(dataset_root))


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)