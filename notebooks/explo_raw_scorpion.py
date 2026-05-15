# %%
from pathlib import Path
from PIL import Image

# %%
raw_data_dir = Path("/home/valentin/workspaces/vfm-geom-xai/data/raw/SCORPION_dataset") 
files = [f for f in raw_data_dir.rglob("*") if f.is_file()]
# %%
print(f"Found {len(files)} files in {raw_data_dir}")
# %%
extensions = set(f.suffix for f in files)
print(f"File extensions found: {extensions}")
# %%
names = set(f.stem for f in files)
print(f"File names found: {names}")

# %%
print(files[:10])

# %%
files[0].parents[0].name

# %%
files[0].parents[1].name

# %%

files[0].parents[2].name
# %%
files[0].stem

# %%
image = Image.open(files[0])
image_size = image.size

# %%
image_size[0] // 224


# %%
