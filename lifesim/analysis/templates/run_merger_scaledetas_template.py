import shutil
from pathlib import Path

merge_folder = Path('$merge_folder')
catalogs     = $catalogs
basepath     = Path('$basepath')
run_name     = '$run_name'
today        = '$today'

for full_name, short_name in catalogs:
    source_basepath = basepath / f'{today}_{short_name}' / 'output' / run_name
    destination_basepath = merge_folder / f'{today}_{short_name}' / 'output' / 'ap_merged'
    destination_basepath.mkdir(parents=True, exist_ok=True)
    for option_folder in (f for f in source_basepath.iterdir() if f.is_dir()):
        dest_folder = destination_basepath / option_folder.name
        dest_folder.mkdir(exist_ok=True)
        for f in option_folder.iterdir():
            if f.suffix == '.hdf5' and 'maxsep' not in f.name:
                shutil.copy2(str(f), str(dest_folder / f.name))

with open(merge_folder / 'config_files' / 'optimizer_jobs.csv', 'w') as f:
    for full_name, short_name in catalogs:
        f.write(f'{short_name},{merge_folder}/{today}_{short_name}/output/\n')