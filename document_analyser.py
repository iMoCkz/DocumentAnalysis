import os
from pathlib import Path

root = 'C:/Users/Maxe/PycharmProjects/DocumentAnalysis/docs'
# get all subfolders from root
year_folders = [f.path for f in os.scandir(root) if f.is_dir()]
#process every folder (for every year)
for year_folder in year_folders:
    # get file paths in folder
    file_paths = []
    for file in Path(year_folder).rglob("*.txt"):
        file_paths.append(file.parent / file.name)
    # count of files for each year
    file_cnt = len(file_paths)
    # read every file's text as string
    docs_per_year = []
    for txt_file in file_paths:
        with open(txt_file) as f:
            txt_file_as_string = f.read()
        docs_per_year.append(txt_file_as_string)