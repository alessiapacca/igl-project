from os import listdir
from os.path import isfile, join

path = "../data/smoothed"
files = [f.split('.')[0] for f in listdir(path) if isfile(join(path, f)) and f.endswith('.obj')]

with open(join(path, "smoothed_mesh_list"), "w") as outfile:
    outfile.write("\n".join(str(item) for item in files))

