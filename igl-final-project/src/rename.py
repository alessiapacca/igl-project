from os import listdir, rename
from os.path import isfile, join

path = "../data/smoothed"
for f in listdir(path):
	if isfile(join(path, f)) and f.endswith('.landmarks'):
		rename(join(path, f), join(path, f.split('.')[0]+'.txt'))