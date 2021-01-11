import os

path1 = "trained_models"
path2 = "build_data"
path3 = "LAMBADA"
path4 = "original_data"
path5 = "saves"

try:
    os.mkdir(path1)
except: 
	pass
try:
    os.mkdir(path2)
except: 
	pass
try:
    os.mkdir(path3)
except: 
	pass
try:
    os.mkdir(path4)
except: 
	pass
try:
    os.mkdir(path5)
except: 
	pass