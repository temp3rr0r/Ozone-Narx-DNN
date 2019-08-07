import sys
import pandas as pd
import os

print ("Usage: minMseReader.py -f <filename>", sys.argv[0])
if len(sys.argv) == 3:
	if str(sys.argv[1]) == "-f":
		if os.path.exists(str(sys.argv[2])):
			min_mse = pd.read_pickle(str(sys.argv[2]))['min_mse'][0]
			print("Previous min_mse: {}".format(min_mse))