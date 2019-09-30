

file_path = "../_original_data/cora.txt"
with open(file_path, "r") as f:
    line = f.readline()
    items = line.split("	")
    print("len", len(items))
    print(items)

from original.load import load_data
outs = load_data("cora")
features = outs[1]