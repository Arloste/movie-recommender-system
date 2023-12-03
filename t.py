import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

file = f"{os.getcwd()}/data/interim/u.user.csv"
df = pd.read_csv(file)

#df.drop(["Unnamed: 0"], axis=1, inplace=True)
#df.set_index("user_id", inplace=True)

print(df.head())
#df.job_id /= 20
#df.to_csv(file)
#print(df.tail(50))


"""
filename = "u.item"
src_path = f"{os.getcwd()}/data/raw/{filename}"
dest_path = f"{os.getcwd()}/data/interim/{filename}_names.csv"

with open(src_path, 'r', encoding='latin-1') as f:
    data = f.readlines()

df = pd.DataFrame()
index = list()
names = list()

for line in data:
    line = line[:-1].split('|')
    index.append(line[0])
    names.append(line[1])

df["item_id"] = index
df["item_name"] = names

df.set_index("item_id", inplace=True)
print(df)
print(dest_path)
df.to_csv(dest_path)
"""
