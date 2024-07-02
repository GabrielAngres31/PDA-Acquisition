# quick_table_join


import pandas as pd
import os




sourcefiles_path = "inference/clump_data"

file_list = [f.split(".")[0] for f in os.listdir(sourcefiles_path)]
print(file_list)



concat_result = pd.read_csv(os.path.join(sourcefiles_path + f"/{file_list[0]}.csv"))
#concat_result = concat_result.reset_index()
concat_result = concat_result.assign(ID = file_list[0])
#print(concat_result.head())
for file in file_list[1:]:
    add_file = pd.read_csv(os.path.join(sourcefiles_path + f"/{file}.csv"))
    add_file = add_file.assign(ID = file)
    concat_result = pd.concat([concat_result, add_file])

pd.DataFrame.to_csv(concat_result, f"{sourcefiles_path}/aggregate.csv")
print(concat_result.shape[0])