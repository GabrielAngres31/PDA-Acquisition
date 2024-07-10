# quick_table_join


import pandas as pd
import os




sourcefiles_path = "inference/clump_data"

file_list = [f.split(".")[0] for f in os.listdir(sourcefiles_path)]
print(file_list)



concat_result = pd.read_csv(os.path.join(sourcefiles_path + f"/cot1.csv"))
#concat_result = concat_result.reset_index()
concat_result = concat_result.assign(ID = 'cot1')
#print(concat_result.head())
for file in ['cot2', 'cot3', 'cot4', 'cot5', 'cot6', 
             'cotE01', 'cotE02', 'cotE03', 'cotE04', 'cotE05', 'cotE06', 
             'cotE07', 'cotE08', 'cotE09', 'cotE10', 'cotE11', 'cotE12', 
             'cotE13', 'cotE14', 
             'trm678_5_COT_02',
             'basl-2_5_COT_02', 
             'basl-2_5_COT_03', 
             'basl-2_5_COT_04']:
    add_file = pd.read_csv(os.path.join(sourcefiles_path + f"/{file}.csv"))
    add_file = add_file.assign(ID = file)
    concat_result = pd.concat([concat_result, add_file])



pd.DataFrame.to_csv(concat_result, f"{sourcefiles_path}/aggregate_reorder.csv")
print(concat_result.shape[0])