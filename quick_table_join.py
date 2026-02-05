# quick_table_join


import pandas as pd
import os
import glob
import re



# sourcefiles_path = "inference/clump_data"

# file_list = [f.split(".")[0] for f in os.listdir(sourcefiles_path)]
# print(file_list)



# concat_result = pd.read_csv(os.path.join(sourcefiles_path + f"/cot1.csv"))
# #concat_result = concat_result.reset_index()
# concat_result = concat_result.assign(ID = 'cot1')
# #print(concat_result.head())
# for file in ['cot2', 'cot3', 'cot4', 'cot5', 'cot6', 
#              'cotE01', 'cotE02', 'cotE03', 'cotE04', 'cotE05', 'cotE06', 
#              'cotE07', 'cotE08', 'cotE09', 'cotE10', 'cotE11', 'cotE12', 
#              'cotE13', 'cotE14', 
#              'trm678_5_COT_02',
#              'basl-2_5_COT_02', 
#              'basl-2_5_COT_03', 
#              'basl-2_5_COT_04']:
#     add_file = pd.read_csv(os.path.join(sourcefiles_path + f"/{file}.csv"))
#     add_file = add_file.assign(ID = file)
#     concat_result = pd.concat([concat_result, add_file])



# pd.DataFrame.to_csv(concat_result, f"{sourcefiles_path}/aggregate_reorder.csv")
# print(concat_result.shape[0])


sourcefiles_path = "only_pored/AZD_test/inference_jan_2025/"

AZD_files_100nMAZD = [
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_1_1_Merged.tif_AZD_jan2025_100nMAZD_1_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_2_1_Merged.tif_AZD_jan2025_100nMAZD_2_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_3_1_Merged.tif_AZD_jan2025_100nMAZD_3_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_4_1_Merged.tif_AZD_jan2025_100nMAZD_4_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_5_1_Merged.tif_AZD_jan2025_100nMAZD_5_1.output.csv'
]

AZD_files_1uMAZD = [ 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_1_1_Merged.tif_AZD_jan2025_1uMAZD_1_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_2_1_Merged.tif_AZD_jan2025_1uMAZD_2_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_3_1_Merged.tif_AZD_jan2025_1uMAZD_3_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_4_1_Merged.tif_AZD_jan2025_1uMAZD_4_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_5_1_Merged.tif_AZD_jan2025_1uMAZD_5_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_6_1_Merged.tif_AZD_jan2025_1uMAZD_6_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_7_1_Merged.tif_AZD_jan2025_1uMAZD_7_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_8_1_Merged.tif_AZD_jan2025_1uMAZD_8_1.output.csv'
]
AZD_files_250nMAZD = [
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_1_1_Merged.tif_AZD_jan2025_250nMAZD_1_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_2_1_Merged.tif_AZD_jan2025_250nMAZD_2_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_3_1_Merged.tif_AZD_jan2025_250nMAZD_3_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_4_1_Merged.tif_AZD_jan2025_250nMAZD_4_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_5_1_Merged.tif_AZD_jan2025_250nMAZD_5_1.output.csv'
]
AZD_files_DMSO = [
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_1_1_Merged.tif_AZD_jan2025_DMSO_1_1.output.csv',
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_2_1_Merged.tif_AZD_jan2025_DMSO_2_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_3_1_Merged.tif_AZD_jan2025_DMSO_3_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_4_1_Merged.tif_AZD_jan2025_DMSO_4_1.output.csv', 
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_5_1_Merged.tif_AZD_jan2025_DMSO_5_1.output.csv'
]

AZD_files_list = [AZD_files_DMSO, AZD_files_100nMAZD, AZD_files_250nMAZD, AZD_files_1uMAZD]


filepaths = [filepath for filepath in glob.glob(sourcefiles_path + "*.csv")]
ID_search = re.compile(r'jan2025_.+_\d_\d')
IDs = [ID_search.search(filepath).group() for filepath in filepaths]

# jan2025_1uMAZD_5_1


files = [pd.read_csv(filepath) for filepath in filepaths]
treat_to_group = {"1uMAZD":1, "250nMAZD":2, "100nMAZD":3, "DMSO":4}

# for i in range(len(filepaths)):
#     files[i]['Conc'] = IDs[i][8:-4]
#     files[i]['ID'] = IDs[i][-3:]
#     files[i]["Full_ID"] = IDs[i][8:]
#     files[i]["Group"] = treat_to_group[IDs[i][8:-4]]    

# df_concat = pd.concat([df for df in files], ignore_index=True)

# df_concat.to_csv("only_pored/AZD_test/concatenated.csv")

# print(IDs)
AZD_label_list = ["DMSO", "100nM_AZD", "250nM_AZD", "1uM_AZD"]

for i, file_list in enumerate(AZD_files_list):
    actual_file = []
    for j, file in enumerate(file_list):
        make = pd.read_csv(file)
        make['Conc'] = AZD_label_list[i]
        make['Label'] = f"{i}_1"
        make["Full_ID"] = f"{AZD_label_list[i]}_{i}_1"
    actual_file.append(make)

    # df_concat = pd.concat([pd.read_csv(file) for file in file_list])
    df_concat = pd.concat([file for file in actual_file])
    df_concat.to_csv(f"only_pored/AZD_test/concat_{AZD_label_list[i]}.csv")
    # print(file_list)

    # df_concat_AZDs = pd.concat([pd.read_csv(key) for key, value in file.iteritems()])
    # print(df_concat_AZDs)


