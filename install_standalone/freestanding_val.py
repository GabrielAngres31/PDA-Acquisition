import training_utils
import src
from training_utils import validate_one_epoch as epval

# import modelsets/model from filepath

# Get expanded validation set from csv
# Generate dataloader
# calculate validation loss and output to console


mean_vloss_list = []
all_vloss_list = []

module.train()


vdataset = src.data.Dataset(validation_filepairs)
vloader  = src.data.create_dataloader(vdataset, batchsize, shuffle=False)

vloss_info = validate_one_epoch(module, vloader)
vloss, vloss_all = vloss_info["mean"], vloss_info["losses"]
    
