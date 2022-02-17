import torch
import numpy as np
    
########################################################
# Control variables

# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputs_dir = './outputs/'
plot_extension = '.png'
Telegram_messages = False
seed = 0

########################################################
# Datasets
# AITEX
aitex_folder = './dataset/AITEX'
aitex_train_dir = aitex_folder + '/trainset/'
aitex_validation_dir = aitex_folder + '/validationset/'
aitex_test_dir = aitex_folder + '/testset/'
aitex_mask_dir = aitex_folder + '/Mask_images/'
CUT_PATCHES = 3

########################################################
# Network parameters

EPOCHS = 60                 # number of epochs for training

MEMORY_SIZE = 20            # number of the memory items
CHANNEL = 3                 # channel of input images
FDIM = 512                  # channel dimension of the features
MDIM = 512                  # channel dimension of the memory items

TRAIN_WORKERS = 2           # number of workers for the train loader
VAL_WORKERS = 1             # number of workers for the validation loader
TEST_WORKERS = 1            # number of workers for the test loader

PATCH_SIZE = 64             # patch size
STRIDE = PATCH_SIZE         # stride of patch

BATCH_SIZE = 128            # batch size
VAL_BATCH_SIZE = 1          # batch size for validation
TEST_BATCH_SIZE = 1         # batch size for testing

ALPHA = 0.7                 # weight for the anomality score
MEMORY_THRESHOLD = 0.01      # threshold for test updating

LEARNING_RATE = 2e-4        # initial learning rate

W_COMPACT_LOSS = 0.01       # weight of the feature compactness loss
W_SEPARATE_LOSS = 0.01      # weight of the feature separateness loss

FRAME_SEQUENCE = 1          # length of the frame sequences

ANOMALY_THRESHOLD = 2       # threshold to consider a patch as anomalous

# list of possible thresholds to update memory items:
th_memory = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# np array of possible anomaly score threshold:
th_anomaly_score_array = np.arange(0.1, 1, 0.05)

