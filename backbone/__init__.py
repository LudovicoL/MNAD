from .AITEX import AitexDataSet, resizeAitex, checkAitex
from .patches import *
from .SSIM import calculate_ssim, plot_ssim_histogram
from config import *
from .metrics import *

import requests
import json
import os
from scipy.ndimage.filters import gaussian_filter
import torch
import numpy as np

def myPrint(string, filename):
    print(string)
    filename.write(string + '\n')

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)

def telegram_bot_sendtext(bot_message):
    """
    Send a notice to a Telegram chat. To use, create a file "tg.ll" in the main folder with this form:
    {
    "token": "",    <-- bot token 
    "idchat": ""    <-- your chat id
    }
    """
    try:
        with open('./tg.ll') as f:
            data = json.load(f)
    except:
        print("ERROR: Can't send message on Telegram. Configure the \'./tg.ll\' file or set Telegram_messages=False.")
        return
    bot_token = data['token']
    bot_chatID = data['idchat']
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return str(response)

def add_noise(inputs, noise_factor=0.3):
    """
    source: https://ichi.pro/it/denoising-autoencoder-in-pytorch-sul-set-di-dati-mnist-184080287458686
    """
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy

def augmentationDataset(dataset):
    ds = []
    widths = []
    heights = []
    for i in range(len(dataset)):
        j = dataset.__getitem__(i)

        ds.append(j)                                                    # Original image
        widths.append(j.shape[2])
        heights.append(j.shape[1])

        noised_image = add_noise(j, noise_factor=0.05)                # Gaussian noise
        ds.append(noised_image)
        widths.append(noised_image.shape[2])
        heights.append(noised_image.shape[1])

        j = np.transpose(j.numpy(), (1, 2, 0))
        ds.append(torch.tensor(np.fliplr(j).copy()).permute(2, 0, 1))   # orizontal flip
        widths.append(j.shape[1])
        heights.append(j.shape[0])

        ds.append(torch.tensor(np.flipud(j).copy()).permute(2, 0, 1))   # vertical flip
        widths.append(j.shape[1])
        heights.append(j.shape[0])

        blurred = gaussian_filter(j, sigma=0.5)                         # blur
        ds.append(torch.tensor(blurred).permute(2, 0, 1))
        widths.append(j.shape[1])
        heights.append(j.shape[0])
        
    return ds, widths, heights


def assemble_pathname(filename, outputs_dir):
    os.makedirs(outputs_dir + '/images/', exist_ok=True)
    return outputs_dir + '/images/' + filename + plot_extension