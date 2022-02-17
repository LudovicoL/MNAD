from cProfile import label
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
import argparse

import backbone as b
from config import *


parser = argparse.ArgumentParser(description="MNAD2")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=TEST_BATCH_SIZE, help='batch size for testing')
parser.add_argument('--h', type=int, default=PATCH_SIZE, help='height of input images')
parser.add_argument('--w', type=int, default=PATCH_SIZE, help='width of input images')
parser.add_argument('--c', type=int, default=CHANNEL, help='channel of input images')
parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=FRAME_SEQUENCE, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=FDIM, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=MDIM, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=MEMORY_SIZE, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=ALPHA, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=MEMORY_THRESHOLD, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=TEST_WORKERS, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='aitex', help='type of dataset: ped2, avenue, shanghai, aitex')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
args.dataset_type = args.dataset_type.upper()
test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

if len(list(b.folders_in(outputs_dir))) == 0:
    print('Run \'train.py\' first!')
    sys.exit(-1)
else:
    folders = sorted(list(b.folders_in(outputs_dir)))
    log_dir = folders[-1] + '/'


if os.path.isfile(log_dir + 'best_th_memory.pth'):
    args.th = torch.load(log_dir + 'best_th_memory.pth')

def show(img):
        npimg = img.numpy()
        plt.imshow(npimg, interpolation='nearest')

def main(log_file):
    # Set manual seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset_type == 'AITEX':
        test_dataset = b.AitexDataSet(aitex_test_dir, masks=True)
        number_of_images = len(test_dataset)
        channels = test_dataset[0][0].shape[0]
        original_height = test_dataset[0][0].shape[1]
        original_width = test_dataset[0][0].shape[2]
        test_dataset, test_widths, test_heights = b.resizeAitex(test_dataset, original_width, original_height, True)
        test_patches, mask_test_patches = b.DivideInPatches(test_dataset, test_folder, PATCH_SIZE, STRIDE, masks=True, save_image_patch=True)
        test_patches = torch.cat(test_patches, dim=0).reshape(-1, channels, PATCH_SIZE, PATCH_SIZE)
        mask_test_patches = torch.cat(mask_test_patches, dim=0).reshape(-1, channels, PATCH_SIZE, PATCH_SIZE)
        b.countAnomalies(test_patches, mask_test_patches, log_dir, save=False)
        test_number_of_patches_for_image = b.calculateNumberPatches(test_widths, test_heights, PATCH_SIZE)

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size = args.batch_size,
                                shuffle=False, num_workers=args.num_workers, drop_last=False)



    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(log_dir + 'model.pth')
    model.cuda()
    m_items = torch.load(log_dir + 'keys.pt')
    labels = np.load(log_dir + 'frame_labels.npy')

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    test_dim = 0
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
        test_dim += len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    b.myPrint('Evaluation of ' + str(args.dataset_type), log_file)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if args.method == 'pred':
            labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        else:
            labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []


    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

    model.eval()

    norm = 0
    anorm = 0
    all_recon_output = []

    count_img = 1
    for k,(imgs, name) in enumerate(tqdm(test_batch)):
    
        if args.method == 'pred':
            if k == label_length-4*(video_num+1):
                video_num += 1

                label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        else:
            if k == label_length:
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']



        # show(grid_img)
        imgs = Variable(imgs).cuda()

        
        if args.method == 'pred':
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)


            mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()  # element wise mean square error
            mse_feas = compactness_loss.item()

            # Calculating the threshold for updating at the test time
            point_sc = point_score(outputs, imgs[:,3*4:])
        
        else:
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)

            mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
            mse_feas = compactness_loss.item()
            
            all_recon_output.append(outputs[0, 0, :].detach().cpu())
            
            count_img += 1

            point_sc = point_score(outputs, imgs)


        if point_sc < args.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test = model.memory.update(query, m_items_test, False)
            norm += 1
        else:
            anorm += 1

        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))

        feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)


    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                        anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
    b.myPrint('The result of ' + str(args.dataset_type), log_file)
    b.myPrint('AUC: ' + str(accuracy*100) + '%', log_file)

     
    
    if args.dataset_type == 'AITEX':
        best_th_anomaly = torch.load(log_dir + 'best_th_anomaly.pth')
        classification = np.where(anomaly_score_total_list < best_th_anomaly, True, False)   

        # Point out the anomalies patches
        j = 0
        for i in range(number_of_images):
            # img = all_recon_output[j:j+test_number_of_patches_for_image[i]]
            # img = torch.stack(img)
            # img = img.unsqueeze(1)
            # print(img.shape)
            # anomaly_score = anomaly_score_total_list[j:j+test_number_of_patches_for_image[i]]
            # threshold = anomaly_score > args.th
            # idx = []
            # for n, e in enumerate(threshold):
            #     if e == False:
            #         idx.append(n)
            # b_mask = torch.zeros(test_number_of_patches_for_image[i], 1, PATCH_SIZE, PATCH_SIZE)
            # p1 = torch.ones(1, PATCH_SIZE, PATCH_SIZE)
            # for id in idx:
            #     b_mask[id] = p1
            # out = b.AssemblePatches(b_mask, 1, 1, test_heights[i], test_widths[i], PATCH_SIZE, STRIDE).__getitem__(0)
            # recon_image = b.AssemblePatches(img, 1, 1, test_heights[i], test_widths[i], PATCH_SIZE, STRIDE).__getitem__(0)
            # out = out.numpy()
            # recon_image = recon_image.numpy()
            # norm_mask = normalize(out)
            # norm_image = normalize(recon_image)
            # contours, _ = find_contours(norm_mask)
            # norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
            # for c in contours:
            #     cv2.drawContours(norm_image, [c], 0, (255, 0, 0), 5)
            # cv2.imwrite(b.assemble_pathname("IMG_Test" + str(i) + ".png", log_dir), norm_image)

            test_patches_with_colors = []
            for k in range(test_number_of_patches_for_image[i]):
                test_patches_with_colors.append(test_patches[j+k].repeat(3, 1, 1))
                if classification[j+k] == True:
                    test_patches_with_colors[k][0, :1, :] = 0.0000
                    test_patches_with_colors[k][0, :, :1] = 0.0000
                    test_patches_with_colors[k][0, -1:, :] = 0.0000
                    test_patches_with_colors[k][0, :, -1:] = 0.0000
                    
                    test_patches_with_colors[k][1, :1, :] = 0.6353
                    test_patches_with_colors[k][1, :, :1] = 0.6353
                    test_patches_with_colors[k][1, -1:, :] = 0.6353
                    test_patches_with_colors[k][1, :, -1:] = 0.6353
                    
                    test_patches_with_colors[k][2, :1, :] = 0.9098
                    test_patches_with_colors[k][2, :, :1] = 0.9098
                    test_patches_with_colors[k][2, -1:, :] = 0.9098
                    test_patches_with_colors[k][2, :, -1:] = 0.9098

            test_patches_with_colors = torch.stack(test_patches_with_colors)

            tensor_reconstructed = b.AssemblePatches(test_patches_with_colors[:test_number_of_patches_for_image[i]], 1, 3, test_heights[i], test_widths[i], PATCH_SIZE, STRIDE).__getitem__(0)
            torchvision.utils.save_image(tensor_reconstructed, b.assemble_pathname('Test_image'+str(i), log_dir))
            j += test_number_of_patches_for_image[i]
        
        TP = []
        TN = []
        FP = []
        FN = []

        print(labels)
        print(classification)

        for i in range(test_size):
            if labels[0][i] == True and classification[i] == True:
                    TP.append(i)
            elif labels[0][i] == False and classification[i] == True:
                    FP.append(i)
            elif labels[0][i] == False and classification[i] == False:
                TN.append(i)
            elif labels[0][i] == True and classification[i] == False:
                FN.append(i)

        print(len(TP), len(FP), len(TN), len(FN))
        precision = b.precision(len(TP), len(FP))
        sensitivity = b.sensitivity(len(TP), len(FN))
        fpr = b.FPR(len(FP), len(TN))
        f1_score = b.F1_score (precision, sensitivity)

        b.myPrint("Precision: " + str(precision), log_file)
        b.myPrint("Sensitivity: " + str(sensitivity), log_file)
        b.myPrint("False Positive Rate : " + str(fpr), log_file)
        b.myPrint("F1-Score : " + str(f1_score), log_file)

    




if __name__ == '__main__':
    log_file = open(log_dir + "log.txt", "a")
    main(log_file)
    if Telegram_messages: b.telegram_bot_sendtext("MNAD2: Testing finished.")
    log_file.close()
