import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
from model.utils import DataLoader
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm import tqdm
import argparse
import shutil
import glob
import time

import backbone as b
from config import *
from datetime import datetime


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size for training')
parser.add_argument('--validation_batch_size', type=int, default=VAL_BATCH_SIZE, help='batch size for validation')
parser.add_argument("-e", '--epochs', type=int, default=EPOCHS, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=W_COMPACT_LOSS, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=W_SEPARATE_LOSS, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=PATCH_SIZE, help='height of input images')
parser.add_argument('--w', type=int, default=PATCH_SIZE, help='width of input images')
parser.add_argument('--c', type=int, default=CHANNEL, help='channel of input images')
parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='initial learning rate')
parser.add_argument('--method', type=str, default='recon', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=FRAME_SEQUENCE, help='Length of the frame sequences')
parser.add_argument('--fdim', type=int, default=FDIM, help='Channel dimension of the features')
parser.add_argument('--mdim', type=int, default=MDIM, help='Channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=MEMORY_SIZE, help='Number of the memory items')
parser.add_argument('--num_workers', type=int, default=TRAIN_WORKERS, help='Number of workers for the train loader')
parser.add_argument('--num_workers_validation', type=int, default=VAL_WORKERS, help='Number of workers for the validation loader')
parser.add_argument('-d' ,'--dataset_type', type=str, default='aitex', help='Type of dataset: ped2, avenue, shanghai, aitex')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='Directory of data')
parser.add_argument('--alpha', type=float, default=ALPHA, help='Weight for the anomality score')
parser.add_argument("-ls", "--load_state", action="store_true", help="Load last trained model.")
parser.add_argument("--validation", action="store_false", default=True, help="Not use validation phase.")
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

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

args.dataset_type = args.dataset_type.upper()
train_folder = args.dataset_path+"/"+args.dataset_type+"/training/frames"
validation_folder = args.dataset_path+"/"+args.dataset_type+"/validating/frames"

log_dir = outputs_dir
date = datetime.now()
date = date.strftime("%Y-%m-%d_%H-%M-%S")
if not args.load_state:
    log_dir = outputs_dir + date + '/'
    os.mkdir(log_dir)
else:
    folders = sorted(list(b.folders_in(outputs_dir)))
    log_dir = folders[-1] + '/'


# Model setting
assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
if args.method == 'pred':
    from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
else:
    from model.Reconstruction import *
    model = convAE(args.c, memory_size = args.msize, feature_dim = args.fdim, key_dim = args.mdim)
params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)


def main(model, log_file):
    # Set manual seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset_type == 'AITEX':
        b.checkAitex()
        train_dataset = b.AitexDataSet(aitex_train_dir)
        channels = train_dataset[0].shape[0]
        original_height = train_dataset[0].shape[1]
        original_width = train_dataset[0].shape[2]
        # train_dataset, _, _ = b.resizeAitex(train_dataset, original_width, original_height)
        train_dataset, _, _ = b.augmentationDataset(train_dataset)
        b.DivideInPatches(train_dataset, train_folder, PATCH_SIZE, STRIDE, masks=False, save_image_patch=True)
        if args.validation:
            validation_dataset = b.AitexDataSet(aitex_validation_dir, masks=True)
            # validation_dataset, _, _ = b.resizeAitex(validation_dataset, original_width, original_height, True)
            validation_patches, mask_validation_patches = b.DivideInPatches(validation_dataset, validation_folder, PATCH_SIZE, STRIDE, masks=True, save_image_patch=True)
            validation_patches = torch.cat(validation_patches, dim=0).reshape(-1, channels, PATCH_SIZE, PATCH_SIZE)        
            mask_validation_patches = torch.cat(mask_validation_patches, dim=0).reshape(-1, channels, PATCH_SIZE, PATCH_SIZE)
        b.countAnomalies(validation_patches, mask_validation_patches, log_dir, save=False)

    b.myPrint('Dataset: ' + str(args.dataset_type) + '\nBatch size: ' + str(args.batch_size) + "\nEpochs: " + str(args.epochs), log_file)
    b.myPrint('Number of memory: ' + str(args.msize), log_file)
    b.myPrint('Patches size: ' + str(args.h) + "x" + str(args.w) + "x" + str(args.c), log_file)

    # Loading dataset
    train_dataset = DataLoader(train_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
    train_size = len(train_dataset)
    b.myPrint("Length (-1) of the frame sequences: {}, \nTrain images: {}".format(args.t_length-1, train_size), log_file)
    train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size,
                                shuffle=True, num_workers=args.num_workers, drop_last=True)

    if args.validation:
        validation_dataset = DataLoader(validation_folder, transforms.Compose([
                    transforms.ToTensor(),
                    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
    
        validation_size = len(validation_dataset)
        b.myPrint("Validation images: {}".format(validation_size), log_file)
        validation_batch = data.DataLoader(validation_dataset, batch_size = args.validation_batch_size,
                                shuffle=False, num_workers=args.num_workers_validation, drop_last=False)
    
    model.cuda()
    loss_func_mse = nn.MSELoss(reduction='none')


    if not args.load_state:
        # Training
        m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items
        for epoch in range(args.epochs):
            b.myPrint("Epoch: {}".format(epoch), log_file)
            model.train()
            
            start = time.time()
            for j, (imgs, name) in enumerate(tqdm(train_batch, ncols=0)):

                imgs = Variable(imgs).cuda()
                
                if args.method == 'pred':
                    outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:12], m_items, True)

                else:
                    outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)
                
                optimizer.zero_grad()
                if args.method == 'pred':
                    loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
                else:
                    loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
                    
                loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
                loss.backward(retain_graph=True)
                optimizer.step()
                
            scheduler.step()
            
            b.myPrint('----------------------------------------', log_file)
            b.myPrint('Epoch:' + str(epoch+1), log_file)
            if args.method == 'pred':
                b.myPrint('Loss: Prediction {:.6f} / Compactness {:.6f} / Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()), log_file)
            else:
                b.myPrint('Loss: Reconstruction {:.6f} / Compactness {:.6f} / Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()), log_file)
            # b.myPrint('Memory_items:', log_file)
            # b.myPrint(m_items, log_file)
            b.myPrint('----------------------------------------', log_file)

            torch.save(model, os.path.join(log_dir, 'model_partial.pth'))
            torch.save(m_items, os.path.join(log_dir, 'keys_partial.pt'))
            b.myPrint("Epoch {}, model saved.".format(epoch), log_file)
            
        b.myPrint('Training is finished', log_file)
        # Save the model and the memory items
        torch.save(model, os.path.join(log_dir, 'model.pth'))
        torch.save(m_items, os.path.join(log_dir, 'keys.pt'))
        os.remove(log_dir + 'model_partial.pth')
        os.remove(log_dir + 'keys_partial.pt')
    else:
        model = torch.load(log_dir + 'model.pth')
        model.cuda()
        m_items = torch.load(log_dir + 'keys.pt')


    # Validation
    if args.validation:
        th_dict = {}
        labels = np.load(log_dir + 'frame_labels.npy')

        for idx, t in enumerate(th_memory):
            b.myPrint('Evaluation with memory threshold: ' + str(t), log_file)
            
            videos = OrderedDict()
            videos_list = sorted(glob.glob(os.path.join(validation_folder, '*')))
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
            for k,(imgs, name) in enumerate(tqdm(validation_batch)):
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

                if point_sc < t:
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

            accuracy = AUC_(anomaly_score_total_list, labels_list)
            b.myPrint('AUC: ' + str(accuracy*100) + '%', log_file)

            th_dict[idx] = accuracy
        
        idx = np.argmax(th_dict.items())
        
        best_th_memory = th_memory[idx]
        b.myPrint("Best threshold for memories update: {}".format(best_th_memory), log_file)
        
        torch.save(best_th_memory, os.path.join(log_dir, 'best_th_memory.pth'))          # Threshold to update memory items
        
        
        acc = np.empty(th_anomaly_score_array.shape)
        for i, t1 in enumerate(th_anomaly_score_array):
            score_with_th = np.array([1 if e > t1 else 0 for e in anomaly_score_total_list]).astype(np.float64)
            accuracy = AUC(score_with_th, np.expand_dims(1 - labels_list, 0))
            b.myPrint("With th: {} --------> AUC: {}".format(t1, accuracy), log_file)
            acc[i] = accuracy

        index = np.argmax(acc)
        b.myPrint("Best threshold for anomaly score detection: {}".format(th_anomaly_score_array[index]), log_file)
        torch.save(th_anomaly_score_array[index], os.path.join(log_dir, 'best_th_anomaly.pth'))
        shutil.rmtree(args.dataset_path+"/"+args.dataset_type+"/validating/", ignore_errors=True)
    
    shutil.rmtree(args.dataset_path+"/"+args.dataset_type+"/training", ignore_errors=True)
    


if __name__ == '__main__':
    log_file = open(log_dir + "log.txt", "a")
    start_time = time.time()
    main(model, log_file)
    b.myPrint("---Execution time: %s seconds ---\n" % (time.time() - start_time), log_file)
    if Telegram_messages: b.telegram_bot_sendtext("MNAD: Training finished.")
    log_file.close()
    