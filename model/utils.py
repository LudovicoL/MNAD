import shutil
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.utils as v_utils

rng = np.random.RandomState(2020)

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    print("######## nindex {}, height {}, width {}, intensity {}, nrows {}".format(nindex, height, width, intensity, nrows))
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    # tensor = d.cpu().numpy()  # make sure tensor is on cpu
    # cv2.imshow("image.png", image_decoded)
    # cv2.waitKey()
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized




class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
        return frames
    
    
    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        # batch_nt = []
        # batch_arr = []
        frame_n = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            # print("########### frame_name: {}".format(self.videos[video_name]['frame'][frame_name+i]))
            frame_n.append(self.videos[video_name]['frame'][frame_name+i])
            if self.transform is not None:
                # print("{} ########### image_{}: {}".format(i, i, self.transform(image).shape))
                batch.append(self.transform(image))

                # arr = image
                # new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
                # batch_arr.append(new_arr)

                # plt.imshow(new_arr)
                # plt.show()
                # new_arr = torch.tensor(new_arr)
                # new_arr = new_arr.unsqueeze(0).permute(0, 3, 1, 2)
                # new_arr = new_arr.permute(2, 0, 1)
                # print("########### new_arr: {}".format(new_arr.shape))
                # batch_nt.append(new_arr)
                # print("########### batch_nt: {}, {}".format(len(batch_nt), batch_nt[index]))
                # batch_nt.append(image)

        # fig = plt.figure(figsize=(100, 100), dpi=100)
        # columns = 5
        # rows = 1
        # for i in range(1, columns * rows + 1):
        #     fig.add_subplot(rows, columns, i)
        #     arr = batch_nt[i-1]
        #     new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
        #     plt.imshow(new_arr)
        # plt.show()

        ########
        # # plot
        # plt.figure(figsize=(10, 10), dpi=150)
        # # plt.figure()
        # result = gallery(np.array(batch_arr), ncols=5)
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # plt.imshow(result)
        # # plt.show()
        ########

        ########
        # # save
        # if not os.path.exists("./pred_results/{}".format(index)):
        #     os.makedirs("./pred_results/{}".format(index))
        # else:
        #     shutil.rmtree("./pred_results/{}".format(index))
        #     os.makedirs("./pred_results/{}".format(index))
        # plt.savefig('./pred_results/{}/grid_item_{}.png'.format(index, index))
        # plt.close()
        ########

        # batch_f = torch.cat(batch_nt, 0)
        # print("########### batch_f: {}".format(batch_f.shape))
        # grid_img = v_utils.make_grid(batch_f, padding=0)
        # grid_img = v_utils.make_grid(batch_nt, padding=0)
        # v_utils.save_image(grid_img, './grid_item_{}.png'.format(index))
        # print("########### batch: {}".format(np.concatenate(batch, axis=0).shape))
        # print("########### batch: {}, {}, {}".format(type(batch), len(batch), batch[0].shape))

        ########
        # print("####### leggo item: {} composto da: {}".format(index, frame_n))
        ########

        return np.concatenate(batch, axis=0), frame_n
        
        
    def __len__(self):
        return len(self.samples)
