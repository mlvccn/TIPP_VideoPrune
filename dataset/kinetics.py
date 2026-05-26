import torch
import os
import warnings
import numpy as np
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None
from PIL import Image
import torch
from torch.utils.data import Dataset
from gluoncv.torch.data import video_transforms, volume_transforms, multiGridHelper, MultiGridBatchSampler
import pandas as pd
import random
import cv2
class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=False,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 use_multigrid=False):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.labels_list=[]
        self.use_multigrid = use_multigrid and (mode == 'train')
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        # cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        # self.dataset_samples = list(cleaned.values[:, 0])
        # self.label_array = list(cleaned.values[:, 2])
        cleaned = pd.read_csv(self.anno_path)
        self.dataset_samples = cleaned['youtube_id'].values.tolist()
        self.label_array =  cleaned['label'].values.tolist()
        self.st = cleaned['time_start'].values.tolist()
        self.ed = cleaned['time_end'].values.tolist() 
        self.validnum= cleaned['validnum'].values.tolist() 
        with open('/data/Kinetics-400/label/label.txt', 'r') as file:
                for _, line in enumerate(file):
                    # Strip whitespace and newline characters from the line
                    label = line.strip()
                    # Record the label with its corresponding line number
                    self.labels_list.append(label)
        #self.filter_invalid_samples()
        if (mode == 'train'):
            if self.use_multigrid:
                self.mg_helper = multiGridHelper()
                self.data_transform = []
                for alpha in range(self.mg_helper.mod_long):
                    tmp = []
                    for beta in range(self.mg_helper.mod_short):
                        info = self.mg_helper.get_resize(alpha, beta)
                        scale_s = info[1]
                        tmp.append(video_transforms.Compose([
                            video_transforms.Resize(int(self.short_side_size / scale_s),
                                                    interpolation='bilinear'),
                            # TODO: multiscale corner cropping
                            video_transforms.RandomResize(ratio=(1, 1.25),
                                                          interpolation='bilinear'),
                            video_transforms.RandomCrop(size=(int(self.crop_size / scale_s),
                                                              int(self.crop_size / scale_s)))]))
                    self.data_transform.append(tmp)

            else:
                self.data_transform = video_transforms.Compose([
                    video_transforms.Resize(int(self.short_side_size),
                                            interpolation='bilinear'),
                    video_transforms.RandomResize(ratio=(1, 1.25),
                                                  interpolation='bilinear'),
                    video_transforms.RandomCrop(size=(int(self.crop_size),
                                                      int(self.crop_size)))])

            self.data_transform_after = video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif (mode == 'validation'):
            # self.data_resize = video_transforms.Compose([
            #     video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            # ])
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.crop_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            # self.test_seg = []
            # self.test_dataset = []
            # self.test_label_array = []
            # for ck in range(self.test_num_segment):
            #     for cp in range(self.test_num_crop):
            #         for idx in range(len(self.label_array)):
            #             sample_label = self.label_array[idx]
            #             self.test_label_array.append(sample_label)
            #             self.test_dataset.append(self.dataset_samples[idx])
            #             self.test_seg.append((ck, cp))
            
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.use_multigrid is True:
                index, alpha, beta = index
                info = self.mg_helper.get_resize(alpha, beta)
                scale_t = info[0]
                data_transform_func = self.data_transform[alpha][beta]
            else:
                scale_t = 1
                data_transform_func = self.data_transform

            sample = self.dataset_samples[index]
            st=self.st[index]
            ed=self.ed[index]
            num=self.validnum[index]
            label=self.label_array[index]
            buffer = self.loadvideo_decord(sample,st,ed,label,num,'train')
            buffer = data_transform_func(buffer)
            buffer = self.data_transform_after(buffer)
            buffer = [torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame for frame in buffer]
            clip = torch.stack(buffer, 0)
            #print(clip.size())
            return clip, self.labels_list.index(self.label_array[index]), index 

        elif self.mode == 'validation':
            #sample = self.test_dataset[index]
           #chunk_nb, split_nb = self.test_seg[index]
            sample = self.dataset_samples[index]
            st=self.st[index]
            ed=self.ed[index]
            label=self.label_array[index]
            num=self.validnum[index]
            buffer = self.loadvideo_decord(sample,st,ed,label,num,'val')
            #print(len(buffer))
           #buffer=self.data_resize(buffer)
            # spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
            #                      / (self.test_num_crop - 1)
            # temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
            #                     / (self.test_num_segment - 1), 0)
            # temporal_start = int(chunk_nb * temporal_step)
            # spatial_start = int(split_nb * spatial_step)
            # if buffer.shape[1] >= buffer.shape[2]:
            #     buffer = buffer[temporal_start:temporal_start + self.clip_len, \
            #            spatial_start:spatial_start + self.short_side_size, :, :]
            # else:
            #     buffer = buffer[temporal_start:temporal_start + self.clip_len, \
            #            :, spatial_start:spatial_start + self.short_side_size, :]
            buffer = self.data_transform(buffer)
            #buffer = [torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame for frame in buffer]
            #clip = torch.stack(buffer, 0)
            #print(clip.size())
            return buffer, self.labels_list.index(self.label_array[index]), index 
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def filter_invalid_samples(self):
        """Filter out samples with invalid paths or missing images."""
        valid_samples = []
        valid_st = []
        valid_ed = []
        valid_labels = []
        valid_num=[]

        # Check each sample
        for i, sample in enumerate(self.dataset_samples):
            print(i)
            st = self.st[i]
            ed = self.ed[i]
            label = self.label_array[i]
            valid,num=self.is_valid_sample(sample, st, ed, label)
            # Attempt to load video frames
            if valid:
                valid_samples.append(sample)
                valid_st.append(st)
                valid_ed.append(ed)
                valid_labels.append(label)
                valid_num.append(num)

        # Update the dataset with only valid samples
        self.dataset_samples = valid_samples
        self.st = valid_st
        self.ed = valid_ed
        self.label_array = valid_labels
        self.save_valid_samples_to_csv(valid_samples, valid_labels, valid_st, valid_ed,valid_num)
    def save_valid_samples_to_csv(self, samples, labels, start_times, end_times,valid_num):
        """Save valid video samples to a CSV file."""
        data = {
            'youtube_id': samples,
            'label': labels,
            'time_start': start_times,
            'time_end': end_times,
            'validnum': valid_num
        }
        valid_df = pd.DataFrame(data)
        valid_df.to_csv('valid_samples.csv', index=False)
        print('Valid samples saved to valid_samples.csv')
    def is_valid_sample(self, sample, st, ed, label):
        """Check if a sample is valid by attempting to load it."""
        #train

        label = label.replace(' ', '_')
        formatted_start = f"{st:06d}"
        formatted_end = f"{ed:06d}"
        fname = os.path.join(self.data_path, label, f"{sample}_{formatted_start}_{formatted_end}")

        #valid

        # label = label.replace(' ', '_')
        # fname = os.path.join(
        # self.data_path,
        #     sample)

        # Check if the main folder exists
        if not os.path.exists(fname):
            return False,0
        file_count = len([f for f in os.listdir(fname) if os.path.isfile(os.path.join(fname, f))])
        # Check if all expected images exist
        # for i in range(st, ed + 1):
        #     img_name = f"img_{i:05d}.jpg"
        #     img_path = os.path.join(fname, img_name)
        #     if not os.path.exists(img_path):
        #         return False


        return True,file_count


    def loadvideo_decord(self, sample, st, ed, label, num, mode):
        """Load video content using Decord and save frames to a new location"""
        if mode == 'train':
            formatted_start = f"{st:06d}"  # Format start time to six digits
            formatted_end = f"{ed:06d}"    # Format end time to six digits
            label = label.replace(' ', '_')
            fname = os.path.join(self.data_path, label, f"{sample}_{formatted_start}_{formatted_end}")
        else:
            # label = label.replace(' ', '_')
            # fname = os.path.join(self.data_path,sample)
            formatted_start = f"{st:06d}"  # Format start time to six digits
            formatted_end = f"{ed:06d}"    # Format end time to six digits
            label = label.replace(' ', '_')
            fname = os.path.join(self.data_path, label, f"{sample}")
        if not os.path.exists(fname):
            print(fname)

        frames = []
        #print(fname)
        
        # 定义目标路径
        # save_base_path = "/home/pingchenhao/Kinetics-400-val/"
        # target_folder = os.path.join(save_base_path, label, f"{sample}")
        # print(target_folder)
        
        # 创建目标路径（如果不存在）
        #os.makedirs(target_folder, exist_ok=True)

        if mode == 'train':
            for i in range(1, num+1, 20):
                if len(frames) < 16:
                    img_name = f"img_{i:05d}.jpg"
                    img_path = os.path.join(fname, img_name)
                    
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(np.array(img))
                    
                    # 将图像保存到新路径
                    # save_img_path = os.path.join(target_folder, img_name)
                    # cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                else:
                    break
            while len(frames) < 16:
                img_name = f"img_{num:05d}.jpg"
                img_path = os.path.join(fname, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(np.array(img))
                
                # 保存图像到新路径
                # save_img_path = os.path.join(target_folder, img_name)
                # cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        else:
            for i in range(1, num+1, 20):
                if len(frames) < 16:
                    img_name = f"img_{i:05d}.jpg"
                    img_path = os.path.join(fname, img_name)
                    
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(np.array(img))
                    
                    # 保存图像到新路径
                    # save_img_path = os.path.join(target_folder, img_name)
                    # cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                else:
                    break
            while len(frames) < 16:
                img_name = f"img_{num:05d}.jpg"
                img_path = os.path.join(fname, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(np.array(img))
                
                # 保存图像到新路径
                # save_img_path = os.path.join(target_folder, img_name)
                # cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return frames


    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)
        


def get_dataset(cfg, mode,loader=True):
    if mode=='train':
        val_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.TRAIN_ANNO_PATH,
                                  data_path=cfg.CONFIG.DATA.TRAIN_DATA_PATH,
                                  mode=mode,
                                  use_multigrid=cfg.CONFIG.DATA.MULTIGRID,
                                  clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                  frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                  num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                                  num_crop=cfg.CONFIG.DATA.NUM_CROP,
                                  keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                  crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                  short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                  new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                  new_width=cfg.CONFIG.DATA.NEW_WIDTH)
    if mode=='validation':
        val_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.VAL_ANNO_PATH,
                                  data_path=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                  mode=mode,
                                  use_multigrid=cfg.CONFIG.DATA.MULTIGRID,
                                  clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                  frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                  num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                                  num_crop=cfg.CONFIG.DATA.NUM_CROP,
                                  keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                  crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                  short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                  new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                  new_width=cfg.CONFIG.DATA.NEW_WIDTH)
    print ('The length of Dataset is {}.'.format(len(val_dataset)))
    if mode=='train' and loader:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE, shuffle=True,
            num_workers=5, sampler=None, pin_memory=True)
        return val_loader
    elif mode=='validation':
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=False,
            num_workers=5, sampler=None, pin_memory=True)
        return val_loader
    elif loader==False:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=True,
            num_workers=5, sampler=None, pin_memory=False)
        return val_loader
