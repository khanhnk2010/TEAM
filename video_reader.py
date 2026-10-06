import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor


"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        #gt_a: ground truth action aka labels
        self.gt_a_list = []
        self.videos = []

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)

        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)


"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.dataset
        self.seq_len = args.seq_len
        self.split = "train"
        self.tensor_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                                         std=[0.225, 0.225, 0.225])])
        self.img_size = args.img_size

        self.way=args.way
        self.eval_way=args.eval_way
        self.shot=args.shot
        self.query_per_class=args.query_per_class

        self.train_split = Split()
        self.val_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self.read_dir()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []

        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size))
        video_transform_list.append(ColorJitter(brightness=0.5,
                                                contrast=0.5,
                                                saturation=0.5,
                                                hue=0.25))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)

    def read_dir(self):
        split_folders = os.listdir(self.data_dir)
        for split in split_folders:
            c = self.get_train_val_or_test_db(split)
            if c == None:
                continue
            class_folders = os.listdir(os.path.join(self.data_dir, split))
            class_folders.sort()
            self.class_folders = class_folders
            for class_folder in class_folders:
                video_folders = os.listdir(os.path.join(self.data_dir, split, class_folder))
                video_folders.sort()
                for video_folder in video_folders:
                    
                    imgs = os.listdir(os.path.join(self.data_dir, split, class_folder, video_folder))
                    if len(imgs) < self.seq_len:
                        continue
                    imgs.sort()
                    paths = [os.path.join(self.data_dir, split, class_folder, video_folder, img) for img in imgs]
                    paths.sort()
                    class_id = class_folders.index(class_folder)
                    c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, val: {}, test: {}".format(len(self.train_split), len(self.val_split), len(self.test_split)))

    """ return the current split being used """
    def get_train_val_or_test_db(self, split):
        if split == 'train':
            return self.train_split
        elif split == 'val':
            return self.val_split
        elif split == 'test':
            return self.test_split
        else:
            return None

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_val_or_test_db(self.split)
        return 1000000
        return len(c)

    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_val_or_test_db(self.split)
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes

    """Loads a single image from a specified path """
    def read_single_image(self, path):
        with Image.open(path) as i:
            i.load()
            return i


    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_val_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        n_frames = len(paths)

        if self.split == "train":
            interval = n_frames // self.args.seq_len
            idxs = [random.randint(ind*interval, ind*interval+interval-1) for ind in range(self.args.seq_len)]
            imgs = [self.read_single_image(paths[i]) for i in idxs]
        else:
            interval = n_frames // self.args.seq_len
            idxs = [int((ind*interval + ind*interval+interval-1) / 2) for ind in range(self.args.seq_len)]
            # idx_f = np.linspace(0, n_frames-1, num=self.args.seq_len)
            # idxs = [int(f) for f in idx_f]
            imgs = [self.read_single_image(paths[i]) for i in idxs]

        if (self.transform is not None):
            if self.split == "train":
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)

        return imgs, vid_id

    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        if self.split == "train":
            c = self.train_split
        elif self.split == "val":
            c = self.val_split
        elif self.split == "test":
            c = self.test_split
        classes = c.get_unique_classes()
        if self.split == "train":
            batch_classes = random.sample(classes, self.way)
        else:
            batch_classes = random.sample(classes, self.eval_way)

        if self.split == "train":
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):

            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
                real_support_labels.append(bc)
            for idx in idxs[self.args.shot:]:
                vid, vid_id = self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)

        s = list(zip(support_set, support_labels, real_support_labels))
        # random.shuffle(s)
        support_set, support_labels, real_support_labels = zip(*s)

        t = list(zip(target_set, target_labels, real_target_labels))
        # random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)

        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes)

        return {"support_set": support_set,
                "support_labels": support_labels,
                "real_support_labels": real_support_labels,
                "target_set": target_set,
                "target_labels": target_labels,
                "real_target_labels": real_target_labels,
                "batch_class_list": batch_classes}
