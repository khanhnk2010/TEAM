import torch
import numpy as np
import argparse
import os
from utils import print_and_log, get_log_files, TestAccuracies, aggregate_accuracy
from model import *

import torchvision
import video_reader
import random
import torch.nn.functional as F
import time
import lr_policy
import math

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.args.num_gpus > 0) else 'cpu')
        self.model = self.init_model()

        self.video_dataset = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.video_dataset, batch_size=1,
                                                        num_workers=self.args.num_workers)
        self.val_accuracies = TestAccuracies([self.args.dataset])

        self.accuracy_fn = aggregate_accuracy

        self.params = [p for p in self.model.parameters() if p.requires_grad]

        self.scaler = torch.amp.GradScaler('cuda')

        if self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(
                self.params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True)
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(
                self.params,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=5e-4)

        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        model = eval(self.args.method)(self.args)
        model = model.to(self.device)

        if torch.cuda.is_available() and self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--method", default="TEAM", help="few-shot method to use")
        parser.add_argument("--dataset", type=str, default="data/hmdb", help="Path to dataset")
        parser.add_argument("--tasks_per_batch", type=int, default=1,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--test_model_name", "-m", default="checkpoint_best_val.pt",
                            help="Path to model to load and test.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--eval_way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", "-qpc", type=int, default=6,
                            help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", "-qpct", type=int, default=1,
                            help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument("--num_val_tasks", type=int, default=300, help="number of random tasks to valid on.")
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=8, help="Num dataloader workers.")
        parser.add_argument("--backbone", default="ResNet")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--pretrained_backbone", "-pt", type=str, default=None,
                            help="pretrained backbone path, used by PAL")
        parser.add_argument("--val_on_test", default=False, action="store_true",
                            help="Danger: Validate on the test set, not the validation set. "
                                 "Use for debugging or checking overfitting on test set. "
                                 "Not good practice to use when developing, hyperparameter tuning or training models.")
        parser.add_argument("--pretrained_checkpoint", "-pc", type=str, default=None)
        parser.add_argument("--agg_num", type=int, default=3)
        parser.add_argument("--test_later", default=False, action="store_true")
        parser.add_argument("--learning_rate", type=float, default=0.001)

        args = parser.parse_args()
        
        checkpoint_dir = 'work'
        if args.way == 5:
            dir_text = '/'.join([args.method, args.backbone, '{}-shot'.format(args.shot), 'an{}'.format(args.agg_num)])
        else:
            dir_text = '/'.join([args.method, args.backbone, '{}-way_{}-shot'.format(args.way, args.shot), 'an{}'.format(args.agg_num)])

        args.training_iterations = 2000
        args.steps_iter = 200
        args.steps = [0, 3, 5, 7]
        args.lrs = [1, 0.5, 0.1, 0.01]
        args.max_epoch = 10
    
        args.checkpoint_dir = os.path.join(checkpoint_dir, args.dataset, dir_text)
            
        args.print_freq = args.training_iterations // 100
        args.val_iter = args.training_iterations // 20

        return args

    def set_lr(self, new_lr):
        for param_idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = new_lr

    def run(self):
        if self.args.pretrained_checkpoint is not None:
            self.load_checkpoint_from_pc(self.args.pretrained_checkpoint)

        accuracies, losses = {}, {}
        total_iterations = self.args.training_iterations

        iteration = self.start_iteration

        best_val_accuracy = 0

        start_time = time.time()
        last_text = ''
        for task_dict in self.video_loader:
            if iteration >= total_iterations:
                break
            torch.set_grad_enabled(True)

            lr = lr_policy.get_epoch_lr(self.args, float(iteration) / self.args.steps_iter)
            self.set_lr(lr)

            task_loss_dict, task_accuracy_dict = self.train_task(task_dict)
            for key in task_loss_dict.keys():
                loss_key = losses.get(key)
                if loss_key is None:
                    losses[key] = [task_loss_dict.get(key)]
                else:
                    losses.get(key).append(task_loss_dict.get(key))
            for key in task_accuracy_dict.keys():
                accuracy_key = accuracies.get(key)
                if accuracy_key is None:
                    accuracies[key] = [task_accuracy_dict.get(key)]
                else:
                    accuracies.get(key).append(task_accuracy_dict.get(key))

            # optimize
            if ((iteration + 1) % self.args.tasks_per_batch == 0) or \
                    (iteration == (total_iterations - 1)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # print training stats
            if (iteration + 1) % self.args.print_freq == 0:
                present_time = time.time()
                run_time = (present_time - start_time)
                ETA = (present_time - start_time) * total_iterations / (iteration + 1)
                text = 'Task [{}/{}], LR:{:.6f}, ETA [{:.0f}h{:.0f}m/{:.0f}h{:.0f}m]'. \
                    format(iteration + 1, total_iterations, lr, run_time // 3600, run_time % 3600 // 60, ETA // 3600, ETA % 3600 // 60)
                accuracies_text = ''
                for key in accuracies.keys():
                    accuracies_text += ', ' + key + ': {:.3f}'.format(torch.Tensor(accuracies.get(key)).mean().item())
                losses_text = ''
                for key in losses.keys():
                    losses_text += ', ' + key + ': {:.3f}'.format(torch.Tensor(losses.get(key)).mean().item())
                print_and_log(self.logfile, text + accuracies_text + losses_text)
                del accuracies, losses
                accuracies, losses = {}, {}

            # validate
            if ((iteration + 1) % self.args.val_iter == 0) or (iteration + 1) == total_iterations:
                if self.args.test_later:
                    self.save_checkpoint(iteration + 1, "checkpoint_{}.pt".format(iteration + 1))
                else:
                    accuracy_dict = self.evaluate("val")
                    acc = accuracy_dict[self.args.dataset]["acc"]
                    con = accuracy_dict[self.args.dataset]["con"]
                    text = 'Test Result Acc: {:.1f}+/-{:.1f}\n'.format(acc, con)

                    # save checkpoint if best validation score
                    if acc > best_val_accuracy:
                        best_val_accuracy = acc
                        self.save_checkpoint(iteration + 1, "checkpoint_best_val.pt")
                        text = '**Best** ' + text
                        last_text = '[{}/{}] '.format(iteration + 1, total_iterations) + text
                    print_and_log(self.logfile, text)

                    self.save_checkpoint(iteration + 1, "checkpoint_final.pt")

            iteration += 1

        # save the final model
        self.save_checkpoint(iteration + 1, "checkpoint_final.pt")
        print_and_log(self.logfile, last_text)

        self.logfile.close()

    def train_task(self, task_dict):
        """
        For one task, runs forward, calculates the loss and accuracy and backprops
        """
        self.model.train()

        task_dict = self.prepare_task(task_dict)
        with torch.autocast(device_type=self.device.type):
            model_dict = self.model(task_dict['support_set'],
                                    task_dict['support_labels'],
                                    task_dict['target_set'],
                                    task_dict['target_labels'])

            target_logits = F.softmax(model_dict['logits'], dim=-1)
            accuracy = self.accuracy_fn(target_logits, task_dict["target_labels"])
            task_accruacy_dict = {'Acc': accuracy}

            task_loss_dict = self.model.loss(task_dict, model_dict)
            loss = 0
            for key in task_loss_dict.keys():
                loss += task_loss_dict.get(key) / self.args.tasks_per_batch
            
        self.scaler.scale(loss).backward(retain_graph=False)

        return task_loss_dict, task_accruacy_dict

    def evaluate(self, mode="val"):
        self.model.eval()
        with torch.no_grad():

            self.video_loader.dataset.split = mode
            if mode == "val":
                n_tasks = self.args.num_val_tasks
            elif mode == "test":
                n_tasks = self.args.num_val_tasks

            accuracy_dict = {}
            accuracies = []
            iteration = 0
            item = self.args.dataset
            for task_dict in self.video_loader:
                if iteration >= n_tasks:
                    break
                iteration += 1

                task_dict = self.prepare_task(task_dict)
                with torch.cuda.amp.autocast():
                    model_dict = self.model(task_dict['support_set'],
                                            task_dict['support_labels'],
                                            task_dict['target_set'])

                    target_logits = F.softmax(model_dict['logits'], dim=-1)
                    acc = self.accuracy_fn(target_logits, task_dict["target_labels"])
                    accuracies.append(acc.item())
                    del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            accuracy_dict[item] = {"acc": accuracy,
                                   "con": confidence}
            self.video_loader.dataset.split = "train"

        return accuracy_dict

    def prepare_task(self, task_dict):
        """
        Remove first batch dimension (as we only ever use a batch size of 1) and move data to device.
        """
        for k in task_dict.keys():
            task_dict[k] = task_dict[k][0].to(self.device)
        return task_dict

    def save_checkpoint(self, iteration, name="checkpoint.pt"):
        d = {'iteration': iteration,
             'model_state_dict': self.model.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict()
             }
        torch.save(d, os.path.join(self.checkpoint_dir, name))

    def load_checkpoint(self, name="checkpoint_final.pt"):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, name))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.start_iteration = checkpoint['iteration']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_checkpoint_from_pc(self, name="checkpoint.pt"):
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)


def main():
    learner = Learner()
    learner.run()


if __name__ == "__main__":
    main()


