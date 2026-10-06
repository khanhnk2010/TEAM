import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import extract_class_indices
from model_util import Discriminative_Pattern_Matching
from model_util import Discriminative_Pattern_Matching_without_sim


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[model] WARNING: no GPU found, running on CPU.")
    return torch.device("cpu")

def load_videomae_v2(checkpoint=DEFAULT_CHECKPOINT, device=None):
    from transformers import AutoConfig, AutoModel
    device = device or get_device()
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, config=config, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[model] loaded {checkpoint} on {device}")
    return model, device

@torch.inference_mode()
def extract_frame_vectors(model, input_clips):
    B = input_clips.shape[0] // seq_len
    clips = input_clips.view(B, seq_len, 3, 224, 224)
    # clips: (B, 16, 3, 224, 224)
    out = model.extract_features(clips)
    # out: (B, 1568, 768)  (or 1569 with CLS)

    B, N, D = out.shape
    temporal_steps = 8
    tokens_per_step = N // temporal_steps  # 196

    x = out.view(B, temporal_steps, tokens_per_step, D)
    frame_vectors = x.mean(dim=2)  # (B, 8, 768)
    return frame_vectors


class CNN_FSHead(nn.Module):
    def __init__(self, args):
        super(CNN_FSHead, self).__init__()
        self.args = args
        self.pretrained_backbone = args.pretrained_backbone
        last_layer_idx = -2

        if args.backbone == "ResNet":
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])
            self.backbone = backbone
            self.mid_dim = 2048
        elif args.backbone == "ViT":
            backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            self.backbone = backbone
            self.mid_dim = 768
        elif args.backbone == "VideoMAE":
            self.backbone, _ = load_videomae_v2()
            self.mid_dim = 768
          
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.seq_len = self.args.seq_len
        self.agg_num = self.args.agg_num

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def get_feats(self, spt, tar):
        if self.args.backbone == "ResNet":
            spt = self.backbone(spt)
            tar = self.backbone(tar)
        elif self.args.backbone == "ViT":
            spt = self.backbone(spt).unsqueeze(dim=-1).unsqueeze(dim=-1)
            tar = self.backbone(tar).unsqueeze(dim=-1).unsqueeze(dim=-1)
        elif self.args.backbone == "VideoMAE":
            spt = extract_frame_vectors(self.backbone, spt)
            tar = extract_frame_vectors(self.backbone, tar)

        return spt, tar

    def get_other(self, x):
        cls_num = x.size(1)
        other = []
        for i in range(cls_num):
            other.append(torch.cat((x[:, 0:i], x[:, i+1:cls_num]), dim=1))
        other = torch.stack(other, dim=1)

        return other

    def get_other_weight(self, x):
        agg_num = x.size(2)
        other = []
        for i in range(agg_num):
            other.append(torch.cat((x[:, :, 0:i], x[:, :, i+1:agg_num]), dim=2))
        other = torch.stack(other, dim=2)

        return other

    def pooling(self, spt, tar):
        spt = spt.reshape(-1, self.args.seq_len, *list(spt.shape[-3:]))
        tar = tar.reshape(-1, self.args.seq_len, *list(tar.shape[-3:]))

        spt = spt.mean(dim=-1).mean(dim=-1)
        tar = tar.mean(dim=-1).mean(dim=-1)

        return spt, tar

    def reshape(self, spt, tar, spt_labels):
        unique_labels = torch.unique(spt_labels)
        spt = [torch.index_select(spt, 0, extract_class_indices(spt_labels, c)) for c in unique_labels]
        spt = torch.stack(spt, dim=0)
        spt = spt.unsqueeze(dim=0)
        tar = tar.unsqueeze(dim=1).unsqueeze(dim=1)

        return spt, tar

    def get_spt_sim(self, spt):
        _, cls_num, t, c = spt.shape
        spt_other = self.get_other(spt)
        spt_other = spt_other.reshape(1, cls_num, cls_num - 1, t, c)
        sim = F.cosine_similarity(spt.unsqueeze(dim=2), spt_other, dim=-1)

        return sim

    def forward(self, spt_images, spt_labels, tar_images):
        raise NotImplementedError

    def distribute_model(self):
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

    def loss(self, task_dict, model_dict):
        loss = {'L': F.cross_entropy(model_dict['logits'], task_dict["target_labels"].long())}

        return loss


class TEAM_pos(CNN_FSHead):
    def __init__(self, args):
        super(TEAM_pos, self).__init__(args)
        n_head = 8
        self.DPM = Discriminative_Pattern_Matching(self.agg_num, 
                                                   d_model=self.mid_dim, 
                                                   n_head=n_head, 
                                                   d_k=self.mid_dim // n_head, 
                                                   d_v=self.mid_dim // n_head,
                                                   seq_len=self.seq_len)
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()

    def get_cum_dists_pos(self, spt, tar, weight=None):
        sim = F.cosine_similarity(spt, tar, dim=-1)
        dists = self.dropout(1 - sim)
        if weight is None:
            cum_dists = dists.sum(dim=-1)
        else:
            cum_dists = (dists * weight).sum(dim=-1)

        return cum_dists

    def get_cum_dists_neg(self, spt, tar, weight=None):
        sim = F.cosine_similarity(spt, tar, dim=-1)
        sim_other = self.get_other(sim)
        dists = self.dropout(1 - sim_other)
        if weight is None:
            cum_dists = dists.sum(dim=-1)
        else:
            cum_dists = (dists * weight.unsqueeze(dim=2)).sum(dim=-1)
        cum_dists = cum_dists.max(dim=-1)[0]

        return cum_dists

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_pos, tar_pos)

        return cum_dists_pos, cum_dists_pos, cum_dists_pos

    def forward(self, spt, spt_labels, tar, tar_labels=None):
        spt, tar = self.get_feats(spt, tar)
        spt, tar = self.pooling(spt, tar)
        spt, tar = self.reshape(spt, tar, spt_labels)
        tar_num, (_, cls_num, ins_num, t, c), agg_num = tar.size(0), spt.shape, self.agg_num
        spt, tar = spt.mean(dim=2), tar.mean(dim=2)

        spt_pos, spt_neg = self.DPM(spt)
        spt_pos, spt_neg = self.relu(spt_pos), self.relu(spt_neg)

        tar_pos, tar_neg = self.DPM(tar)
        tar_pos, tar_neg = self.relu(tar_pos), self.relu(tar_neg)

        spt_pos_sim, spt_neg_sim = self.get_spt_sim(spt_pos), self.get_spt_sim(spt_neg)
        spt_other = self.get_other(spt)
        spt_disc_pos, spt_disc_neg = self.DPM(spt, spt_other, spt_pos_sim, spt_neg_sim)
        spt_disc_pos, spt_disc_neg = self.relu(spt_disc_pos), self.relu(spt_disc_neg)
        spt_disc_pos, spt_disc_neg = spt_disc_pos.mean(dim=2), spt_disc_neg.mean(dim=2)

        cum_dists, cum_dists_pos, cum_dists_neg = self.get_cum_dists(spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg)

        return_dict = {'logits': -cum_dists,
                       'logits_pos': -cum_dists_pos,
                       'logits_neg': -cum_dists_neg}

        return return_dict

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long())}

        return loss


class TEAM_neg_with_pos_loss(TEAM_pos):
    def __init__(self, args):
        super(TEAM_neg_with_pos_loss, self).__init__(args)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_pos, tar_neg) + self.get_cum_dists_neg(spt_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        return cum_dists_neg, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


class TEAM_pos_neg(TEAM_pos):
    def __init__(self, args):
        super(TEAM_pos_neg, self).__init__(args)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_pos, tar_neg) + self.get_cum_dists_neg(spt_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        cum_dists = cum_dists_pos + cum_dists_neg

        return cum_dists, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


class TEAM_disc_without_sim(TEAM_pos):
    def __init__(self, args):
        super(TEAM_disc_without_sim, self).__init__(args)
        self.DPM = Discriminative_Pattern_Matching_without_sim(self.agg_num, seq_len=self.seq_len, d_model=self.d_model)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_disc_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_disc_pos, tar_neg) + self.get_cum_dists_neg(spt_disc_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        cum_dists = cum_dists_pos + cum_dists_neg

        return cum_dists, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


class TEAM(TEAM_pos):
    def __init__(self, args):
        super(TEAM, self).__init__(args)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_disc_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_disc_pos, tar_neg) + self.get_cum_dists_neg(spt_disc_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        cum_dists = cum_dists_pos + cum_dists_neg

        return cum_dists, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.way = 3
            self.shot = 1
            self.query_per_class = 3
            self.trans_dropout = 0.1
            self.seq_len = 4
            self.img_size = 224
            self.backbone = "ViT"
            self.num_gpus = 1
            self.cls_num = 10
            self.agg_num = 30
            self.fea_num = 30
            self.repeat = 2
            self.coefficient = 0.1

            self.lam = 0.1
            self.alpha = 0.1

    args = ArgsObject()
    torch.manual_seed(0)

    device = 'cpu'
    model = TEAM(args).to(device)

    support_imgs = torch.rand(args.way * args.shot, args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class, args.seq_len, 3, args.img_size, args.img_size).to(device)

    support_imgs = support_imgs.reshape(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size)
    target_imgs = target_imgs.reshape(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size)

    support_labels = torch.tensor([n for n in range(args.way)] * args.shot).to(device)
    target_labels = torch.tensor([n for n in range(args.way)] * args.query_per_class).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))

    task_dict = {}
    task_dict["support_set"] = support_imgs
    task_dict["support_labels"] = support_labels
    task_dict["target_set"] = target_imgs
    task_dict["target_labels"] = target_labels

    model_dict = model(support_imgs, support_labels, target_imgs, target_labels)

    loss = model.loss(task_dict, model_dict)
    print(loss)
