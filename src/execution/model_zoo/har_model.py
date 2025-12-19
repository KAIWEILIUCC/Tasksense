import math
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.nn.init import constant_, normal_
from torch.optim.lr_scheduler import LambdaLR
# from har_rgbd_test import get_har_rgbd


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def create_optimizer(module):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
    {'params': [p for n, p in module.named_parameters() if not any(
        nd in n for nd in no_decay)], 'weight_decay': 5e-4},
    {'params': [p for n, p in module.named_parameters() if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=0.01,
                        momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 2**20)
    return optimizer, scheduler

class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        # self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach() # new model
                ema_v = esd[k] # history model
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

class Identity(torch.nn.Module):
    def forward(self, input):
        return input

class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output
    
class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)
    
class TSN(nn.Module):
    def __init__(
        self,
        num_class=30,
        num_segments=8,
        modality="RGB",
        base_model="resnet18",
        new_length=None,
        consensus_type="avg",
        before_softmax=True,
        dropout=0.8,
        img_feature_dim=256,
        crop_num=1,
        partial_bn=True,
        print_spec=False,
        pretrain="imagenet",
        is_shift=True,
        shift_div=8,
        shift_place="blockres",
        fc_lr5=False,
        temporal_pool=False,
        non_local=False,
    ):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim
        self.crop_num = crop_num
        self.pretrain = pretrain
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.num_class = num_class
        self.base_model_name = base_model
        self.pretrain = pretrain
        self.fc_lr5 = fc_lr5
        self.print_spec = print_spec
        self.partial_bn = partial_bn
        if not before_softmax and consensus_type != "avg":
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            # self.new_length = 1 if modality == "RGB" else 5
            self.new_length = 1
        else:
            self.new_length = new_length

        if self.print_spec:
            print(
                (
                    """
            Initializing TSN with base model: {}.
            TSN Configurations:
            input_modality:     {}
            num_segments:       {}
            new_length:         {}
            consensus_module:   {}
            dropout_ratio:      {}
            img_feature_dim:    {}
            """.format(
                        base_model,
                        self.modality,
                        self.num_segments,
                        self.new_length,
                        consensus_type,
                        self.dropout,
                        self.img_feature_dim,
                    )
                )
            )
    
        self._prepare_base_model(base_model)
        feature_dim = self._prepare_tsn(num_class)
        
        if self.modality == 'Depth':
            if print_spec: print("Converting the ImageNet model to a Depth init model")
            self.base_model = self._construct_l_model(self.base_model)
            if print_spec: print("Done. Depth model ready...")
        
        self.consensus = ConsensusModule(consensus_type)
        
        if not self.before_softmax:
            self.softmax = nn.Softmax()
        
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
    
    def _prepare_tsn(self, num_class=30):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.classifier = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.classifier = nn.Linear(feature_dim, num_class)
            
        std = 0.001
        if self.classifier is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.classifier, 'weight'):
                normal_(self.classifier.weight, 0, std)
                constant_(self.classifier.bias, 0)
        return feature_dim
    
    def _prepare_base_model(self, base_model):
        if self.print_spec: print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                if self.print_spec: print('Adding temporal shift...')
                from src.execution.model_zoo.ops.temporal_shift import \
                    make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                if self.print_spec: print('Adding non-local module...')
                from src.execution.model_zoo.ops.non_local import \
                    make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc' 
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            
            if self.modality == 'Depth':
                self.input_mean = [2500]
                self.input_std = [0.25]
            
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
    
    def partialBN(self, enable):
        self._enable_pbn = enable
    
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]
            
    def forward(self, input, no_reshape=False):
        input = input.contiguous().clone()
        B, C, T, W, H = input.size()
        input = input.transpose(1, 2).contiguous()
        
        if not no_reshape:
            if self.modality == "RGB":
                sample_len = 3
            elif self.modality == "Depth":
                sample_len = 1
            
            input = input.view((-1, sample_len) + input.size()[-2:])
            base_out = self.base_model(input)
        else:
            base_out = self.base_model(input)
        
        cls_out = self.classifier(base_out.clone())
        if not self.before_softmax:
            cls_out = self.softmax(cls_out)
        feat_out = base_out.clone()
        
        if self.reshape:
            if self.is_shift and self.temporal_pool:
                feat_out = feat_out.view((-1, self.num_segments // 2) + feat_out.size()[1:]).contiguous()
                cls_out = cls_out.view((-1, self.num_segments // 2) + cls_out.size()[1:]).contiguous()
            else:
                feat_out = feat_out.view((-1, self.num_segments) + feat_out.size()[1:])
                cls_out = cls_out.view((-1, self.num_segments) + cls_out.size()[1:])
            feat_out = self.consensus(feat_out)
            cls_out = self.consensus(cls_out)
            return feat_out.squeeze(1), cls_out.squeeze(1)
        else:
            return feat_out, cls_out
    
    def _construct_l_model(self, base_model):
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (1 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.print_spec: print('#' * 30, 'Warning! No L pretrained model is found')
        return base_model


