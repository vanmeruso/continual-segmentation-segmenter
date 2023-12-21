import os
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from utils import eval_seg
from torch.utils import data
import torch.nn.functional as F
from utils import dist_utils

from utils.optim.factory import create_optimizer, create_scheduler

from utils import ext_transforms as et
from datasets import Cityscapes, Acdc

from core.Segmenter.utils import inference
from core import Segmenter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datasets import imutils
import torch.cuda.amp as amp

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='./configs/cityscapes.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--gpu_ids", type=list, default=[0,1],help="GPU ID")
parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
parser.add_argument("--test_only", action='store_true', default=False)

##################################################################
                         # for prompt #
#G-prompt parameters
parser.add_argument('--use_g_prompt', default=True, type=bool, help='if using G-Prompt')
parser.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt')
parser.add_argument('--g_prompt_layer_idx', default=[0, 1], type=int, nargs = "+", help='the layer index of the G-Prompt')
parser.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool, help='if using the prefix tune for G-Prompt')

# E-Prompt parameters
parser.add_argument('--use_e_prompt', default=True, type=bool, help='if using the E-Prompt')
parser.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs = "+", help='the layer index of the E-Prompt')
parser.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool, help='if using the prefix tune for E-Prompt')

# prompt pool
parser.add_argument('--prompt_pool', default=True, type=bool,)
parser.add_argument('--size', default=10, type=int,)
parser.add_argument('--length', default=20,type=int, )
parser.add_argument('--top_k', default=1, type=int, )
parser.add_argument('--initializer', default='uniform', type=str,)
parser.add_argument('--prompt_key', default=True, type=bool,)
parser.add_argument('--prompt_key_init', default='uniform', type=str)
parser.add_argument('--use_prompt_mask', default=True, type=bool)
parser.add_argument('--mask_first_epoch', default=False, type=bool)
parser.add_argument('--shared_prompt_pool', default=True, type=bool)
parser.add_argument('--shared_prompt_key', default=False, type=bool)
parser.add_argument('--batchwise_prompt', default=False, type=bool)
parser.add_argument('--embedding_key', default='cls', type=str)
parser.add_argument('--pull_constraint', default=True)
parser.add_argument('--pull_constraint_coeff', default=1.0, type=float)
parser.add_argument('--same_key_value', default=False, type=bool)
#############################################################



def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_dataset(opts):

    train_transform = et.ExtCompose([
        et.ExtRandomCrop(size=(opts.dataset.crop_size[0], opts.dataset.crop_size[0] ), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    

    if opts.dataset.name == 'cityscapes':
        train_dst = Cityscapes(root=opts.dataset.data_root,
                                split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.dataset.data_root,
                                split='val', transform=val_transform)
    else:
        train_dst = Acdc(root=opts.dataset.data_root,
                                split='train', domain= opts.dataset.domain, transform=train_transform)
        val_dst = Acdc(root=opts.dataset.data_root,
                                split='val', domain= opts.dataset.domain, transform=val_transform)
        
    dataset_dict = {}
    dataset_dict['train'] = train_dst
    
    dataset_dict['test'] = val_dst
    
    return dataset_dict

def validate(model,criterion, data_loader,device, window_size,window_stride, num_class, local_rank):
    
    intersection_meter = eval_seg.AverageMeter()
    union_meter = eval_seg.AverageMeter()
    preds, gts = [], []
    model.eval()
    
    if local_rank == 0:
        data_loader = tqdm(data_loader)

    with torch.no_grad():
       for inputs, labels  in data_loader:
            inputs = inputs.to(device, dtype = torch.float32, non_blocking = True) 
            labels = labels.to(device, dtype = torch.long, non_blocking = True)
            ori_shape = inputs.shape[2:]
            seg_pred = inference(model,inputs,ori_shape,window_size,window_stride,inputs.shape[0],device, 19)

            pred = seg_pred.argmax(dim = 1).detach()
            intersection, union, target = eval_seg.intersectionAndUnion(pred.cpu().numpy(), labels.cpu().numpy(), num_class, 255)
            reduced_intersection = torch.from_numpy(intersection).to(device)
            reduced_union = torch.from_numpy(union).to(device)
            reduced_target = torch.from_numpy(target).to(device)

            reduced_intersection = dist_utils.all_reduce_tensor(reduced_intersection)
            reduced_union = dist_utils.all_reduce_tensor(reduced_union)
            reduced_target = dist_utils.all_reduce_tensor(reduced_target)
            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy()) 
        
    iou_class = (intersection_meter.sum / (union_meter.sum + 1e-10)) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class
 

def train(opts):
    """
        
        opts.random_seed = 1
    """
    num_workers = 4
    #writer = SummaryWriter('runs/segformer')
    
    torch.cuda.set_device(args.gpu_ids[args.local_rank])
    dist.init_process_group(backend=args.backend,)
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    torch.cuda.manual_seed_all(opts.random_seed)
    random.seed(opts.random_seed)
    torch.backends.cudnn.deterministic = True
    
    dataset_dict = get_dataset(opts)
    train_sampler = DistributedSampler(dataset_dict['train'],shuffle=True)
    val_sampler = DistributedSampler(dataset_dict['test'],shuffle=False)

    train_loader = data.DataLoader(
        dataset_dict['train'], 
        batch_size=opts.dataset.batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True, 
        prefetch_factor=4)
    test_loader = data.DataLoader(
        dataset_dict['test'], sampler = val_sampler, batch_size=opts.dataset.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    scaler = amp.GradScaler(enabled=True)
    model = Segmenter(backbone=opts.backbone, num_classes=opts.dataset.num_classes, pretrained=True)
    """
    # for prompt 
    model = segformer(backbone=opts.backbone,
                num_classes=opts.dataset.num_classes,
                embedding_dim=768,
                pretrained=True,
                prompt_length=args.length,
                embedding_key=args.embedding_key,
                prompt_init=args.prompt_key_init,
                prompt_pool=args.prompt_pool,
                prompt_key=args.prompt_key,
                pool_size=args.size,
                top_k=args.top_k,
                batchwise_prompt=args.batchwise_prompt,
                prompt_key_init=args.prompt_key_init,
                use_prompt_mask=args.use_prompt_mask,
                use_g_prompt=args.use_g_prompt,
                g_prompt_length=args.g_prompt_length,
                g_prompt_layer_idx=args.g_prompt_layer_idx,
                use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
                use_e_prompt=args.use_e_prompt,
                e_prompt_layer_idx=args.e_prompt_layer_idx,
                use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
                same_key_value=args.same_key_value,)
    """

    param_groups = model.get_param_groups()
    model.to(device)
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))["model_state"]  
        
        model.load_state_dict(checkpoint, strict=True)  
        if args.local_rank==0:
            print("Model restored from %s" % args.ckpt)
        del checkpoint  # free memory
    """
    for i, (name, param) in enumerate(model.encoder.named_parameters()):
        param.requires_grad = False
    """
    model = DistributedDataParallel(model, device_ids=[args.gpu_ids[args.local_rank]], find_unused_parameters=True)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=opts.dataset.ignore_index)
    criterion = criterion.to(device)

    train_sampler.set_epoch(0)
    max_iters = opts.train.epochs * len(train_loader)
    val_interval = max(100, max_iters // 10)

    optimizer_kwargs=dict(
            opt=opts.optimizer.opt,
            lr=opts.optimizer.lr,
            weight_decay=opts.optimizer.weight_decay,
            momentum=opts.optimizer.momentum,
            clip_grad=None,
            sched="polynomial",
            epochs=opts.train.epochs,
            min_lr=opts.optimizer.min_lr,
            poly_power=opts.optimizer.poly_power,
            poly_step_size=1,
        )

    optimizer_kwargs["iter_max"] = max_iters
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)


    
    def save_ckpt(path):
        torch.save({"model_state": model.module.state_dict()}, path)
        
        if args.local_rank==0:
            print("Model saved as %s" % path)
    

    if args.test_only:
        if args.local_rank == 0:
            print("model testing start")
        model.eval()
        
        _, val_score = validate(model=model, criterion=criterion, device= device, data_loader=test_loader)
        if args.local_rank==0: 
            print("mIOU: %f"%(val_score['Mean IoU']))    
        return
    

    if args.local_rank==0:
        print("==============================================")
        print("  Device: %s" % device)
        print( "  opts : ")
        print(opts)
        print("==============================================")
    
        print("Dataset: %s, Train set: %d, Test set: %d" %
          (opts.dataset.name, len(dataset_dict['train']), len(dataset_dict['test'])))

        print(f"train epoch : {opts.train.epochs} , iterations : {max_iters} , val_interval : {val_interval}")
    

    cur_epochs = 0
    
    for n_iter in range(max_iters):
        model.train()
        try:
            inputs, labels = next(train_loader_iter)
        except:
            train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            inputs, labels = next(train_loader_iter)
            cur_epochs += 1
        
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast(True):
            outputs= model(inputs, train =True)
            seg_loss = criterion(outputs, labels)
        
        dist_utils.barrier()

        scaler.scale(seg_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        #optimizer.zero_grad()
        #seg_loss.backward()
        #optimizer.step()
        #lr_scheduler.step_update(n_iter)
        torch.cuda.synchronize()

        if (n_iter+1) % opts.train.log_iters == 0 and args.local_rank==0:
            delta, eta = cal_eta(time0, n_iter+1, max_iters)
            lr = optimizer.param_groups[0]['lr']
            print("[Epochs: %d Iter: %d] Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %f"%(cur_epochs, n_iter+1, delta, eta, lr, seg_loss.item())) 
  
            
        if (n_iter+1) % val_interval == 0:
            if args.local_rank==0:
                print('Validating...')
            miou, miou_class = validate(model=model, criterion=criterion, device= device, data_loader=test_loader, window_size = 768, window_stride = 512,  num_class = opts.dataset.num_classes, local_rank = args.local_rank)  
        
            if args.local_rank==0:
                print("mIOU: %f"%(miou))
   
    
    if args.local_rank==0:
        if args.ckpt is not None:
                    prefix = args.ckpt.split('.')
                    previous = prefix[1].lstrip('/')
                    save_ckpt(previous+'_'+ opts.dataset.domain + '.pth')
        else:
            save_ckpt('checkpoints/%s.pth'% (opts.dataset.name))


    print("end")
    return True

if __name__ == "__main__":
    
    args = parser.parse_args()
    opts = OmegaConf.load(args.config)
    
    train(opts=opts)

