r""" PATNet training (validation) code """
import sys
sys.path.insert(0, "../")
import argparse
import torch
from thop import profile
import torch.optim as optim
import torch.nn as nn
from model.DAGNN import OneModel
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

def train(epoch, model, dataloader, optimizer, training):
    r""" Train PATNet """

    utils.fix_randseed(42) if training else utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    # if training:
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        logit_mask = model(batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), batch['query_img'])
        pred_mask = torch.argmax(logit_mask, dim=1)

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    return avg_loss, miou, fb_iou

if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Unleashing Generalization Potential of VFM for Cross-Domain Few-Shot Segmentation')
    parser.add_argument('--datapath', type=str, default='./Dataset')
    parser.add_argument('--benchmark', type=str, default='pascal')
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50', help='backbone of semantic segmentation model')   
    parser.add_argument('--logpath', type=str, default='./DAGNN')
    parser.add_argument('--bsz', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--niter', type=int, default=20)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--sam2_weight', type=str, default='./pretrained/sam2_hiera_small.pt', help='number of support pairs')
    parser.add_argument('--sam2_config', type=str, default='../sam2_configs/sam2_hiera_s.yaml', help='number of support pairs')
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, default='./pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', help="Path to the pretrained DINO checkpoint (.pth). "
                                                            "Use ViT-B/16 checkpoint for --dino_size b, or ViT-S/16 checkpoint for --dino_size s.")
    parser.add_argument("--dino_size", type=str, default="base", choices=["base", "small"], help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = OneModel(args)
    for param in model.sam2.image_encoder.parameters():
        param.requires_grad = False
    for param in model.sam2.sam_prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.sam2.obj_ptr_proj.parameters():
        param.requires_grad = False
    for param in model.sam2.mask_downsample.parameters():
        param.requires_grad = False

    #  dinov3
    for param in model.dinov3.parameters():
        param.requires_grad = False

    total_params = 0
    total_trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            total_trainable_params += param.numel()

    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {total_trainable_params}")

    # Device setup
    # Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model).cuda()

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}]) 
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_val = FSSDataset.build_dataloader('fss', args.bsz, args.nworker, '0', 'val')

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        Logger.save_model_miou(model, epoch, val_miou)
        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss.float().item(), 'val_loss': val_loss.float().item()}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
