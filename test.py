r""" Cross-Domain Few-Shot Semantic Segmentation testing code """
import argparse
import torch
import torch.nn as nn
from model.DAGNN import OneModel
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

def test(model, dataloader, shot):
    r""" Test PATNet """

    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_shot(batch, shot=shot)
        assert pred_mask.size() == batch['query_mask'].size()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou

if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Cross-Domain Few-Shot Semantic Segmentation Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='./Dataset')
    parser.add_argument('--benchmark', type=str, default='isic', choices=['fss', 'lung', 'isic', 'deepglobe'])
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50', help='backbone of semantic segmentation model')  
    parser.add_argument('--logpath', type=str, default='./')
    parser.add_argument('--bsz', type=int, default=60)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='./logs/DAGNN.log/ISIC.pt')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--sam2_weight', type=str, default='./pretrained/sam2_hiera_large.pt', help='number of support pairs')
    parser.add_argument('--sam2_config', type=str, default='../sam2_configs/sam2_hiera_l.yaml', help='number of support pairs')
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, default='./pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', help="Path to the pretrained DINO checkpoint (.pth). "
                                                            "Use ViT-B/16 checkpoint for --dino_size b, or ViT-S/16 checkpoint for --dino_size s.")
    parser.add_argument("--dino_size", type=str, default="small", choices=["base", "small"], help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = OneModel(args)
    model.eval()

    # Device setup
    model = nn.DataParallel(model).cuda()

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load), strict=False)

    # Helper classes (for testing) initialization
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.shot)

    # Test PATNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.shot)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')