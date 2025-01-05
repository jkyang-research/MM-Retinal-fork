"""
预训练主函数
"""

import argparse
import torch
import os

from keepfit.pretraining.data.dataloader import get_loader
from keepfit.pretraining.data.transforms import augmentations_pretraining
from keepfit.modeling.model import KeepFITModel
from keepfit.modeling.misc import set_seeds

from local_data.constants import *


def process(args):
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')
    torch.distributed.init_process_group(backend='gloo')
    seed = 42
    set_seeds(seed, use_cuda=True)

    # 创建dataloader   数据预处理   合并数据集   返回｛"train": train_loader｝
    # Create dataloader; Data Preprocessing; Merge data set; Return {"train": train_loader}
    dataloaders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                             datasets=args.datasets, balance=args.balance, batch_size=args.batch_size,
                             num_workers=args.num_workers, banned_categories=args.banned_categories,
                             caption=args.caption, augment_description=args.augment_description,knowledge_dict=args.knowledge_dict)

    # 定义模型  define model
    model = KeepFITModel(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=args.load_weights, vision_pretrained=True,
                       weights_path=args.weights_path)
    model.to(device)
    # 并行化  parallelization
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
    #                                                   find_unused_parameters=True)
    model.fit(dataloaders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining,
              local_rank=args.local_rank,knowledge_dict=args.knowledge_dict)


def main():
    parser = argparse.ArgumentParser()

    # 数据相关  Data
    parser.add_argument('--data_root_path', default=PATH_RESIZED_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN)                         # 预处理好的数据pair  Pre-processed data pair
    parser.add_argument('--datasets', default=["39_MM_Retinal_dataset"])
    parser.add_argument('--banned_categories', default=['myopia', 'cataract', 'macular hole', 'retinitis pigmentosa',
                                                        "myopic", "myope", "myop", "retinitis"])                   # OOD实验，删除对应类别  OOD experiment, delete corresponding category
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN+"keepfit/", help='output path')
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]")                    # prompt模板  prompt template
    parser.add_argument('--augment_description', default=True, type=lambda x: (str(x).lower() == 'true'))   # 将类标变为医学描述？  Turn the category into a medical description?
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))        # 是否平衡数据集？ 两种dataloader  data set balanced?

    # 模型架构  model architecture
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v1 -- efficientnet')      # img tower

    # 训练超参数  Training hyperparameter
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--weights_path', default=None)                         # 本地模型权重  model weight
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))  # 是否加载预训练权重  load the pre-training weight？
    parser.add_argument('--knowledge_dict', default=True, type=lambda x: (str(x).lower() == 'true'))  # knowledge_dict

    # 设备  Device
    parser.add_argument('--store_num', default=1, type=int)                                         # 存储间隔
    parser.add_argument('--num_workers', default=8, type=int, help='workers number for DataLoader')
    parser.add_argument("--local_rank", type=int, default=-1)


    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"     
    os.environ["USE_LIBUV"] = "0"                                                                
    main()