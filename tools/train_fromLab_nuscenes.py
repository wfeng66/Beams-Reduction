import _init_path
import argparse
import datetime
import glob
import os, logger, sys
import numpy as np
from pathlib import Path
from test import repeat_eval_ckpt, eval_single_ckpt 

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from collections import namedtuple
from eval_utils import eval_utils

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment_utils import ExperimentLogger
from utils.dataset_utils import create_fold_split, split_dataset
import pandas as pd



def log_metrics_to_file(result_dict, csv_file, args):
    """
    Log metrics to a CSV file
    """
    import os
    import csv
    
    # Define metadata for the experiment
    metrics_to_save = {
        'method': 'VoxelNeXt',
        'dataset': 'NuScenesDataset' if 'nuscenes' in args.cfg_file.lower() else 'KITTIDataset',
        'dataset_version': args.dataset_version,
        'pretrained': str(args.pretrained_model is not None),
        'train': str(args.train),
        'epochs': str(args.epochs),
    }
    
    # For NuScenes dataset, extract per-class AP@0.5 values
    if 'nuscenes' in args.cfg_file.lower():
        # Define NuScenes classes
        classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                  'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        
        # Add AP values for each class
        for cls_name in classes:
            key = f'{cls_name}_AP@0.5'
            if key in result_dict:
                # The values should already be in [0,1], convert to percentage
                metrics_to_save[key] = str(round(result_dict[key] * 100, 2))
            else:
                metrics_to_save[key] = '0.0'
    else:
        # For KITTI dataset, keep the original metrics
        for key in ['Car_3d/moderate_R40', 'Pedestrian_3d/moderate_R40', 'Cyclist_3d/moderate_R40']:
            metrics_to_save[key] = str(result_dict.get(key, 0.0))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Check if the file exists to determine if we need headers
    file_exists = os.path.isfile(csv_file)
    
    # Write to CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_to_save.keys())
        
        # Write headers if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the metrics
        writer.writerow(metrics_to_save)
    
    print(f"Metrics saved to {csv_file}")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    parser.add_argument('--n_folds', type=int, default=5, help='number of folds for cross-validation')
    # parser.add_argument('--dataset_version', type=str, choices=['I', 'II'], required=True, help='dataset version (I or II)')
    parser.add_argument('--dataset_version', type=str, required=True, help='dataset version (how much damage)')
    parser.add_argument('--experiment_log', type=str, default='experiment_log.csv', help='path to experiment log file')
    parser.add_argument('--train', type=lambda x: (str(x).lower() == 'True'), default=True, help='Set to True to train, False to evaluate')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='hold-out test ratio from training set')
    parser.add_argument('--train_pct', type=float, default=1.0, help='percentage of training set to use')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    # args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    # if args.set_cfgs is not None:
    #     cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def select_percentage(dataset, percentage):
    """Randomly select a percentage of the dataset"""
    if percentage >= 1.0:
        return dataset
    
    num_samples = int(len(dataset) * percentage)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return dataset.select_samples(indices)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # Define output directory first
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create main logger first
    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    # Log basic info
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    

    # load full training set (only KITTI training folder)
    full_train_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs,
        current_fold=None
    )

    # Apply percentage selection if needed
    if float(args.train_pct) < 1.0:
        print(f"Selecting {int(args.train_pct)*100:.1f}% of the data")
        full_train_set = select_percentage(full_train_set, float(args.train_pct))

    total = len(full_train_set)
    indices = np.arange(total)
    np.random.seed(0)
    np.random.shuffle(indices)
    num_test = int(args.test_ratio * total)
    test_idx = indices[:num_test]
    main_idx = indices[num_test:]

    # test_set = full_train_set.select_samples(test_idx)
    # main_set = full_train_set.select_samples(main_idx)

    # # prepare final test loader
    # test_sampler = DistributedSampler(test_set, shuffle=False) if dist_train else None
    # test_loader = DataLoader(
    #     test_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     sampler=test_sampler,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     collate_fn=test_set.collate_batch
    # )

    # cross-validation on main_set
    # make hold-out test and validation sets non-training
    test_set = full_train_set.select_samples(test_idx)
    test_set.training = False
    test_set.data_augmentor = None
    main_set = full_train_set.select_samples(main_idx)

    # prepare final test loader
    test_sampler = DistributedSampler(test_set, shuffle=False) if dist_train else None
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=test_set.collate_batch
    )


    # Store metrics for all folds
    all_fold_metrics = []
    
    if args.train:
        # Cross-validation training and evaluation for all folds
        for fold in range(args.n_folds):
            print(f"\n{'='*20} Fold {fold+1}/{args.n_folds} {'='*20}\n")
            
            
            # Set fold-specific output directories
            fold_output_dir = output_dir / f'fold_{fold}'
            fold_ckpt_dir = fold_output_dir / 'ckpt'
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Create fold-specific logger
            fold_log_file = fold_output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
            fold_logger = common_utils.create_logger(fold_log_file, rank=cfg.LOCAL_RANK)
            fold_logger.info(f"{'='*20} Fold {fold+1}/{args.n_folds} {'='*20}")
            
            if cfg.LOCAL_RANK == 0:
                os.system('cp %s %s' % (args.cfg_file, fold_output_dir))

            tb_log = SummaryWriter(log_dir=str(fold_output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

            fold_logger.info("----------- Create dataloader & network & optimizer -----------")
            # train_set, train_loader, train_sampler = build_dataloader(
            #     dataset_cfg=cfg.DATA_CONFIG,
            #     class_names=cfg.CLASS_NAMES,
            #     batch_size=args.batch_size,
            #     dist=dist_train, 
            #     workers=args.workers,
            #     logger=fold_logger,  # Use fold-specific logger
            #     training=True,
            #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            #     total_epochs=args.epochs,
            #     seed=666 if args.fix_random_seed else None,
            #     current_fold=fold
            # )

            # val_set, val_loader, val_sampler = build_dataloader(
            #     dataset_cfg=cfg.DATA_CONFIG,
            #     class_names=cfg.CLASS_NAMES,
            #     batch_size=args.batch_size,
            #     dist=dist_train,
            #     workers=args.workers,
            #     logger=fold_logger,
            #     training=False,
            #     merge_all_iters_to_one_epoch=False,
            #     total_epochs=args.epochs,
            #     current_fold=fold
            # )

            # test_set, test_loader, test_sampler = build_dataloader(
            #     dataset_cfg=cfg.DATA_CONFIG,
            #     class_names=cfg.CLASS_NAMES,
            #     batch_size=args.batch_size,
            #     dist=dist_train,
            #     workers=args.workers,
            #     logger=fold_logger,
            #     training=False,
            #     merge_all_iters_to_one_epoch=False,
            #     total_epochs=args.epochs,
            #     current_fold=None
            # )

            # indices for this fold
            train_idx, val_idx = create_fold_split(len(main_set), args.n_folds, fold)
            train_set = main_set.select_samples(train_idx)
            val_set = main_set.select_samples(val_idx)
            # disable augmentation and mark as eval mode on validation
            val_set.training = False
            val_set.data_augmentor = None

            # dataloaders
            train_sampler = DistributedSampler(train_set) if dist_train else None
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.workers,
                pin_memory=True,
                collate_fn=train_set.collate_batch
            )
            val_sampler = DistributedSampler(val_set, shuffle=False) if dist_train else None
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=args.workers,
                pin_memory=True,
                collate_fn=val_set.collate_batch
            )

            print(f"Train set size: {len(train_set)}")
            print(f"Val set size: {len(val_set)}")
            print(f"Train loader batches: {len(train_loader)}")
            print(f"Val loader batches: {len(val_loader)}")

            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()

            optimizer = build_optimizer(model, cfg.OPTIMIZATION)

            # load checkpoint if it is possible
            start_epoch = it = 0
            last_epoch = -1
            # if args.pretrained_model is not None:
            #     model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=fold_logger)

            # if args.ckpt is not None:
            #     it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=fold_logger)
            #     last_epoch = start_epoch + 1
            # else:
            #     ckpt_list = glob.glob(str(fold_ckpt_dir / '*.pth'))
                      
            #     if len(ckpt_list) > 0:
            #         ckpt_list.sort(key=os.path.getmtime)
            #         while len(ckpt_list) > 0:
            #             try:
            #                 it, start_epoch = model.load_params_with_optimizer(
            #                     ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=fold_logger
            #                 )
            #                 last_epoch = start_epoch + 1
            #                 break
            #             except:
            #                 ckpt_list = ckpt_list[:-1]

            model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
            if dist_train:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
            fold_logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
            fold_logger.info(model)

            # Create schedulers for this fold
            lr_scheduler, lr_warmup_scheduler = build_scheduler(
                optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
                last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
            )

            # -----------------------start training---------------------------
            fold_logger.info('**********************Start training %s/%s(%s) FOLD %d**********************'
                        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag, fold))

            best_metric = None

            # fold_logger.info(f"Training epoch {epoch + 1}/{args.epochs}")
            
            train_model(
                model,
                optimizer,
                train_loader,
                val_loader,
                model_func=model_fn_decorator(),
                lr_scheduler=lr_scheduler,
                optim_cfg=cfg.OPTIMIZATION,
                start_epoch=start_epoch,
                total_epochs=args.epochs,
                start_iter=it,
                rank=cfg.LOCAL_RANK,
                tb_log=tb_log,
                ckpt_save_dir=fold_ckpt_dir,
                train_sampler=train_sampler,
                lr_warmup_scheduler=lr_warmup_scheduler,
                ckpt_save_interval=args.ckpt_save_interval,
                max_ckpt_save_num=args.max_ckpt_save_num,
                merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
                logger=fold_logger,
                logger_iter_interval=args.logger_iter_interval,
                ckpt_save_time_interval=args.ckpt_save_time_interval,
                use_logger_to_record=args.use_tqdm_to_record,
                show_gpu_stat=not args.wo_gpu_stat,
                use_amp=args.use_amp,
                cfg=cfg
            )

            # val_metric = eval_utils.eval_one_epoch(
            #     cfg, args, model, val_loader, epoch_id=epoch, logger=fold_logger, result_dir=fold_output_dir / 'eval'
            # )

            # mean_ap = np.mean([
            #     val_metric['Car_3d/moderate_R40'],
            #     val_metric['Pedestrian_3d/moderate_R40'],
            #     val_metric['Cyclist_3d/moderate_R40'],
            # ])
            # if (best_metric is None) or (mean_ap > best_metric):
            #     best_metric = mean_ap
            #     torch.save(model.state_dict(), str(fold_ckpt_dir / 'best_model.pth'))
            #     fold_logger.info(f"New best model saved at epoch {epoch} with mean 3D AP {mean_ap:.4f}")

            # Clean up for next fold
            if tb_log is not None:
                tb_log.close()
            
            # After training each fold, check if `best_model.pth` exists and evaluate
            best_model_path = fold_ckpt_dir / 'best_model.pth'
            if os.path.exists(best_model_path):
                # Rebuild the model and load the best weights
                print("="*40, "Final Evaluation!!!", best_model_path)
                model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
                model.load_state_dict(torch.load(str(best_model_path)))
                model.cuda()
                model.eval()

                # Evaluate on validation or test set
                val_metric = eval_utils.eval_one_epoch(
                    cfg, args, model, test_loader, epoch_id='best', logger=fold_logger, result_dir=fold_output_dir / 'eval'
                )

                print("="*100, "val_metric:", sep="\n")
                print(val_metric)
                fold_logger.info(f"Final evaluation of best model: {val_metric}")
                all_fold_metrics.append(val_metric)
            else:
                fold_logger.warning(f"No best_model.pth found in {fold_ckpt_dir}, skipping final evaluation.")
                all_fold_metrics.append({})

            print(f"Train set size: {len(train_set)}")
            print(f"Val set size: {len(val_set)}")
            print(f"Train loader batches: {len(train_loader)}")
            print(f"Val loader batches: {len(val_loader)}")
    else:
        # Only evaluate pretrained model on test set ONCE
        test_set, test_loader, test_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train,
            workers=args.workers,
            logger=logger,
            training=False,
            merge_all_iters_to_one_epoch=False,
            total_epochs=args.epochs,
            current_fold=None
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
        model.cuda()
        model.eval()

        test_metric = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, epoch_id='pretrained', logger=logger, result_dir=output_dir / 'eval'
        )
    
    # Create output directory and csv file path
    if 'output_dir' in args:
        output_dir = args.output_dir
    else:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'nuscenes_metrics.csv')

    # After obtaining the metrics from all folds, calculate average metrics
    if args.train:
        if all_fold_metrics:
            # Calculate average metrics across all folds
            # First identify all metric keys across all folds
            all_keys = set()
            for fold_metric in all_fold_metrics:
                all_keys.update(fold_metric.keys())
            
            # Calculate averages for each metric
            avg_metrics = {}
            for key in all_keys:
                # Only consider folds that have this metric
                valid_metrics = [fm[key] for fm in all_fold_metrics if key in fm]
                if valid_metrics:
                    avg_metrics[key] = sum(valid_metrics) / len(valid_metrics)
            
            # Log the average metrics
            logger.info("Logging average metrics across all folds")
            log_metrics_to_file(avg_metrics, csv_file, args)
        else:
            logger.warning("No metrics available from any fold. Skipping metrics logging.")
    else:
        # In evaluation mode, we already have test_metric
        log_metrics_to_file(test_metric, csv_file, args)



if __name__ == '__main__':
    main()
