import os
import argparse
import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from models.matgtm_bgd import MatGTM
from utils.stopatepoch import StopAtEpoch
from utils.bdg_multitrends import ZeroShotDataset
import warnings
# from utils.time_flops import TimeFLOPsCallback
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    print(args)
    
    wandb.login(key=args.wandb_api_key, relogin=True, force=True)
    torch.autograd.set_detect_anomaly(True)
 
    pl.seed_everything(args.seed)

    # Load sales data
    train_df = pd.read_csv(Path(args.data_folder + 'train.csv'), parse_dates=['release_date'])
    test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])

    # Load category and color encodings
    cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'))
    col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'))
    fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)

    train_loader = ZeroShotDataset(args, train_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict,
                                   fab_dict, args.trend_len)\
                    .get_loader(batch_size=args.batch_size, train=True,num_workers=os.cpu_count())
    test_loader = ZeroShotDataset(args, test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict,
                                  fab_dict, args.trend_len)\
                    .get_loader(batch_size=1, train=False,num_workers=os.cpu_count())
    
    
    model = MatGTM(
        args=args,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_attn_heads,
        enc_num_layers=args.num_enc_layers,
        dec_num_layers=args.num_hidden_layers,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        granularity_scales=args.granularity_scales,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
        weight_init=args.weight_init,
        gpu_num=args.gpu_num,
        gaf=args.gaf,
        gaf_image_size=args.gaf_image_size, 
        visionencoder_out_dim=args.visionencoder_out_dim,
        gaf_method=args.gaf_method
    )
    
    # progress_bar = TQDMProgressBar(refresh_rate=10)
    
    existing_run_id = args.wandb_run_id
    
    wandb.init(entity=args.wandb_entity,
              project=args.wandb_proj, 
              id=existing_run_id,
              config=args, name=args.wandb_run,
              resume=args.wandb_resume,
              settings=wandb.Settings(_service_wait=600))
              
    wandb_logger = pl_loggers.WandbLogger()
    wandb_logger.watch(model)
    
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_savename = args.model_type
    
    # time_flops_callback = TimeFLOPsCallback(
    #     profile_activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_flops=True
    # )

    epoch_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/'+args.model_type,
        filename=model_savename+"---{epoch}---"+dt_string,
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True
        )
    
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/'+args.model_type,
        filename=model_savename+"best-{epoch}-{valid_mae:.2f}---"+dt_string,
        monitor='valid_mae',
        mode='min',
        save_top_k=1,
        )
        
    stop_at_200 = StopAtEpoch(stop_epoch=200)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=[args.gpu_num],
                         max_epochs=args.epochs,
                         check_val_every_n_epoch=5,
                         num_sanity_val_steps=2,
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         callbacks=[epoch_checkpoint_callback,
                         best_checkpoint_callback,
                        #  stop_at_200
                        ])
    
    print('current run folder :',os.getcwd())
    
    ckpt_path = args.ckpt_path
    if ckpt_path is not None:
        if os.path.exists(ckpt_path):
            print(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found. Starting training from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")
        ckpt_path = None
        
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader,
                ckpt_path=ckpt_path)

def parse_granularity_scales(granularity_scales_str):
    return [float(x) for x in granularity_scales_str.split()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot forecasting')
    
    parser.add_argument('--data_folder', type=str, default='' , help='path of dataset folder')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--granularity_scales',type=parse_granularity_scales, default=[1, 1/2, 1/4], 
                    help="List of granularity scales for each layer. Example: '1 0.5 0.25'")
    
    parser.add_argument('--M',type=int,default=52,help='Max size of Latent Query Tokens')
    parser.add_argument('--gaf',type=bool,default=False)
    parser.add_argument('--gaf_image_size',type=float,default=1.0)
    parser.add_argument('--visionencoder_out_dim',type=int,default=768)
    parser.add_argument('--num_enc_layers',type=int,default=2)
    parser.add_argument('--gaf_method',type=str,default='difference')
    parser.add_argument('--weight_init',type=str,default='xavier')
    
    parser.add_argument('--model_type', type=str, default='Matgtm')
    parser.add_argument('--wandb_api_key', type=str, default='',help='wandb api key')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_proj', type=str, default='')
    parser.add_argument('--wandb_run', type=str, default='')
    parser.add_argument('--wandb_run_id',type=str, default=None)
    parser.add_argument('--wandb_resume',type=str,default='allow')
    parser.add_argument('--ckpt_path',type=str,default=None)
    
    args = parser.parse_args()
    run(args)



