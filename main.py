import os
import torch
import pandas as pd
import numpy as np
import sys
import logging as lg
import datetime as dt
import random as r
import ssl
import wandb
ssl._create_default_https_context = ssl._create_unverified_context

from src.utils.data import get_loaders
from src.utils import name_match
from src.utils.early_stopping import EarlyStopper
from config.parser import Parser
import warnings
warnings.filterwarnings("ignore")

def main():
    runs_accs = []
    runs_fgts = []
    
    parser = Parser()
    args = parser.parse()
    
    cf = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch = lg.StreamHandler()
    
    for run_id in range(args.start_seed, args.start_seed + args.n_runs):
        # Re-parse tag. Useful when using multiple runs.
        args = parser.parse()
        args.run_id = run_id

        if args.sweep:
            wandb.init()
            for key in wandb.config.keys():
                setattr(args, key, wandb.config[key])

        # Seed initilization
        if args.n_runs > 1: args.seed = run_id
        np.random.seed(args.seed)
        r.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available() and args.learner == 'OnProDML':
            torch.cuda.manual_seed(args.seed)
        elif torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Learner
        if args.learner is not None:
            learner = name_match.learners[args.learner](args)
            if args.resume: learner.resume(args.model_state, args.buffer_state)
        else:
            raise Warning("Please select the desired learner.")
        
        # logs
        # Define logger and timstamp
        logfile = f'{args.tag}.log'
        if not os.path.exists(args.logs_root): os.mkdir(args.logs_root)

        ff = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logger = lg.getLogger()
        fh = lg.FileHandler(os.path.join(args.logs_root, logfile))
        ch.setFormatter(cf)
        fh.setFormatter(ff)
        logger.addHandler(fh)
        logger.addHandler(ch)
        if args.verbose:
            logger.setLevel(lg.DEBUG)
            logger.warning("Running in VERBOSE MODE.")
        else:
            logger.setLevel(lg.INFO)

        lg.info("=" * 60)
        lg.info("=" * 20 + f"RUN N°{run_id} SEED {args.seed}" + "=" * 20)
        lg.info("=" * 60)        
        lg.info("Parameters used for this training")
        lg.info("=" * 20)
        lg.info(args)

        # Dataloaders
        dataloaders = get_loaders(args)

        # wandb initilization
        if not args.no_wandb and not args.sweep:
            wandb.init(
                project=f"{args.learner}",
                config=args.__dict__
            )
            
        # Training
        # Class incremental training
        if args.training_type == 'inc':
            for task_id in range(args.n_tasks):
                task_name = f"train{task_id}"
                if args.train:
                    learner.train(
                        dataloader=dataloaders[task_name],
                        task_name=task_name,
                        task_id=task_id,
                        dataloaders=dataloaders
                        )
                else:
                    model_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/ckpt_train{task_id}.pth")
                    mem_idx = int(len(dataloaders['train']) * args.batch_size / args.n_tasks) * (task_id + 1)
                    buffer_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/memory_{mem_idx}.pkl")
                    learner.resume(model_state, buffer_state)
                learner.before_eval()

                if args.learner == 'ERCCLDC' or args.learner == 'OnProCCLDC' or args.learner == 'OCMCCLDC' \
                        or args.learner == 'GSACCLDC' or args.learner == 'DERPPCCLDC' or args.learner == 'ERACECCLDC':
                    ens_acc, ens_fgt, n1_acc, n1_fgt, n2_acc, n2_fgt = learner.evaluate(dataloaders, task_id)
                    ncm_ens_acc, ncm_ens_fgt, ncm_n1_acc, ncm_n1_fgt, ncm_n2_acc, ncm_n2_fgt = learner.evaluate_clustering(dataloaders, task_id)
                    avg_acc = ens_acc
                    avg_fgt = ens_fgt
                    if not args.no_wandb:
                        wandb.log({
                            "ens_avg_acc": ens_acc,
                            "ens_avg_fgt": ens_fgt,
                            "n1_avg_acc": n1_acc,
                            "n1_avg_fgt": n1_fgt,
                            "n2_avg_acc": n2_acc,
                            "n2_avg_fgt": n2_fgt,
                            "ncm_ens_avg_acc": ncm_ens_acc,
                            "ncm_ens_avg_fgt": ncm_ens_fgt,
                            "ncm_n1_avg_acc": ncm_n1_acc,
                            "ncm_n1_avg_fgt": ncm_n1_fgt,
                            "ncm_n2_avg_acc": ncm_n2_acc,
                            "ncm_n2_avg_fgt": ncm_n2_fgt,
                            "task_id": task_id
                        })
                        if args.wandb_watch:
                            wandb.watch(learner.model, learner.criterion, log="all", log_freq=1)
                else:
                    avg_acc, avg_fgt = learner.evaluate(dataloaders, task_id)
                    ncm_avg_acc, ncm_avg_fgt = learner.evaluate_clustering(dataloaders, task_id)
                    if not args.no_wandb:
                        wandb.log({
                            "avg_acc": avg_acc,
                            "avg_fgt": avg_fgt,
                            "ncm_avg_acc": ncm_avg_acc,
                            "ncm_avg_fgt": ncm_avg_fgt,
                            "task_id": task_id
                        })
                        if args.wandb_watch:
                            wandb.watch(learner.model, learner.criterion, log="all", log_freq=1)
                learner.after_eval()
            learner.save_results()
        # Uniform training (offline)
        elif args.training_type == 'uni':
            # early_stopper = EarlyStopper(patience=args.es_patience, min_delta=args.es_delta)
            for e in range(args.epochs):
                learner.train(dataloaders['train'], epoch=e)
                avg_acc = learner.evaluate_offline(dataloaders, epoch=e)
                avg_fgt = 0
                # if early_stopper.early_stop(avg_acc):
                #     break
                if not args.no_wandb:
                    wandb.log({
                            "Accuracy": avg_acc,
                            "loss": learner.loss
                        })
            learner.save_results_offline()
        runs_accs.append(avg_acc)
        runs_fgts.append(avg_fgt)

        # compute plasticity/stability metrics
        if not args.no_wandb:
            bt = learner.backward_transfer()
            rf = learner.reletive_forgetting()
            la = learner.learning_accuracy()
            wandb.log(
                {
                    "backward_transfer": bt,
                    "reletive_forgetting": rf,
                    "learning_accuracy": la
                }
            )
        if not args.no_wandb:
            wandb.finish()

        # save model and buffer to "./checkpoints"
        path = os.path.join('./checkpoints', args.dataset, args.learner, str(args.mem_size), str(args.seed))
        learner.save(path)
    
    # Exits the program
    sys.exit(0)


if __name__ == '__main__':
    main()
  