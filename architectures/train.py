import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import ignite
import ignite.distributed as idist
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine import create_supervised_evaluator, Engine, State, Events, _prepare_batch
from ignite.metrics import Loss, Accuracy, TopKCategoricalAccuracy, RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed

from .utils import get_train_eval_datasets, get_model_skeleton

# Deterministic experiment ####################################################################################
seed = 123

manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

###############################################################################################################

def get_save_handler(config):
    title = config['experiment']['title']
    ID = str(config['experiment']['ID']).zfill(2)
    
    return DiskSaver('experiments/{0}/{1}/checkpoints'.format(title, ID), require_empty=True, create_dir=True)



def get_tb_logger(config):
    title = config['experiment']['title']
    ID = str(config['experiment']['ID']).zfill(2)
    
    return SummaryWriter('experiments/{0}/{1}/tensorboards'.format(title, ID))


def get_dataloader(config):
    if idist.get_rank() > 0:
        idist.barrier()
    
    train_dataset, eval_dataset = get_train_eval_datasets(config['dataloader']['dataset'])
    
    if idist.get_rank() == 0:
        idist.barrier()
        
    train_dataloader = idist.auto_dataloader(train_dataset, **config['dataloader']['train_loader_params'])
    eval_dataloader = idist.auto_dataloader(eval_dataset, **config['dataloader']['eval_loader_params'])
    
    return {'train': train_dataloader, "eval": eval_dataloader}



def create_trainer(model, optimizer, loss_function, scheduler, config): 
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=idist.device(), non_blocking=True)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        if config['gradient_clipping'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping'])
        optimizer.step()
        
        return {'loss': loss.detach(), 'y_pred': y_pred, 'y': y}
    
    trainer = DeterministicEngine(_update)
    
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'Train/Loss')
    RunningAverage(Accuracy(output_transform=lambda x: [x['y_pred'], x['y']])).attach(trainer, 'Train/Top-1')
    RunningAverage(TopKCategoricalAccuracy(k=5, output_transform=lambda x: [x['y_pred'], x['y']])).attach(trainer, 'Train/Top-5')
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def schedule(engine):
        scheduler.step()
            
    if config['resume_from'] is not None:
        checkpoint_fp = Path(config['resume_from'])
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location='cpu')
        Checkpoint.load_objects(to_load={"trainer": trainer, "model": model, "optimizer": optimizer},
                                checkpoint=checkpoint)
        model = model.to(idist.device())
        scheduler.optimizer = optimizer
        scheduler.last_epoch = trainer.state.epoch
                
    return trainer



def create_evaluator(model, config):
    evaluator = create_supervised_evaluator(model, device=idist.device(), non_blocking=True)
    
    Accuracy(output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'Eval/Top-1')
    TopKCategoricalAccuracy(k=5, output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'Eval/Top-5')
        
    return evaluator



def register_tb_logger_handlers(trainer, evaluator, optimizer, tb_logger, config):
    @trainer.on(Events.ITERATION_COMPLETED(every=config['train_constants']['log_train_stats_every_iters']))
    def _log_train_statistics(engine):
        tb_logger.add_scalar('Train/Loss', engine.state.metrics['Train/Loss'], global_step=engine.state.iteration)
        tb_logger.add_scalar('Train/Top-1', engine.state.metrics['Train/Top-1'], global_step=engine.state.iteration)
        tb_logger.add_scalar('Train/Top-5', engine.state.metrics['Train/Top-5'], global_step=engine.state.iteration)
        
    @evaluator.on(Events.COMPLETED)
    def _log_evaluation_statistics(engine):
        tb_logger.add_scalar('Eval/Top-1', engine.state.metrics['Eval/Top-1'], global_step=trainer.state.epoch)
        tb_logger.add_scalar('Eval/Top-5', engine.state.metrics['Eval/Top-5'], global_step=trainer.state.epoch) 
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def _log_train_params(engine):
        tb_logger.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step=engine.state.epoch)
        
        
        
def prepare(config):
    model = get_model_skeleton(config['model'], config['dataloader']['dataset'])
    model = model.to(idist.device())
    #model = DP(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    model = DDP(model, device_ids=[idist.get_local_rank()], find_unused_parameters=True)
    
    optimizer = optim.__dict__[config['optimizer']['type']](
        model.parameters(), **config['optimizer']['params']
    )
    optimizer = idist.auto_optim(optimizer)
    
    loss_function = nn.__dict__[config['loss_function']['type']]().to(idist.device())
    
    scheduler = optim.lr_scheduler.__dict__[config['scheduler']['type']](
        optimizer, **config['scheduler']['params']
    )
    
    return model, optimizer, loss_function, scheduler
        
        
        
def train_process(local_rank, config):
    dataloader = get_dataloader(config)
    model, optimizer, loss_function, scheduler = prepare(config)
    trainer = create_trainer(model, optimizer, loss_function, scheduler, config)
    evaluator = create_evaluator(model, config)    
    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "scheduler": scheduler}
    
    @trainer.on(Events.EPOCH_COMPLETED(every=config['train_constants']['save_every_epochs']) | Events.COMPLETED)
    def evaluate(engine):
        evaluator.run(dataloader['eval'])
        
    checkpoint_handler = Checkpoint(
        to_save=to_save,
        save_handler=get_save_handler(config),
        score_function=lambda engine: engine.state.metrics['Eval/Top-1'],
        score_name='eval_top1_accuracy',
        n_saved=None,
        global_step_transform=global_step_from_engine(trainer)
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)
    
    if idist.get_rank() == 0:
        tb_logger = get_tb_logger(config)
        register_tb_logger_handlers(trainer, evaluator, optimizer, tb_logger, config)
    
    try:
        trainer.run(dataloader['train'], max_epochs=config['train_constants']['max_epochs'])
        evaluator.run(dataloader['eval'])
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
    if idist.get_rank() == 0:
        tb_logger.close()
        
        
        
def run(config):
    #train_process(0, config)
    with idist.Parallel(backend='nccl', nproc_per_node=8) as parallel:
        parallel.run(train_process, config) 