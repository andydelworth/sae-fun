from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from sae import SAE
from activation_data import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='sae_out/')
parser.add_argument('--layer_num', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--activation_file', type=str, default='/persist/adelworth/sae-fun/activations.h5')
parser.add_argument('--hidden_dim_multiplier', type=int, default=4)
parser.add_argument('--l1_lambda', type=float, default=5.0)
args = parser.parse_args()

# Initialize distributed training
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Load model and move to GPU
def load_model(local_rank):
    model = SAE(input_dim=4096, hidden_dim=4096 * args.hidden_dim_multiplier).to(torch.bfloat16)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        print(f'Model loaded with {sum(p.numel() for p in model.parameters())} parameters')
    return model

def validate(model, validation_loader, local_rank, writer, epoch_num):
    model.eval()
    val_mse_loss = torch.tensor(0.0, device=local_rank)
    val_l1_loss = torch.tensor(0.0, device=local_rank)
    num_val_batches = torch.tensor(0, device=local_rank)
    
    with torch.no_grad():
        for tqdm_batch in tqdm(validation_loader, desc=f'Validation Epoch {epoch_num}', disable=local_rank != 0):
            batch = tqdm_batch.to(torch.bfloat16).to(local_rank)
            reconstruction, sparse_representation = model(batch)
            mse_loss = torch.nn.functional.mse_loss(reconstruction, batch)
            l1_loss = torch.nn.functional.l1_loss(sparse_representation, torch.zeros_like(sparse_representation))
            
            val_mse_loss += mse_loss.item()
            val_l1_loss += l1_loss.item()
            num_val_batches += 1
    
    # Gather validation metrics across all processes
    dist.all_reduce(val_mse_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_l1_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_val_batches, op=dist.ReduceOp.SUM)
    
    # Calculate validation averages
    avg_val_mse = val_mse_loss.item() / num_val_batches.item()
    avg_val_l1 = val_l1_loss.item() / num_val_batches.item()
    
    # Log validation metrics only on main process
    if local_rank == 0:
        writer.add_scalar('Loss/val_mse', avg_val_mse, epoch_num)
        writer.add_scalar('Loss/val_l1', avg_val_l1, epoch_num)
        writer.add_scalar('Loss/val_total', avg_val_mse + args.l1_lambda * avg_val_l1, epoch_num)
    
    return avg_val_mse, avg_val_l1

def train(model, local_rank):
    train_loader, validation_loader = get_data_loaders(args.activation_file, args.layer_num, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f'Starting training on device {local_rank}...')

    # Setup TensorBoard only on main process
    if local_rank == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        writer = SummaryWriter(f'{args.output_dir}/{current_time}_lambda_{args.l1_lambda}_hidden_mult_{args.hidden_dim_multiplier}_layernum_{args.layer_num}_lr_{args.lr}')
    else:
        writer = None

    # Run validation
    avg_val_mse, avg_val_l1 = validate(model, validation_loader, local_rank, writer, 0)
    
    # Print epoch summary only on main process
    if local_rank == 0:
        print(f'Epoch 0: Val MSE: {avg_val_mse:.4f}, Val L1: {avg_val_l1:.4f}')
    
    for epoch_num in range(args.num_epochs):
        dist.barrier()
        model.train()
        train_mse_loss = torch.tensor(0.0, device=local_rank)
        train_l1_loss = torch.tensor(0.0, device=local_rank)
        num_batches = torch.tensor(0, device=local_rank)
        
        # Step-level loss tracking
        step_mse_loss = torch.tensor(0.0, device=local_rank)
        step_l1_loss = torch.tensor(0.0, device=local_rank)
        step_count = torch.tensor(0, device=local_rank)
        
        for step, tqdm_batch in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch_num}', disable=local_rank != 0)):
            batch = tqdm_batch.to(torch.bfloat16).to(local_rank)
            reconstruction, sparse_representation = model(batch)
            mse_loss = torch.nn.functional.mse_loss(reconstruction, batch)
            l1_loss = torch.nn.functional.l1_loss(sparse_representation, torch.zeros_like(sparse_representation))
            loss = mse_loss + args.l1_lambda * l1_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate step-level losses
            step_mse_loss += mse_loss.item()
            step_l1_loss += l1_loss.item()
            step_count += 1
            
            # Log every 10 steps
            if (step + 1) % 10 == 0:
                # Gather step metrics across all processes
                dist.all_reduce(step_mse_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_l1_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_count, op=dist.ReduceOp.SUM)
                
                # Calculate step averages
                avg_step_mse = step_mse_loss.item() / step_count.item()
                avg_step_l1 = step_l1_loss.item() / step_count.item()
                
                # Log step metrics only on main process
                if local_rank == 0:
                    writer.add_scalar('Loss/step_mse', avg_step_mse, epoch_num * len(train_loader) + step)
                    writer.add_scalar('Loss/step_l1', avg_step_l1, epoch_num * len(train_loader) + step)
                    writer.add_scalar('Loss/step_total', avg_step_mse + args.l1_lambda * avg_step_l1, epoch_num * len(train_loader) + step)
                
                # Reset step accumulators
                step_mse_loss = torch.tensor(0.0, device=local_rank)
                step_l1_loss = torch.tensor(0.0, device=local_rank)
                step_count = torch.tensor(0, device=local_rank)
            
            # Accumulate epoch-level losses
            train_mse_loss += mse_loss.item()
            train_l1_loss += l1_loss.item()
            num_batches += 1
        
        # Gather metrics across all processes
        dist.all_reduce(train_mse_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_l1_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        
        # Calculate averages
        avg_train_mse = train_mse_loss.item() / num_batches.item()
        avg_train_l1 = train_l1_loss.item() / num_batches.item()
        
        # Log training metrics only on main process
        if local_rank == 0:
            writer.add_scalar('Loss/train_mse', avg_train_mse, epoch_num)
            writer.add_scalar('Loss/train_l1', avg_train_l1, epoch_num)
            writer.add_scalar('Loss/train_total', avg_train_mse + args.l1_lambda * avg_train_l1, epoch_num)
        
        # Run validation
        avg_val_mse, avg_val_l1 = validate(model, validation_loader, local_rank, writer, epoch_num + 1)
        
        # Print epoch summary only on main process
        if local_rank == 0:
            print(f'Epoch {epoch_num}: Train MSE: {avg_train_mse:.4f}, Train L1: {avg_train_l1:.4f}, '
                  f'Val MSE: {avg_val_mse:.4f}, Val L1: {avg_val_l1:.4f}')
    
    if local_rank == 0:
        writer.close()
    
    if local_rank == 0:
        torch.save(model.module.state_dict(), f'{args.output_dir}/{current_time}_lambda_{args.l1_lambda}_hidden_mult_{args.hidden_dim_multiplier}_layernum_{args.layer_num}_lr_{args.lr}/final_model.pth')

def main():
    local_rank = setup_ddp()
    model = load_model(local_rank)
    train(model, local_rank)

if __name__ == '__main__':
    main()