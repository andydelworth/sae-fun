import gc
import math
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from sae import SAE, BatchTopKSAE
from activation_data import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchviz import make_dot


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='sae_out/')
parser.add_argument('--layer_num', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--activation_file', type=str, default='/persist/adelworth/sae-fun/token_activations_500m.h5')
parser.add_argument('--hidden_dim_multiplier', type=int, default=64)
parser.add_argument('--l1_lambda', type=float, default=1e-4)
parser.add_argument('--save_every_n_steps', type=int, default=100000)
parser.add_argument('--validate_every_n_steps', type=int, default=25000)
parser.add_argument('--grad_clip_norm', type=float, default=1.0)
parser.add_argument('--resample_dead', default=False, action='store_true')
parser.add_argument('--sae_type', type=str, default='relu')
parser.add_argument('--init_val', action='store_true', default=True)
args = parser.parse_args()

assert args.sae_type in ('relu', 'batch_top_k')

# Initialize distributed training
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Load model and move to GPU
def load_model(local_rank):
    if args.sae_type == 'relu':
        model = SAE(input_dim=4096, hidden_dim=4096 * args.hidden_dim_multiplier).to(torch.float32)
    elif args.sae_type == 'batch_top_k':
        model = BatchTopKSAE(input_dim=4096, hidden_dim=4096 * args.hidden_dim_multiplier, k=16).to(torch.float32)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        print(f'Model loaded with {sum(p.numel() for p in model.parameters())} parameters')
    return model

def validate(model, validation_loader, local_rank, writer, step_num):
    model.eval()
    val_mse_loss = torch.tensor(0.0, device=local_rank)
    val_l1_loss = torch.tensor(0.0, device=local_rank)
    val_l0_loss = torch.tensor(0.0, device=local_rank)
    total_samples = torch.tensor(0, device=local_rank)
    
    # Track active features using a boolean tensor
    hidden_dim = model.module.hidden_dim
    active_features = torch.zeros(hidden_dim, dtype=torch.bool, device=local_rank)
    
    with torch.no_grad():
        for tqdm_batch in tqdm(validation_loader, desc=f'Validation @ Step {step_num}', disable=local_rank != 0):
            batch = tqdm_batch.to(torch.float32).to(local_rank)
            batch_size = batch.shape[0]
            reconstruction, sparse_representation = model(batch)
            
            # Mean across features, sum across batch
            mse_loss = torch.nn.functional.mse_loss(reconstruction, batch, reduction='none').mean(dim=1).sum()
            # reparam invariant l1 loss
            decoder_column_norms = torch.norm(model.module.decoder[0].weight, dim=0) * sparse_representation
            decoder_invariant_l1_loss = torch.sum(decoder_column_norms * sparse_representation)

            # For L0, count average number of non-zeros per sample
            l0_loss = (sparse_representation != 0).float().mean(dim=1).sum()

            # Track active features
            if sparse_representation.shape[0] > 0:
                active_indices = sparse_representation.nonzero()
                if active_indices.numel() > 0:
                    active_features.index_fill_(0, active_indices[:, 1].unique(), True)
            
            val_mse_loss += mse_loss
            val_l1_loss += decoder_invariant_l1_loss
            val_l0_loss += l0_loss
            total_samples += batch_size
    
    # Gather validation metrics across all processes
    dist.all_reduce(val_mse_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_l1_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_l0_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    # Combine active features across all processes (logical OR)
    dist.all_reduce(active_features.int(), op=dist.ReduceOp.MAX)
    active_features = active_features.bool()
    
    # Count dead features
    dead_feature_count = hidden_dim - active_features.sum().item()
    
    # Calculate validation averages (per sample)
    avg_val_mse = val_mse_loss.item() / total_samples.item() if total_samples.item() > 0 else 0
    avg_val_l1 = val_l1_loss.item() / total_samples.item() if total_samples.item() > 0 else 0
    avg_val_l0 = val_l0_loss.item() / total_samples.item() if total_samples.item() > 0 else 0

    # Log validation metrics only on main process
    if local_rank == 0:
        writer.add_scalar('Loss/val_mse', avg_val_mse, step_num)
        writer.add_scalar('Loss/val_l1', avg_val_l1, step_num)
        writer.add_scalar('Loss/val_l1_times_lambda', avg_val_l1 * args.l1_lambda, step_num)
        writer.add_scalar('Loss/val_total', avg_val_mse + args.l1_lambda * avg_val_l1, step_num)
        writer.add_scalar('Stats/val_l0', avg_val_l0, step_num)
        writer.add_scalar('Stats/val_dead_feature_count', dead_feature_count, step_num)
        writer.add_scalar('Stats/active_feature_percentage', 100 * (hidden_dim - dead_feature_count) / hidden_dim, step_num)
    
    if args.resample_dead and step_num > 0:
        resample_dead_features(model, active_features, local_rank)

    return avg_val_mse, avg_val_l1

def resample_dead_features(model, active_features, local_rank):
    if local_rank != 0:
        return

    with torch.no_grad():
        # encoder: Linear(input_dim, hidden_dim)
        encoder = model.module.encoder[0]
        decoder = model.module.decoder[0]
        input_dim = encoder.weight.shape[1]
        hidden_dim = encoder.weight.shape[0]

        dead_indices = (~active_features).nonzero(as_tuple=False).flatten()
        for i in dead_indices:
            # Resample encoder rows as random interpolations of existing latents
            # This probably isn't ideal. Could be better to init as high-MSE token latents.
            new_latent = torch.mean(torch.randint(low=0, high=hidden_dim, size=(5)))
            encoder.weight[i] = new_latent
            if encoder.bias is not None:
                bound = 1 / math.sqrt(input_dim)
                encoder.bias[i].uniform_(-bound, bound)

            # Set decoder column i = encoder row i^T
            decoder.weight[:, i] = encoder.weight[i].detach().clone()

    # Sync updated weights from rank 0 to all others
    for p in model.module.encoder.parameters():
        dist.broadcast(p.data, src=0)
    for p in model.module.decoder.parameters():
        dist.broadcast(p.data, src=0)

'''
TODO:
- add pre-encoder bias
- neuron resampling
- add normalization of activations?
'''

def train(model, local_rank):
    train_loader, validation_loader = get_data_loaders(args.activation_file, args.layer_num, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f'Starting training on device {local_rank}...')

    # Setup TensorBoard and checkpoint directory only on main process
    if local_rank == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        run_dir = f'{args.output_dir}/{current_time}_lambda_{args.l1_lambda}_hidden_mult_{args.hidden_dim_multiplier}_layernum_{args.layer_num}_lr_{args.lr}'
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(run_dir)
    else:
        writer = None
        run_dir = None

    if args.init_val:
        # Run initial validation
        avg_val_mse, avg_val_l1 = validate(model, validation_loader, local_rank, writer, 0)
        
        # Print initial validation summary only on main process
        if local_rank == 0:
            print(f'Initial validation: Val MSE: {avg_val_mse:.4f}, Val L1: {avg_val_l1:.4f}')
    
    global_step = 0
    
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

            # if step == 100 and local_rank == 0:
            #     make_dot(loss, params=dict(model.named_parameters())).render("loss_graph", format="pdf")
            #     print("Graph saved to loss_graph.pdf")
            
            batch = tqdm_batch.to(torch.float32).to(local_rank)
            reconstruction, sparse_representation = model(batch)

            # if (step + 1) % 1000 == 0 and local_rank == 0:
            #     print(torch.cuda.memory_summary())
            #     print('TENSOR COUNTS:')
            #     print(len([obj for obj in gc.get_objects() if torch.is_tensor(obj)]))
            #     # print('Retained Graph Refs:')
            #     # print([t for t in gc.get_objects().shape if torch.is_tensor(t) and t.grad_fn is not None])
            #     print(sparse_representation.shape, sparse_representation.requires_grad)

            mse_loss = torch.nn.functional.mse_loss(reconstruction, batch)
            # TODO - make this a reparameterisation-invariant L1 penalty
            decoder_column_norms = torch.norm(model.module.decoder[0].weight, dim=0) * sparse_representation
            decoder_invariant_l1_loss = torch.sum(decoder_column_norms * sparse_representation)
            loss = mse_loss
            if args.sae_type == 'relu' or True:
                loss = loss + args.l1_lambda * decoder_invariant_l1_loss
            loss.backward()
            
            # Increment global step counter
            global_step += 1
            
            # Accumulate step-level losses
            step_mse_loss += mse_loss.item()
            step_l1_loss += decoder_invariant_l1_loss.item()
            step_count += 1
            
            # Log every 100 steps
            if (step + 1) % 100 == 0:
                # Gather step metrics across all processes
                dist.all_reduce(step_mse_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_l1_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_count, op=dist.ReduceOp.SUM)
                
                # Calculate step averages
                avg_step_mse = step_mse_loss.item() / step_count.item()
                avg_step_l1 = step_l1_loss.item() / step_count.item()
                
                # Calculate relative reconstruction error
                relative_error = torch.norm(reconstruction - batch) / torch.norm(batch)
                
                # Get weight and gradient statistics
                total_weight_norm = 0
                total_grad_norm = 0

                encoder_norm = math.sqrt(torch.sum(model.module.encoder[0].weight ** 2).item())
                decoder_norm = math.sqrt(torch.sum(model.module.decoder[0].weight ** 2).item())
                weight_norm = math.sqrt(sum(torch.sum(w ** 2).item() for w in model.parameters() if w.grad is not None)) # doesnt do mean/var norms i think? unclear
                grad_norm = math.sqrt(sum(torch.sum(w.grad ** 2).item() for w in model.parameters() if w.grad is not None))
                encoder_grad_norm = math.sqrt(torch.sum(model.module.encoder[0].weight.grad ** 2).item())
                decoder_grad_norm = math.sqrt(torch.sum(model.module.decoder[0].weight.grad ** 2).item())
                
                # Log metrics only on main process
                if local_rank == 0:
                    # Loss metrics
                    writer.add_scalar('Loss/step_mse', avg_step_mse, global_step)
                    writer.add_scalar('Loss/step_l1', avg_step_l1, global_step)
                    writer.add_scalar('Loss/step_total', avg_step_mse + args.l1_lambda * avg_step_l1, global_step)
                    writer.add_scalar('Loss/relative_error', relative_error.item(), global_step)
                    
                    # Parameter statistics
                    writer.add_scalar('Parameters/weight_norm', weight_norm, global_step)
                    writer.add_scalar('Parameters/grad_norm', grad_norm, global_step)
                    writer.add_scalar('Parameters/encoder_norm', encoder_norm, global_step)
                    writer.add_scalar('Parameters/decoder_norm', decoder_norm, global_step)
                    writer.add_scalar('Parameters/encoder_grad_norm', encoder_grad_norm, global_step)
                    writer.add_scalar('Parameters/decoder_grad_norm', decoder_grad_norm, global_step)
                    
                    # Activation statistics
                    # writer.add_histogram('Activations/sparse_representation', sparse_representation.flatten(), global_step)
                    writer.add_scalar('Activations/sparsity_1e-6', (sparse_representation.abs() < 1e-6).float().mean(), global_step)
                    writer.add_scalar('Activations/sparsity_1e-1', (sparse_representation.abs() < 1e-1).float().mean(), global_step)
                    writer.add_scalar('Activations/max_activation', sparse_representation.abs().max(), global_step)
                
                # Reset step accumulators
                step_mse_loss = torch.tensor(0.0, device=local_rank)
                step_l1_loss = torch.tensor(0.0, device=local_rank)
                step_count = torch.tensor(0, device=local_rank)
            
            # Run validation every 100k steps
            if global_step % args.validate_every_n_steps == 0:
                # Ensure all gradients are synced before validation
                if dist.is_initialized():
                    dist.barrier()
                
                # Run validation
                avg_val_mse, avg_val_l1 = validate(model, validation_loader, local_rank, writer, global_step)
                
                # Print validation summary only on main process
                if local_rank == 0:
                    print(f'Step {global_step}: Val MSE: {avg_val_mse:.4f}, Val L1: {avg_val_l1:.4f}')
                
                # Switch back to training mode
                model.train()
            
            # Save checkpoint every 5000 steps
            if global_step % args.save_every_n_steps == 0 and local_rank == 0:
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch_num,
                }
                torch.save(checkpoint, f'{run_dir}/checkpoint_step_{global_step}.pt')
                print(f'Checkpoint saved at step {global_step}')

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)

            if (step + 1) % 100 == 0 and local_rank == 0:
                clipped_grad_norm = math.sqrt(sum(torch.sum(w.grad ** 2).item() for w in model.parameters() if w.grad is not None))
                writer.add_scalar('Parameters/clipped_grad_norm', clipped_grad_norm, global_step)

            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate epoch-level losses
            train_mse_loss += mse_loss.item()
            train_l1_loss += decoder_invariant_l1_loss.item()
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
        
        # Save the final model
        torch.save(model.module.state_dict(), f'{run_dir}/final_model.pth')

def main():
    local_rank = setup_ddp()
    model = load_model(local_rank)
    train(model, local_rank)

if __name__ == '__main__':
    main()