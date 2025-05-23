import math
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

# Distributed training setup

def setup_ddp():
    """Initialize distributed training and return local rank."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def load_model(local_rank, args):
    """Create SAE model and wrap with DDP."""
    model = SAE(input_dim=4096, hidden_dim=4096 * args.hidden_dim_multiplier).to(torch.float32)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        print(f'Model loaded with {sum(p.numel() for p in model.parameters())} parameters')
    return model


def resample_dead_features(model, active_features, local_rank):
    """Resample dead features in the SAE model (rank 0 only)."""
    if local_rank != 0:
        return
    with torch.no_grad():
        encoder = model.module.encoder[0]
        decoder = model.module.decoder[0]
        input_dim = encoder.weight.shape[1]
        hidden_dim = encoder.weight.shape[0]
        dead_indices = (~active_features).nonzero(as_tuple=False).flatten()
        for i in dead_indices:
            new_latent = torch.mean(torch.randint(low=0, high=hidden_dim, size=(5,)))
            encoder.weight[i] = new_latent
            if encoder.bias is not None:
                bound = 1 / math.sqrt(input_dim)
                encoder.bias[i].uniform_(-bound, bound)
            decoder.weight[:, i] = encoder.weight[i].detach().clone()
    for p in model.module.encoder.parameters():
        dist.broadcast(p.data, src=0)
    for p in model.module.decoder.parameters():
        dist.broadcast(p.data, src=0)


def compute_losses_and_stats(model, batch, args):
    """Compute MSE, L1, L0 losses and active indices for a batch."""
    reconstruction, sparse_representation = model(batch)
    mse_loss = torch.nn.functional.mse_loss(reconstruction, batch, reduction='none').mean(dim=1).sum() if args.get('reduction_none', False) else torch.nn.functional.mse_loss(reconstruction, batch)
    decoder_column_norms = torch.norm(model.module.decoder[0].weight, dim=0) * sparse_representation
    decoder_invariant_l1_loss = torch.sum(decoder_column_norms * sparse_representation)
    l0_loss = (sparse_representation != 0).float().mean(dim=1).sum() if args.get('reduction_none', False) else (sparse_representation != 0).float().mean()
    active_indices = sparse_representation.nonzero()
    return mse_loss, decoder_invariant_l1_loss, l0_loss, active_indices, reconstruction, sparse_representation


def log_metrics(writer, metrics_dict, global_step, prefix=""):
    """Log a dictionary of metrics to TensorBoard."""
    if writer is None:
        return
    for key, value in metrics_dict.items():
        writer.add_scalar(f"{prefix}{key}", value, global_step)


def all_reduce_tensors(tensors, op=dist.ReduceOp.SUM):
    """All-reduce a list of tensors across distributed processes."""
    for t in tensors:
        dist.all_reduce(t, op=op)


def run_epoch(model, dataloader, local_rank, args, writer=None, global_step=0, train_mode=True, step_callback=None, epoch_num=None, run_dir=None, optimizer=None):
    """Run one epoch of training or validation.\nNote: PyTorch DataLoader does not support resuming mid-epoch; only epoch-level resuming is possible."""
    if train_mode:
        model.train()
    else:
        model.eval()
    mse_loss_total = torch.tensor(0.0, device=local_rank)
    l1_loss_total = torch.tensor(0.0, device=local_rank)
    l0_loss_total = torch.tensor(0.0, device=local_rank)
    total_samples = torch.tensor(0, device=local_rank)
    hidden_dim = model.module.hidden_dim
    active_features = torch.zeros(hidden_dim, dtype=torch.bool, device=local_rank)
    step_count = torch.tensor(0, device=local_rank)
    for step, tqdm_batch in enumerate(tqdm(dataloader, desc=('Train' if train_mode else 'Validation'), disable=local_rank != 0)):
        batch = tqdm_batch.to(torch.float32).to(local_rank)
        args_dict = vars(args) if hasattr(args, '__dict__') else args
        args_dict['reduction_none'] = not train_mode
        mse_loss, decoder_invariant_l1_loss, l0_loss, active_indices, reconstruction, sparse_representation = compute_losses_and_stats(model, batch, args_dict)
        batch_size = batch.shape[0]
        if sparse_representation.shape[0] > 0 and active_indices.numel() > 0:
            active_features.index_fill_(0, active_indices[:, 1].unique(), True)
        mse_loss_total += mse_loss
        l1_loss_total += decoder_invariant_l1_loss
        l0_loss_total += l0_loss
        total_samples += batch_size
        step_count += 1
        if train_mode:
            loss = mse_loss
            if args.sae_type == 'relu' or True:
                loss = loss + args.l1_lambda * decoder_invariant_l1_loss
            loss.backward()
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()
        if (step + 1) % 100 == 0 and writer is not None and local_rank == 0:
            avg_step_mse = mse_loss_total.item() / step_count.item()
            avg_step_l1 = l1_loss_total.item() / step_count.item()
            relative_error = torch.norm(reconstruction - batch) / torch.norm(batch)
            encoder_norm = math.sqrt(torch.sum(model.module.encoder[0].weight ** 2).item())
            decoder_norm = math.sqrt(torch.sum(model.module.decoder[0].weight ** 2).item())
            weight_norm = math.sqrt(sum(torch.sum(w ** 2).item() for w in model.parameters() if w.grad is not None))
            grad_norm = math.sqrt(sum(torch.sum(w.grad ** 2).item() for w in model.parameters() if w.grad is not None))
            encoder_grad_norm = math.sqrt(torch.sum(model.module.encoder[0].weight.grad ** 2).item()) if model.module.encoder[0].weight.grad is not None else 0.0
            decoder_grad_norm = math.sqrt(torch.sum(model.module.decoder[0].weight.grad ** 2).item()) if model.module.decoder[0].weight.grad is not None else 0.0
            metrics = {
                'Loss/step_mse': avg_step_mse,
                'Loss/step_l1': avg_step_l1,
                'Loss/step_total': avg_step_mse + args.l1_lambda * avg_step_l1,
                'Loss/relative_error': relative_error.item(),
                'Parameters/weight_norm': weight_norm,
                'Parameters/grad_norm': grad_norm,
                'Parameters/encoder_norm': encoder_norm,
                'Parameters/decoder_norm': decoder_norm,
                'Parameters/encoder_grad_norm': encoder_grad_norm,
                'Parameters/decoder_grad_norm': decoder_grad_norm,
                'Activations/sparsity_1e-6': (sparse_representation.abs() < 1e-6).float().mean(),
                'Activations/sparsity_1e-1': (sparse_representation.abs() < 1e-1).float().mean(),
                'Activations/max_activation': sparse_representation.abs().max(),
            }
            log_metrics(writer, metrics, global_step)
            if train_mode:
                clipped_grad_norm = math.sqrt(sum(torch.sum(w.grad ** 2).item() for w in model.parameters() if w.grad is not None))
                log_metrics(writer, {'Parameters/clipped_grad_norm': clipped_grad_norm}, global_step)
        if step_callback is not None:
            step_callback(step, global_step)
        global_step += 1
        if train_mode and global_step % args.save_every_n_steps == 0 and local_rank == 0 and run_dir is not None and optimizer is not None:
            save_checkpoint(model, optimizer, run_dir, global_step, epoch_num)
    all_reduce_tensors([mse_loss_total, l1_loss_total, l0_loss_total, total_samples], op=dist.ReduceOp.SUM)
    all_reduce_tensors([active_features.int()], op=dist.ReduceOp.MAX)
    active_features = active_features.bool()
    avg_mse = mse_loss_total.item() / total_samples.item() if total_samples.item() > 0 else 0
    avg_l1 = l1_loss_total.item() / total_samples.item() if total_samples.item() > 0 else 0
    avg_l0 = l0_loss_total.item() / total_samples.item() if total_samples.item() > 0 else 0
    dead_feature_count = hidden_dim - active_features.sum().item()
    if not train_mode and writer is not None and local_rank == 0:
        metrics = {
            'Loss/val_mse': avg_mse,
            'Loss/val_l1': avg_l1,
            'Loss/val_l1_times_lambda': avg_l1 * args.l1_lambda,
            'Loss/val_total': avg_mse + args.l1_lambda * avg_l1,
            'Stats/val_l0': avg_l0,
            'Stats/val_dead_feature_count': dead_feature_count,
            'Stats/active_feature_percentage': 100 * (hidden_dim - dead_feature_count) / hidden_dim,
        }
        log_metrics(writer, metrics, global_step)
    if not train_mode and args.resample_dead and global_step > 0:
        resample_dead_features(model, active_features, local_rank)
    return avg_mse, avg_l1


def validate(model, validation_loader, local_rank, writer, step_num, args):
    """Run validation and return average MSE and L1 losses."""
    avg_val_mse, avg_val_l1 = run_epoch(
        model, validation_loader, local_rank, args, writer=writer, global_step=step_num, train_mode=False
    )
    return avg_val_mse, avg_val_l1


def save_checkpoint(model, optimizer, run_dir, global_step, epoch_num):
    """Save model and optimizer state to a checkpoint file."""
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'epoch': epoch_num,
    }
    torch.save(checkpoint, f'{run_dir}/checkpoint_step_{global_step}.pt')
    print(f'Checkpoint saved at step {global_step}')


def load_checkpoint(model, optimizer, checkpoint_path, local_rank):
    """Load model and optimizer state from a checkpoint file. Returns global_step and epoch."""
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint.get('global_step', 0)
    epoch = checkpoint.get('epoch', 0)
    print(f"Resumed from checkpoint {checkpoint_path} at step {global_step}, epoch {epoch}")
    return global_step, epoch


def train(model, local_rank, args):
    """Main training loop, supports checkpoint resuming and initial validation."""
    train_loader, validation_loader = get_data_loaders(args.activation_file, args.layer_num, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Starting training on device {local_rank}...')
    if local_rank == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        run_dir = f'{args.output_dir}/{current_time}_lambda_{args.l1_lambda}_hidden_mult_{args.hidden_dim_multiplier}_layernum_{args.layer_num}_lr_{args.lr}'
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(run_dir)
    else:
        writer = None
        run_dir = None
    global_step = 0
    start_epoch = 0
    if getattr(args, 'resume_checkpoint', None):
        global_step, start_epoch = load_checkpoint(model, optimizer, args.resume_checkpoint, local_rank)
    if getattr(args, 'init_val', False):
        avg_val_mse, avg_val_l1 = validate(model, validation_loader, local_rank, writer, global_step, args)
        if local_rank == 0:
            print(f'Initial validation: Val MSE: {avg_val_mse:.4f}, Val L1: {avg_val_l1:.4f}')
    for epoch_num in range(start_epoch, args.num_epochs):
        dist.barrier()
        avg_mse, avg_l1 = run_epoch(
            model, train_loader, local_rank, args, writer=writer, global_step=global_step, train_mode=True, epoch_num=epoch_num, run_dir=run_dir, optimizer=optimizer
        )
        global_step += len(train_loader)
        # Validation every n steps
        if (epoch_num + 1) % (args.validate_every_n_steps // len(train_loader)) == 0:
            avg_val_mse, avg_val_l1 = validate(model, validation_loader, local_rank, writer, global_step, args)
            if local_rank == 0:
                print(f'Epoch {epoch_num+1}: Val MSE: {avg_val_mse:.4f}, Val L1: {avg_val_l1:.4f}')


def main():
    """Parse arguments, setup DDP, and launch training."""
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder (SAE) with distributed PyTorch.")
    parser.add_argument('--output_dir', type=str, default='sae_out/', help='Output directory for logs and checkpoints')
    parser.add_argument('--layer_num', type=int, default=16, help='Layer number to train on')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--activation_file', type=str, default='/persist/adelworth/sae-fun/token_activations_500m.h5', help='Path to activation file')
    parser.add_argument('--hidden_dim_multiplier', type=int, default=64, help='Hidden dimension multiplier')
    parser.add_argument('--l1_lambda', type=float, default=1e-4, help='L1 regularization lambda')
    parser.add_argument('--save_every_n_steps', type=int, default=25000, help='Checkpoint save frequency (steps)')
    parser.add_argument('--validate_every_n_steps', type=int, default=25000, help='Validation frequency (steps)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--resample_dead', default=False, action='store_true', help='Resample dead features during validation')
    parser.add_argument('--init_val', default=False, action='store_true', help='Run initial validation before training')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    local_rank = setup_ddp()
    model = load_model(local_rank, args)
    train(model, local_rank, args)

if __name__ == "__main__":
    main()

'''
TODO:
- add pre-encoder bias
- neuron resampling
- add normalization of activations?
'''
