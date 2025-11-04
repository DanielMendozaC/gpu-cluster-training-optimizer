#!/usr/bin/env python3
"""
Distributed Data Parallel Training with PyTorch
Optimized for multi-GPU infrastructure with NCCL backend
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
from gpu_monitor import GPUMonitor


class ResNetTrainer:
    def __init__(self, rank, world_size, args):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        
        # Initialize process group
        self.setup_distributed()
        
        # Setup model, data, optimizer
        self.model = self.setup_model()
        self.train_loader, self.val_loader = self.setup_data()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=args.lr, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # Initialize monitoring
        self.monitor = GPUMonitor(rank) if args.monitor else None
        
    def setup_distributed(self):
        """Initialize distributed training environment"""
        # Set device
        torch.cuda.set_device(self.rank)
        
        # Initialize process group with NCCL backend
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        # NCCL optimization settings
        os.environ['NCCL_DEBUG'] = 'WARN'
        os.environ['NCCL_P2P_LEVEL'] = 'NVL'  # Use NVLink when available
        
        if self.rank == 0:
            print(f"✓ Initialized DDP with {self.world_size} GPUs (NCCL backend)")
    
    def setup_model(self):
        """Setup model with DDP wrapper"""
        # Create model
        model = torchvision.models.resnet50(pretrained=False, num_classes=10)
        model = model.to(self.rank)
        
        # Wrap with DDP - key parameters for optimization
        model = DDP(
            model, 
            device_ids=[self.rank],
            bucket_cap_mb=25,  # Optimized bucket size for gradient communication
            gradient_as_bucket_view=True  # Memory optimization
        )
        
        return model
    
    def setup_data(self):
        """Setup distributed data loaders"""
        # Data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Download datasets (only on rank 0)
        if self.rank == 0:
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test
            )
        
        # Barrier to ensure download completes
        dist.barrier()
        
        # Load datasets on all ranks
        if self.rank != 0:
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=False, transform=transform_train
            )
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=False, transform=transform_test
            )
        
        # DistributedSampler for proper data sharding
        train_sampler = DistributedSampler(
            trainset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            testset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # DataLoaders with pinned memory for faster transfer
        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            testset,
            batch_size=self.args.batch_size * 2,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)  # Important for proper shuffling
        
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.rank), targets.to(self.rank)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass (gradient sync happens automatically)
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Logging
            if self.rank == 0 and batch_idx % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                throughput = (batch_idx + 1) * self.args.batch_size * self.world_size / elapsed
                
                print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.3f} | "
                      f"Throughput: {throughput:.0f} img/s")
                
                if self.monitor:
                    self.monitor.log_metrics(epoch, batch_idx, loss.item(), throughput)
        
        # Average loss across all processes
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.rank), targets.to(self.rank)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Gather metrics from all processes
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # All-reduce to get global metrics
        metrics = torch.tensor([avg_loss, accuracy]).to(self.rank)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= self.world_size
        
        return metrics[0].item(), metrics[1].item()
    
    def train(self):
        """Main training loop"""
        if self.rank == 0:
            print("\n" + "="*60)
            print(f"Starting training on {self.world_size} GPUs")
            print(f"Batch size per GPU: {self.args.batch_size}")
            print(f"Total batch size: {self.args.batch_size * self.world_size}")
            print("="*60 + "\n")
        
        best_acc = 0
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Learning rate schedule
            self.scheduler.step()
            
            # Logging (rank 0 only)
            if self.rank == 0:
                print(f"\nEpoch {epoch} Complete:")
                print(f"  Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%")
                print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}\n")
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_checkpoint(epoch, val_acc)
        
        # Training complete
        if self.rank == 0:
            total_time = time.time() - start_time
            print("\n" + "="*60)
            print(f"Training Complete!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Best accuracy: {best_acc:.2f}%")
            print("="*60 + "\n")
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
        }
        torch.save(checkpoint, f'checkpoint_best.pth')
        print(f"✓ Saved checkpoint (accuracy: {accuracy:.2f}%)")
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.monitor:
            self.monitor.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Distributed Training with PyTorch DDP')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--monitor', action='store_true', help='Enable GPU monitoring')
    
    args = parser.parse_args()
    
    # Get distributed training info from environment (set by torchrun)
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Create trainer and run
    trainer = ResNetTrainer(rank, world_size, args)
    
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == '__main__':
    main()