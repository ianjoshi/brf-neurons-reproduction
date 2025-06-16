import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn
import math
from datetime import datetime
from tqdm import tqdm
import random
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from gsc.data.SpeechCommandsDataLoader import SpeechCommandsDataLoader
from gsc.data.Preprocessor import Preprocessor
from gsc.train import tools

import snn

###################################################################
# General Settings
###################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not available. Running on CPU may be slow.")

if device.type == "cuda":
    pin_memory = True
    num_workers = 1  # Increased for faster data loading
else:
    pin_memory = False
    num_workers = 0  # Non-zero for CPU to parallelize data loading

print(f"Using device: {device}")
print(f"Number of workers: {num_workers}")
print(f"Pin memory: {pin_memory}")

####################################################################
# DataLoader Setup
####################################################################

sequence_length = 100 
input_size = 13
hidden_size = 128
num_classes = 35

train_batch_size = 16
val_batch_size = 64
test_batch_size = 64

# Percentage of data to use (e.g., 25 for 25%)
data_percentage = 25
seed = 42

# Set random seed for reproducibility
random.seed(seed)

loader_factory = SpeechCommandsDataLoader(
    root="./gsc-experiments/data",
    sequence_length=sequence_length,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    cache_data=True,
    preload_cache=True,
    data_percentage=data_percentage,
    seed=seed
)

# Get the DataLoaders
train_loader, val_loader, test_loader = loader_factory.get_loaders()

# Get dataset sizes
train_dataset_size = len(train_loader.dataset)
val_dataset_size = len(val_loader.dataset)
test_dataset_size = len(test_loader.dataset)

# Update step counts
total_train_steps = len(train_loader)
total_val_steps = len(val_loader)
total_test_steps = len(test_loader)

print(f"Using {data_percentage}% of data:")
print(f"Train dataset size: {train_dataset_size} samples")
print(f"Validation dataset size: {val_dataset_size} samples")
print(f"Test dataset size: {test_dataset_size} samples")
print(f"Train steps per epoch: {total_train_steps}")
print(f"Validation steps per epoch: {total_val_steps}")
print(f"Test steps per epoch: {total_test_steps}")

# Preprocessor for batch formatting
preprocessor = Preprocessor(normalize_inputs=True, permute_inputs=True)

####################################################################
# Model Setup
####################################################################

mask_prob = 0.0
omega_mean = 3.0
omega_std = 5.0
b_offset_a = 0.1
b_offset_b = 1.0
out_adaptive_tau_mem_mean = 20.0
out_adaptive_tau_mem_std = 1.0
sub_seq_length = 0
dt = 0.01

model = snn.models.BRFRSNN_SD(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=omega_mean,
    adaptive_omega_b=omega_std,
    adaptive_b_offset_a=b_offset_a,
    adaptive_b_offset_b=b_offset_b,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    sub_seq_length=sub_seq_length,
    output_bias=False,
).to(device)

# Optimize model with torch.jit
model = torch.jit.script(model)

####################################################################
# Setup Experiment (Optimizer etc.)
####################################################################

rand_num = random.randint(1, 10000)
criterion = torch.nn.NLLLoss()
optimizer_lr = 0.01
gradient_clip_value = 1.0
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)

epochs_num = 100

scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch_count: 1. - epoch_count / epochs_num)

opt_str = "{}_Adam({}),script-bw,NLL,LinearLR,no_gc".format(rand_num, optimizer_lr)
net_str = "13,{},35,bs={},ep={}".format(hidden_size, train_batch_size, epochs_num)
unit_str = "BRF(omega{},{},b{},{})LI({},{})".format(
    omega_mean, omega_std, b_offset_a, b_offset_b,
    out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std
)

comment = opt_str + "," + net_str + "," + unit_str

writer = SummaryWriter(comment=comment)

start_time = datetime.now().strftime("%m-%d_%H-%M-SS")
print(f"\nTraining started at: {start_time}")
print(f"Configuration: {comment}")
print(f"\nModel architecture:\n{model}")

save_path = "./gsc-experiments/models/{}_".format(start_time) + comment + ".pt"
save_init_path = "./gsc-experiments/models/{}_init_".format(start_time) + comment + ".pt"

torch.save({'model_state_dict': model.state_dict()}, save_init_path)

min_val_loss = float('inf')
loss_value = 1.0
iteration = 0
end_training = False

####################################################################
# Training Loop
####################################################################

run_time = tools.PerformanceCounter()
tools.PerformanceCounter.reset(run_time)

for epoch in range(epochs_num + 1):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{epochs_num}")
    print(f"{'='*50}")
    
    model.eval()

    with torch.no_grad():
        # Validation
        val_loss = 0
        val_correct = 0
        val_total_spikes = torch.tensor(0.0).to(device)
        print("\nRunning validation...")
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        for i, (inputs, targets) in enumerate(val_pbar):
            inputs, targets = preprocessor.process_batch(inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, num_spikes = model(inputs)
            val_total_spikes += num_spikes  # Remove .item() since num_spikes is an int

            loss = tools.apply_seq_loss(
                criterion=criterion,
                outputs=outputs,
                targets=targets
            )
            val_loss_value = loss.item()
            val_loss += val_loss_value

            val_correct += tools.count_correct_prediction(
                predictions=outputs,
                targets=targets
            )
            
            val_pbar.set_postfix({'loss': f'{val_loss_value:.4f}'})

        val_loss /= total_val_steps
        val_acc = (val_correct / val_dataset_size) * 100.0
        val_sop = val_total_spikes / val_dataset_size

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)
        writer.add_scalar("sop/val", val_sop, epoch)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_value,
                },
                save_path
            )

        print(
            f"\nValidation Results:"
            f"\n  Loss: {val_loss:.6f}"
            f"\n  Accuracy: {val_acc:.4f}%"
            f"\n  SOP: {val_sop:.2f}"
            f"\n  Best loss so far: {min_val_loss:.6f} (epoch {min_val_epoch})"
        )

        # Testing
        test_loss = 0
        test_correct = 0
        test_total_spikes = torch.tensor(0.0).to(device)
        print("\nRunning testing...")
        
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for i, (inputs, targets) in enumerate(test_pbar):
            inputs, targets = preprocessor.process_batch(inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, num_spikes = model(inputs)
            test_total_spikes += num_spikes  # Remove .item() since num_spikes is an int

            loss = tools.apply_seq_loss(
                criterion=criterion,
                outputs=outputs,
                targets=targets
            )
            test_loss_value = loss.item()
            test_loss += test_loss_value

            test_correct += tools.count_correct_prediction(
                predictions=outputs,
                targets=targets
            )
            
            test_pbar.set_postfix({'loss': f'{test_loss_value:.4f}'})

        test_loss /= total_test_steps
        test_acc = (test_correct / test_dataset_size) * 100.0
        test_sop = test_total_spikes / test_dataset_size

        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("accuracy/test", test_acc, epoch)
        writer.add_scalar("sop/test", test_sop, epoch)

        print(
            f"\nTest Results:"
            f"\n  Loss: {test_loss:.6f}"
            f"\n  Accuracy: {test_acc:.4f}%"
            f"\n  SOP: {test_sop:.2f}"
        )

    print(
        "Epoch [{:4d}/{:4d}]  |  Summary | Loss/val: {:.6f}, Accuracy/val: {:8.4f}  | "
        "Loss/test: {:.6f}, Accuracy/test: {:8.4f} | SOP: {:.2f}"
        .format(epoch, epochs_num, val_loss, val_acc, test_loss, test_acc, test_sop),
        flush=True
    )

    writer.flush()

    if epoch < epochs_num:
        print("\nStarting training phase...")
        print_train_loss = 0
        print_train_correct = 0
        model.train()
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for i, (inputs, targets) in enumerate(train_pbar):
            current_batch_size = len(inputs)
            inputs, targets = preprocessor.process_batch(inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(inputs)

            loss = tools.apply_seq_loss(
                criterion=criterion,
                outputs=outputs,
                targets=targets
            )
            loss_value = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()

            print_train_loss += loss_value

            batch_correct = tools.count_correct_prediction(
                predictions=outputs,
                targets=targets
            )
            print_train_correct += batch_correct

            batch_accuracy = (batch_correct / current_batch_size) * 100.0

            writer.add_scalar("Loss/train", loss_value, iteration)
            writer.add_scalar("accuracy/train", batch_accuracy, iteration)

            train_pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'acc': f'{batch_accuracy:.2f}%'
            })

            if math.isnan(loss_value):
                end_training = True
                break

            iteration += 1

        print_train_loss /= total_train_steps
        print_acc = (print_train_correct / train_dataset_size) * 100.0

        print(
            f"\nTraining Results:"
            f"\n  Loss: {print_train_loss:.6f}"
            f"\n  Accuracy: {print_acc:.4f}%"
            f"\n  Learning rate: {scheduler.get_last_lr()[0]:.6f}"
        )

        writer.flush()
        scheduler.step()

    if end_training:
        break

print('\nTraining completed!')
print(f'Best validation loss: {min_val_loss:.6f} at epoch {min_val_epoch}')
print(f'Total training time: {tools.PerformanceCounter.time(run_time) / 3600:.2f} hours')