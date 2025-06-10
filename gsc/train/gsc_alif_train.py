import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn
import math
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from gsc.data.SpeechCommandsDataLoader import SpeechCommandsDataLoader
from gsc.data.Preprocessor import Preprocessor
from gsc.train import tools

import snn
import random

###################################################################
# General Settings
###################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = 0

print(f"Using device: {device}")
print(f"Number of workers: {num_workers}")
print(f"Pin memory: {pin_memory}")

####################################################################
# DataLoader Setup
####################################################################

sequence_length = 1300
input_size = 13
hidden_size = 36
num_classes = 35

train_batch_size = 4
val_batch_size = 9981 
test_batch_size = 11005

loader_factory = SpeechCommandsDataLoader(
    root="./gsc-experiments/data",
    sequence_length=sequence_length,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

train_loader, val_loader, test_loader = loader_factory.get_loaders()

# Preprocessor for batch formatting
preprocessor = Preprocessor(normalize_inputs=False)

####################################################################
# Model setup
####################################################################

# Fraction of elements in hidden.linear.weight to be zero
mask_prob = 0.0

# ALIF alpha tau_mem init normal distribution
adaptive_tau_mem_mean = 20.0
adaptive_tau_mem_std = 0.5

# ALIF rho tau_adp init normal distribution
adaptive_tau_adp_mean = 7.0
adaptive_tau_adp_std = 0.2

# LI alpha tau_mem init normal distribution
out_adaptive_tau_mem_mean = 20.0
out_adaptive_tau_mem_std = 0.5

sub_seq_length = 10

hidden_bias = True
output_bias = True

model = snn.models.SimpleALIFRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    mask_prob=mask_prob,
    adaptive_tau_mem_mean=adaptive_tau_mem_mean,
    adaptive_tau_mem_std=adaptive_tau_mem_std,
    adaptive_tau_adp_mean=adaptive_tau_adp_mean,
    adaptive_tau_adp_std=adaptive_tau_adp_std,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    sub_seq_length=sub_seq_length,
    hidden_bias=hidden_bias,
    output_bias=output_bias
).to(device)

####################################################################
# Setup Experiment (Optimizer etc.)
####################################################################

# Prevent overwriting in slurm
rand_num = random.randint(1, 10000)

criterion = torch.nn.NLLLoss()

optimizer_lr = 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)

# Number of iterations per epoch
total_train_steps = len(train_loader)
total_val_steps = len(val_loader)
total_test_steps = len(test_loader)

epochs_num = 400

# Learning rate scheduling
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch_count: 1. - epoch_count / epochs_num)

# Logging
opt_str = "{}_Adam({}),NLL,LinearLR,no_gc".format(rand_num, optimizer_lr)
net_str = "RSNN(4,36,6,sub_seq_{},bs_{},ep_{},h_o_bias(True))"\
    .format(sub_seq_length, train_batch_size, epochs_num)
unit_str = "ALIF(tau_m({},{}),tau_a({},{}),linMask_{})LI(tau_m({},{}))"\
    .format(adaptive_tau_mem_mean, adaptive_tau_mem_std, adaptive_tau_adp_mean, adaptive_tau_adp_std, mask_prob,
            out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std)

comment = opt_str + "," + net_str + "," + unit_str

writer = SummaryWriter(comment=comment)

start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
print(f"\nTraining started at: {start_time}")
print(f"Configuration: {comment}")
print(f"\nModel architecture:")
print(model)
print("\nModel parameters:")
print(model.state_dict())

save_path = "./gsc-experiments/models/{}_".format(start_time) + comment + ".pt"
save_init_path = "./gsc-experiments/models/{}_init_".format(start_time) + comment + ".pt"

# Save initial parameters for analysis
torch.save({'model_state_dict': model.state_dict()}, save_init_path)

min_val_loss = float('inf')
loss_value = 1.0 # Dummy init for val.
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
    
    # Evaluation mode for validation and testing
    model.eval()

    with torch.no_grad():
        # Validation
        val_loss = 0
        val_correct = 0
        print("\nRunning validation...")

        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = preprocessor.process_batch(inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)[0]

            loss = tools.apply_seq_loss(
                criterion=criterion,
                outputs=outputs,
                targets=targets[sub_seq_length:, :, :]
            )
            val_loss_value = loss.item() / (sequence_length - sub_seq_length)
            val_loss += val_loss_value

            val_correct += tools.count_correct_prediction(
                predictions=outputs,
                targets=targets[sub_seq_length:, :, :]
            )

        val_loss /= total_val_steps
        val_acc = (val_correct / (val_batch_size * (sequence_length - sub_seq_length))) * 100.0

        # Log current val loss and accuracy
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)

        # Save current best model.
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            saved_best_model = model.state_dict()
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
            f"\n  Best loss so far: {min_val_loss:.6f} (epoch {min_val_epoch})"
        )

        # Testing
        test_loss = 0
        test_correct = 0
        print("\nRunning testing...")

        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = preprocessor.process_batch(inputs, targets)
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)[0]

            loss = tools.apply_seq_loss(
                criterion=criterion,
                outputs=outputs,
                targets=targets[sub_seq_length:, :, :]
            )
            test_loss_value = loss.item() / (sequence_length - sub_seq_length)
            test_loss += test_loss_value

            test_correct += tools.count_correct_prediction(
                predictions=outputs,
                targets=targets[sub_seq_length:, :, :]
            )

        test_loss /= total_test_steps
        test_acc = (test_correct / (test_batch_size * (sequence_length - sub_seq_length))) * 100.0

        # Log current test loss and accuracy
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("accuracy/test", test_acc, epoch)

        print(
            f"\nTest Results:"
            f"\n  Loss: {test_loss:.6f}"
            f"\n  Accuracy: {test_acc:.4f}%"
        )

    print(
        "Epoch [{:4d}/{:4d}]  |  Summary | Loss/val: {:.6f}, Accuracy/val: {:8.4f}  | "
        " Loss/test: {:.6f}, Accuracy/test: {:8.4f}"
        .format(epoch, epochs_num, val_loss, val_acc, test_loss, test_acc),
        flush=True
    )

    # Update logging outputs
    writer.flush()

    # Training
    # Run training from 0 to 399 epochs
    if epoch < epochs_num:
        print("\nStarting training phase...")
        print_train_loss = 0
        print_train_correct = 0

        # Go to training mode
        model.train()

        for i, (inputs, targets) in enumerate(train_loader):
            current_batch_size = len(inputs)
            inputs, targets = preprocessor.process_batch(inputs, targets)
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear gradients
            optimizer.zero_grad()
            outputs = model(inputs)[0]

            # Accumulate loss for each time step and batch
            loss = tools.apply_seq_loss(
                criterion=criterion,
                outputs=outputs,
                targets=targets[sub_seq_length:, :, :]
            )
            loss_value = loss.item() / (sequence_length - sub_seq_length)

            # Calculate the gradients
            loss.backward()

            # Perform learning step
            optimizer.step()

            # Sum up loss_value for each iteration
            print_train_loss += loss_value

            # Calculate batch accuracy
            batch_correct = tools.count_correct_prediction(
                predictions=outputs,
                targets=targets[sub_seq_length:, :, :]
            )
            print_train_correct += batch_correct

            batch_accuracy = (
                batch_correct / (current_batch_size * (sequence_length - sub_seq_length))
            ) * 100.0

            # Log current loss and accuracy
            writer.add_scalar("Loss/train", loss_value, iteration)
            writer.add_scalar("accuracy/train", batch_accuracy, iteration)

            if math.isnan(loss_value):
                end_training = True
                break

            iteration += 1

        print_train_loss /= total_train_steps
        print_acc = (
            print_train_correct / (train_batch_size * (sequence_length - sub_seq_length))
        ) * 100.0

        print(
            f"\nTraining Results:"
            f"\n  Loss: {print_train_loss:.6f}"
            f"\n  Accuracy: {print_acc:.4f}%"
            f"\n  Learning rate: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Update logging outputs
        writer.flush()

        # Apply learning rate scheduling
        scheduler.step()

        if end_training:
            print("\nTraining stopped due to NaN loss!")
            break

print('\nTraining completed!')
print(f'Best validation loss: {min_val_loss:.6f} at epoch {min_val_epoch}')
print(f'Total training time: {tools.PerformanceCounter.time(run_time) / 3600:.2f} hours')






