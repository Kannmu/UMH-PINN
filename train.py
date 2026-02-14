"""
Training Loop for PINN-based Inverse Design

Optimizes the focal point trajectory to maximize shear stress at target
while improving focus efficiency via destructive interference.

Multi-platform support: CUDA, MPS (Apple Silicon), CPU
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

import config
from device_utils import get_device, print_device_info
from physics_layer import ViscoelasticWave3D
from model import NeuralSplineTrajectory
from loss import compute_loss
from training_io import (
    save_wave_gif,
    save_trajectory_plot,
    plot_training_history,
    setup_training_logger,
    close_training_logger,
    create_csv_logger,
    write_csv_row,
)

def train(
    n_epochs: int = config.TRAIN_CONFIG["n_epochs"],
    lr: float = config.TRAIN_CONFIG["lr"],
    target_pos: tuple = config.TRAIN_CONFIG["target_pos"],
    loss_weights: Optional[Dict[str, float]] = None,
    save_interval: int = config.TRAIN_CONFIG["save_interval"],
    output_dir: str = config.TRAIN_CONFIG["output_dir"],
    batch_size: int = config.TRAIN_CONFIG["batch_size"],
    device: Optional[torch.device] = None,
):

    """
    Main training loop.

    Args:
        n_epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        target_pos: (x, y) target position in meters
        loss_weights: Dict of loss component weights
        save_interval: Save visualization every N epochs
        output_dir: Directory for outputs
        device: Torch device (auto-detected if None)
    """
    # Set device
    if device is None:
        device = get_device(verbose=True)

    # Print device info
    print(f"\nTraining on: {device}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Initialize components
    print("Initializing NeuralSplineTrajectory model...")
    model_params = dict(config.MODEL_CONFIG)
    model = NeuralSplineTrajectory(**model_params).to(device)
    print(model)

    print("Initializing physics engine...")
    physics_params = dict(config.PHYSICS_CONFIG)
    sim_cycles = physics_params.get("sim_cycles", 1)
    physics_params.pop("sim_cycles", None)
    physics_params["n_steps"] = model_params["n_steps"] * sim_cycles
    physics_params["device"] = device
    physics = ViscoelasticWave3D(**physics_params)

    if config.TRAIN_CONFIG.get("use_tf32", False) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if config.TRAIN_CONFIG.get("compile", False):
        print("Compiling model and physics with torch.compile... (This may take a while on first run)")
        model = torch.compile(model)
        print("Model compiled.")
    else:
        print("Running in eager mode (torch.compile disabled).")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Default loss weights (Optimized for NeuralTrajectory)
    if loss_weights is None:
        loss_weights = dict(config.LOSS_WEIGHTS)

    # Training history
    history = {
        'total': [],
        'contrast': [],
        'signal': [],
        'noise': [],
        'snr': [],
        'trajectory_length': [],
        'jerk': [],
    }

    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Target: {target_pos}")
    print(f"Loss weights: {loss_weights}")
    print("-" * 60)

    # Setup loggers (overwrite each run)
    logger = setup_training_logger(output_dir)
    logger.info("Training start")
    logger.info(f"device={device}")
    logger.info(f"target_pos={target_pos}")
    logger.info(f"loss_weights={loss_weights}")
    logger.info(f"n_epochs={n_epochs} | lr={lr} | save_interval={save_interval}")

    csv_file = None
    csv_writer = None

    # Training loop
    pbar = tqdm(range(n_epochs), desc="Training")
    try:
        for epoch in pbar:
            optimizer.zero_grad()

            # Forward pass
            # Only request full stress history on visualization steps
            should_visualize = (epoch % save_interval == 0 or epoch == n_epochs - 1)
            return_vz = False
            return_stress_history = should_visualize
            trajectory = model(batch_size=batch_size)

            wave_output = physics(
                trajectory,
                return_vz=return_vz,
                return_stress_history=return_stress_history,
            )

            # Compute loss
            loss, loss_dict = compute_loss(
                wave_output, trajectory, target_pos, physics,
                target_radius=config.LOSS_TARGET_RADIUS,
                weights=loss_weights,
            )

            if csv_writer is None:
                csv_file, csv_writer, csv_fieldnames = create_csv_logger(
                    output_dir,
                    loss_dict,
                )
                logger.info(f"csv_header={csv_fieldnames}")

            # Backward pass
            loss.backward()

            # Gradient Clipping to prevent NaN/Exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Record history
            for key in history:
                if key in loss_dict:
                    history[key].append(loss_dict[key])

            # Save CSV row (overwrite file each run)
            write_csv_row(csv_writer, csv_file, epoch, lr, target_pos, loss_dict)

            # Save log row
            log_kv = " | ".join([f"{k}={loss_dict[k]:.8f}" for k in loss_dict])
            logger.info(f"epoch={epoch} | {log_kv}")

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_dict["total"]:.4f}',
                'snr': f'{loss_dict["snr"]:.3f}',
                'traj_len': f'{loss_dict["trajectory_length"]:.4f}',
                'jerk': f'{loss_dict["jerk"]:.4f}',
            })

            # Check for NaN
            if np.isnan(loss_dict['total']):
                print(f"\n[Error] Loss is NaN at epoch {epoch}!")
                print(f"Loss dict: {loss_dict}")
                logger.error(f"Loss is NaN at epoch {epoch} | loss_dict={loss_dict}")
                break

            # Log
            if epoch % 10 == 0:
                if epoch == 0:
                    header = f"{'Epoch':^6} | {'Total':^10} | {'Contrast':^10} | {'Signal':^10} | {'Noise':^10} | {'TrajLen':^10} | {'Jerk':^10} | {'SNR':^10}"
                    tqdm.write(header)
                    tqdm.write("-" * len(header))

                log_str = (
                    f"{epoch:^6d} | "
                    f"{loss_dict['total']:^10.4f} | "
                    f"{loss_dict['contrast']:^10.4f} | "
                    f"{loss_dict['signal']:^10.4f} | "
                    f"{loss_dict['noise']:^10.4f} | "
                    f"{loss_dict['trajectory_length']:^10.4f} | "
                    f"{loss_dict['jerk']:^10.4f} | "
                    f"{loss_dict['snr']:^10.3f}"
                )
                tqdm.write(log_str)

            # Visualization
            should_visualize = (epoch % save_interval == 0 or epoch == n_epochs - 1)

            if should_visualize:
                save_trajectory_plot(
                    trajectory, epoch, output_dir, target_pos, loss_dict
                )
            if epoch == n_epochs - 1:
                # Ensure we have vz_history for the final GIF if needed, though gif only uses stress
                # save_wave_gif uses 'stress_history' which is always returned
                save_wave_gif(wave_output, trajectory, epoch, output_dir, target_pos)
    finally:
        if csv_file is not None:
            csv_file.close()
        close_training_logger(logger)

    # Save final results
    print("\nTraining complete!")
    print(f"Final loss: {loss_dict}")
    logger.info("Training complete")
    logger.info(f"Final loss: {loss_dict}")

    # Save model
    model_path = os.path.join(output_dir, 'trajectory_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'trajectory': model().detach().cpu(),
        'history': history,
        'config': {
            'target_pos': target_pos,
            'loss_weights': loss_weights,
            'model_type': 'neural_spline',
        }
    }, model_path)
    print(f"Saved model: {model_path}")

    # Plot training history
    plot_training_history(history, output_dir)

    return model, history




def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train PINN for ultrasound haptics')
    parser.add_argument('--epochs', type=int, default=config.TRAIN_CONFIG["n_epochs"], help='Number of epochs')
    parser.add_argument('--lr', type=float, default=config.TRAIN_CONFIG["lr"], help='Learning rate')
    parser.add_argument('--target-x', type=float, default=config.TRAIN_CONFIG["target_pos"][0], help='Target X position (m)')
    parser.add_argument('--target-y', type=float, default=config.TRAIN_CONFIG["target_pos"][1], help='Target Y position (m)')
    parser.add_argument('--output', type=str, default=config.TRAIN_CONFIG["output_dir"], help='Output directory')
    parser.add_argument('--save-interval', type=int, default=config.TRAIN_CONFIG["save_interval"], help='Save interval')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Force specific device (auto-detect if not specified)')

    args = parser.parse_args()

    # Print device info
    print_device_info()

    # Override device if specified
    device = None
    if args.device:
        device = torch.device(args.device)
        print(f"Forcing device: {device}")

    train(
        n_epochs=args.epochs,
        lr=args.lr,
        target_pos=(args.target_x, args.target_y),
        output_dir=args.output,
        save_interval=args.save_interval,
        batch_size=config.TRAIN_CONFIG["batch_size"],
        device=device,
    )


if __name__ == "__main__":
    main()
