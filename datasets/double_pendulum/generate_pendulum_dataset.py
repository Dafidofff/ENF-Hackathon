from double_pendulum import DoublePendulum, generate_random_initial_state
from animation import create_frames
import numpy as np
import argparse
import os
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate double pendulum dataset')
    parser.add_argument('--num_sequences', type=int, help='Number of sequences to generate',
                        default=100)
    parser.add_argument('--output_dir', type=str, default='./data/double_pendulum',
                       help='Directory to save the dataset')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Duration of each sequence in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for the sequences')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pendulum
    pendulum = DoublePendulum(L1=1.5, L2=1.5, M1=1.0, M2=1.0)
    
    # Calculate number of timesteps
    num_timesteps = int(args.duration * args.fps)
    
    # Initialize array to store all frames
    # Shape: (num_sequences, timesteps, H, W, 3)
    dataset = np.zeros((args.num_sequences, num_timesteps, 64, 64, 3), dtype=np.uint8)
    initial_conditions = np.zeros((args.num_sequences, 4))
    
    # Generate sequences with progress bar
    for i in tqdm(range(args.num_sequences), desc="Generating sequences"):
        # Generate random initial conditions
        initial_state = generate_random_initial_state()
        initial_conditions[i] = initial_state
        
        # Simulation parameters
        dt = 1/args.fps
        
        # Simulate
        _, solution = pendulum.simulate(initial_state, args.duration, dt)
        
        # Render sequence
        frames = create_frames(pendulum, solution, size=64)
        dataset[i] = frames
    
    # Save the dataset
    print("\nSaving dataset...")
    np.save(os.path.join(args.output_dir, 'pendulum_videos.npy'), dataset)
    np.save(os.path.join(args.output_dir, 'initial_conditions.npy'), initial_conditions)
    
    print(f"Dataset saved in {args.output_dir}")
    print(f"Videos shape: {dataset.shape}")
    print(f"Initial conditions shape: {initial_conditions.shape}")

if __name__ == "__main__":
    main() 