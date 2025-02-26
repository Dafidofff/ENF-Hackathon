import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os

def play_sequence(sequence, save_path=None, fps=30):
    """
    Display or save an animation of a single sequence.
    
    Args:
        sequence: numpy array of shape (T, H, W, 3)
        save_path: if provided, save animation to this path
        fps: frames per second for playback/saving
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    # Create image plot
    img = ax.imshow(sequence[0])
    
    def animate(frame):
        img.set_array(frame)
        return [img]
    
    anim = FuncAnimation(fig, animate, frames=sequence,
                        interval=1000/fps, blit=True)
    
    if save_path:
        anim.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize double pendulum dataset')
    parser.add_argument('--data_dir', type=str, default='pendulum_dataset',
                       help='Directory containing the dataset')
    parser.add_argument('--sequence_idx', type=int, default=0,
                       help='Index of sequence to visualize')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='If provided, save animations to this directory')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for playback')
    
    args = parser.parse_args()
    
    # Load dataset
    data_path = os.path.join(args.data_dir, 'pendulum_videos.npy')
    ic_path = os.path.join(args.data_dir, 'initial_conditions.npy')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print("Loading dataset...")
    dataset = np.load(data_path)
    initial_conditions = np.load(ic_path)
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Initial conditions shape: {initial_conditions.shape}")
    
    # Get sequence to visualize
    if args.sequence_idx >= len(dataset):
        raise ValueError(f"Sequence index {args.sequence_idx} out of range (max {len(dataset)-1})")
    
    sequence = dataset[args.sequence_idx][:, :, :, 0]
    ic = initial_conditions[args.sequence_idx]
    
    print(f"\nViewing sequence {args.sequence_idx}")
    print(f"Initial conditions: θ1={ic[0]:.2f}, ω1={ic[1]:.2f}, θ2={ic[2]:.2f}, ω2={ic[3]:.2f}")
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f'pendulum_{args.sequence_idx:04d}.mp4')
        print(f"Saving animation to {save_path}")
        play_sequence(sequence, save_path, args.fps)
    else:
        print("Displaying animation... (close window to exit)")
        play_sequence(sequence, fps=args.fps)

if __name__ == "__main__":
    main() 