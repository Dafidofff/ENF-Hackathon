import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np


def create_frames(pendulum, solution, size=64):
    """
    Generate animation frames for a double pendulum simulation without displaying.
    
    Args:
        pendulum: Pendulum object with get_coordinates method
        solution: Array of shape (n_steps, 4) containing theta1, omega1, theta2, omega2
        size: Size of the output frames in pixels
        
    Returns:
        numpy array of shape (n_steps, size, size, 3) containing RGB frames
    """
    # Use Agg backend which doesn't require a display
    matplotlib.use('Agg')
    
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Create visual elements with thicker lines and larger balls
    line, = ax.plot([], [], 'w-', lw=4)  # Increased line width
    ball1 = patches.Circle((0, 0), 0.25, fc='white')
    ball2 = patches.Circle((0, 0), 0.25, fc='white')
    ax.add_patch(ball1)
    ax.add_patch(ball2)
    
    # Initialize storage for frames
    frames = []
    
    # Generate each frame
    for state in solution:
        theta1, _, theta2, _ = state
        x1, y1, x2, y2 = pendulum.get_coordinates(theta1, theta2)
        
        # Update positions
        line.set_data([0, x1, x2], [0, y1, y2])
        ball1.center = (x1, y1)
        ball2.center = (x2, y2)
        
        # Render and capture frame
        fig.canvas.draw()
        # Get the ARGB buffer and reshape to include alpha channel
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape((size, size, 4))
        # Remove alpha channel
        data = data[:, :, 1:]  # Keep only RGB channels
        frames.append(data.copy())
    
    plt.close()
    
    return np.array(frames)