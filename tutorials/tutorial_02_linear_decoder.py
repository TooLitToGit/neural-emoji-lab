"""
Tutorial 02: Linear Decoder - Building the Neural Brain
========================================================

This tutorial implements a simple linear neural decoder that learns to
reconstruct emojis from 128-dimensional latent codes. We use the 
Moore-Penrose pseudoinverse to "solve" for optimal weights in one step.

Key Concepts:
- Latent space (high-dimensional coordinate system)
- Linear projection (matrix multiplication)
- Pseudoinverse (analytical weight solving)
- Feature compression (3072 pixels → 128 dimensions)

Educational Goal:
Understanding how neural networks "learn" can be as simple as solving
a system of linear equations. No backpropagation needed!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import HTML
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

LATENT_DIMENSIONS = 128  # The "DNA code" size
IMAGE_SIZE = 32          # 32x32 pixels
PIXEL_COUNT = IMAGE_SIZE ** 2 * 3  # 3072 total RGB values

# Load DNA from previous tutorial
DNA_FILE = 'tutorial_01_dna.npy'


# =============================================================================
# STEP 1: LATENT SPACE INITIALIZATION
# =============================================================================

def initialize_latent_codes(num_emojis, latent_dim, seed=42):
    """
    Generate random latent codes for each emoji.
    
    Why random initialization:
    - We need a unique coordinate for each emoji in 128-D space
    - Random Gaussian distribution spreads them evenly
    - The seed ensures reproducibility
    
    Think of this as assigning GPS coordinates to each emoji
    in a 128-dimensional universe.
    
    Args:
        num_emojis: Number of landmarks (10 in our case)
        latent_dim: Dimensionality of latent space (128)
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Shape (num_emojis, latent_dim)
    """
    print(f"🧭 Initializing {num_emojis} landmarks in {latent_dim}-D space...")
    
    np.random.seed(seed)
    latent_codes = np.random.randn(num_emojis, latent_dim)
    
    # Verify separation (measure average distance)
    distances = []
    for i in range(num_emojis):
        for j in range(i + 1, num_emojis):
            dist = np.linalg.norm(latent_codes[i] - latent_codes[j])
            distances.append(dist)
    
    avg_distance = np.mean(distances)
    print(f"   ✓ Average landmark separation: {avg_distance:.2f}")
    print(f"   ✓ This ensures distinct 'neighborhoods' for each emoji\n")
    
    return latent_codes


# =============================================================================
# STEP 2: WEIGHT MATRIX TRAINING (THE "LEARNING")
# =============================================================================

def train_linear_decoder(latent_codes, pixel_data):
    """
    Solve for weight matrix using the Moore-Penrose pseudoinverse.
    
    The Goal:
    Find matrix W such that:  latent_code @ W = pixels
    
    Traditional AI: Train for hours using gradient descent
    Our approach: Solve analytically in one step!
    
    The Math:
    Given: X (latents) and Y (pixels), find W
    Solution: W = pseudoinverse(X) @ Y
    
    Why this works:
    - For 10 landmarks, we have 10 equations and 128 unknowns
    - System is underdetermined (more unknowns than equations)
    - Pseudoinverse finds the "smoothest" solution
    - Creates a continuous manifold (fills in the gaps)
    
    Args:
        latent_codes: Shape (num_emojis, 128)
        pixel_data: Shape (num_emojis, 3072)
    
    Returns:
        np.ndarray: Weight matrix, shape (128, 3072)
    """
    print("🧠 Training the neural decoder...")
    print(f"   Problem: {latent_codes.shape} → {pixel_data.shape}")
    
    # Center the pixel data (mean = 0)
    # This helps the network learn "deviations from average"
    centered_pixels = pixel_data - 0.5
    
    # Compute pseudoinverse
    print("   Calculating pseudoinverse...")
    latent_pseudoinverse = np.linalg.pinv(latent_codes)
    print(f"   Pseudoinverse shape: {latent_pseudoinverse.shape}")
    
    # Solve for weights
    weight_matrix = latent_pseudoinverse @ centered_pixels
    print(f"   ✓ Weight matrix shape: {weight_matrix.shape}")
    
    # Calculate reconstruction error
    reconstructed = (latent_codes @ weight_matrix) + 0.5
    error = np.mean((reconstructed - pixel_data) ** 2)
    print(f"   ✓ Training error (MSE): {error:.6f}")
    print(f"   ✓ Training complete!\n")
    
    return weight_matrix


# =============================================================================
# STEP 3: THE DECODER FUNCTION
# =============================================================================

def decode_latent(latent_vector, weight_matrix, image_size=32):
    """
    Reconstruct image from latent code.
    
    This is the "inference" step - the AI's output.
    
    The Process:
    1. Multiply latent vector by weight matrix
    2. Add back the mean (0.5)
    3. Reshape flat array back into image
    4. Clip to valid range [0, 1]
    
    Args:
        latent_vector: Shape (128,)
        weight_matrix: Shape (128, 3072)
        image_size: Output image dimension
    
    Returns:
        np.ndarray: RGB image, shape (32, 32, 3)
    """
    # Matrix multiplication: (1, 128) @ (128, 3072) = (1, 3072)
    flat_pixels = (latent_vector @ weight_matrix) + 0.5
    
    # Reshape to image
    image = flat_pixels.reshape(image_size, image_size, 3)
    
    # Clip to valid range
    return np.clip(image, 0, 1)


# =============================================================================
# STEP 4: LATENT INTERPOLATION (THE "HALLUCINATION")
# =============================================================================

def interpolate_latents(latent_a, latent_b, t):
    """
    Linear interpolation between two latent codes.
    
    This is where the magic happens! By interpolating in latent
    space, we create "hybrid" emojis that never existed in training.
    
    The Math:
    z(t) = (1-t) * z_a + t * z_b
    
    When t=0: Full emoji A
    When t=0.5: 50/50 blend
    When t=1: Full emoji B
    
    Args:
        latent_a: First latent vector
        latent_b: Second latent vector
        t: Interpolation parameter [0, 1]
    
    Returns:
        np.ndarray: Interpolated latent vector
    """
    return (1 - t) * latent_a + t * latent_b


# =============================================================================
# VISUALIZATION: RECONSTRUCTION TEST
# =============================================================================

def visualize_reconstruction(dna_array, latent_codes, weight_matrix, labels):
    """
    Compare original emojis vs. neural reconstructions.
    
    This verifies the decoder learned to represent our landmarks.
    """
    num_emojis = len(dna_array)
    
    fig, axes = plt.subplots(2, num_emojis, figsize=(20, 5))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('Neural Reconstruction Test', color='#00FFCC', fontsize=14)
    
    for i in range(num_emojis):
        # Original
        axes[0, i].imshow(dna_array[i])
        axes[0, i].set_title(f'{labels[i]}', color='#00FFCC', fontsize=8)
        axes[0, i].axis('off')
        
        # Reconstructed
        reconstructed = decode_latent(latent_codes[i], weight_matrix)
        axes[1, i].imshow(reconstructed)
        axes[1, i].set_title('Decoded', color='#00FF00', fontsize=8)
        axes[1, i].axis('off')
    
    axes[0, 0].text(-0.5, 0.5, 'ORIGINAL', rotation=90, 
                    va='center', ha='right', color='#00FFCC',
                    fontsize=12, transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.5, 0.5, 'NEURAL', rotation=90,
                    va='center', ha='right', color='#00FF00',
                    fontsize=12, transform=axes[1, 0].transAxes)
    
    plt.tight_layout()
    plt.savefig('tutorial_02_reconstruction.png', dpi=150, facecolor='#0a0a0a')
    print("💾 Saved reconstruction test: tutorial_02_reconstruction.png")
    plt.show()


# =============================================================================
# VISUALIZATION: INTERPOLATION TEST
# =============================================================================

def visualize_interpolation(latent_codes, weight_matrix, labels, pairs):
    """
    Show smooth transitions between emoji pairs.
    
    This demonstrates the continuous nature of the latent space.
    """
    num_pairs = len(pairs)
    steps = 7  # Number of interpolation steps
    
    fig, axes = plt.subplots(num_pairs, steps, figsize=(14, num_pairs * 2.5))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('Latent Space Interpolation', color='#00FFCC', fontsize=14)
    
    for row, (idx_a, idx_b) in enumerate(pairs):
        for col in range(steps):
            t = col / (steps - 1)  # 0.0 to 1.0
            
            # Interpolate
            latent_blend = interpolate_latents(
                latent_codes[idx_a],
                latent_codes[idx_b],
                t
            )
            
            # Decode
            image = decode_latent(latent_blend, weight_matrix)
            
            # Display
            ax = axes[row, col] if num_pairs > 1 else axes[col]
            ax.imshow(image)
            ax.axis('off')
            
            # Label endpoints
            if col == 0:
                ax.set_title(labels[idx_a], color='#00FFCC', fontsize=8)
            elif col == steps - 1:
                ax.set_title(labels[idx_b], color='#00FFCC', fontsize=8)
            else:
                ax.set_title(f'{int(t*100)}%', color='#666', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('tutorial_02_interpolation.png', dpi=150, facecolor='#0a0a0a')
    print("💾 Saved interpolation: tutorial_02_interpolation.png")
    plt.show()


# =============================================================================
# VISUALIZATION: BRAIN STATE (LATENT VECTOR)
# =============================================================================

def visualize_brain_state(latent_vector, title="Brain State"):
    """
    Visualize a latent vector as a bar chart.
    
    This shows which "dimensions" are active for a given emoji.
    - Cyan bars: Positive activation (adding features)
    - Magenta bars: Negative activation (subtracting features)
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    x = np.arange(len(latent_vector))
    colors = ['#00FFCC' if v > 0 else '#FF00FF' for v in latent_vector]
    
    ax.bar(x, latent_vector, color=colors, width=1.0, alpha=0.7)
    ax.axhline(0, color='#333', linewidth=1)
    ax.set_ylim(-3, 3)
    ax.set_xlim(0, len(latent_vector))
    ax.set_title(title, color='#00FFCC', pad=15)
    ax.set_xlabel('Latent Dimension', color='#666')
    ax.set_ylabel('Activation', color='#666')
    ax.tick_params(colors='#444')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#222')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# EDUCATIONAL SUMMARY
# =============================================================================

def print_summary(latent_codes, weight_matrix):
    """Display key statistics about the trained model."""
    print("=" * 70)
    print("📊 LINEAR DECODER SUMMARY")
    print("=" * 70)
    print(f"Latent space:           {latent_codes.shape}")
    print(f"Weight matrix:          {weight_matrix.shape}")
    print(f"Parameters learned:     {weight_matrix.size:,}")
    print(f"Compression ratio:      {PIXEL_COUNT / LATENT_DIMENSIONS:.1f}x")
    print(f"   (3072 pixels → 128 dimensions)")
    print(f"\nModel size:             {weight_matrix.nbytes / 1024:.2f} KB")
    print(f"Inference cost:         1 matrix multiplication")
    print("=" * 70)
    print("\n✅ The 'brain' is trained! It can now hallucinate new emojis.\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TUTORIAL 02: LINEAR DECODER")
    print("=" * 70 + "\n")
    
    # Load DNA from previous tutorial
    if not os.path.exists(DNA_FILE):
        print(f"❌ DNA file not found: {DNA_FILE}")
        print("Please run: python tutorial_01_dna_extraction.py first")
        exit(1)
    
    emoji_dna = np.load(DNA_FILE)
    num_emojis = len(emoji_dna)
    
    # Labels
    emoji_labels = [
        "0: Alien", "1: Ghost", "2: Skull", "3: Swords", "4: Gem",
        "5: Fire", "6: Eye", "7: Clover", "8: Potion", "9: Vortex"
    ]
    
    # Step 1: Initialize latent codes
    latent_codes = initialize_latent_codes(num_emojis, LATENT_DIMENSIONS)
    
    # Step 2: Flatten pixel data for training
    flat_pixels = emoji_dna.reshape(num_emojis, -1)
    
    # Step 3: Train decoder
    weight_matrix = train_linear_decoder(latent_codes, flat_pixels)
    
    # Display summary
    print_summary(latent_codes, weight_matrix)
    
    # Visualize reconstruction
    visualize_reconstruction(emoji_dna, latent_codes, weight_matrix, emoji_labels)
    
    # Visualize interpolation
    test_pairs = [
        (1, 2),  # Ghost → Skull
        (4, 5),  # Gem → Fire
    ]
    visualize_interpolation(latent_codes, weight_matrix, emoji_labels, test_pairs)
    
    # Show an example brain state
    print("\n🧠 Example brain state for 'Ghost' emoji:")
    visualize_brain_state(latent_codes[1], title="Brain State: Ghost")
    
    # Save for next tutorial
    np.save('tutorial_02_latents.npy', latent_codes)
    np.save('tutorial_02_weights.npy', weight_matrix)
    print("\n💾 Saved model: tutorial_02_latents.npy, tutorial_02_weights.npy")
    print("\n🎓 Next: tutorial_03_linearity_trap.py\n")
