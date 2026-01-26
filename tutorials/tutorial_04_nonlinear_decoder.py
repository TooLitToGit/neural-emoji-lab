"""
Tutorial 04: Non-Linear Decoder - Breaking the Linearity Trap
==============================================================

This tutorial implements a 2-layer neural network using the Extreme
Learning Machine (ELM) technique with high-gain Tanh activation.
This fixes the "ghosting" problem from the linear model.

Key Concepts:
- Hidden layers (expansion into higher dimensions)
- Non-linear activation (Tanh function)
- High-gain initialization (forcing hard decisions)
- Extreme Learning Machine (instant training trick)

Educational Goal:
Understanding why deep learning works - non-linearity enables true
feature morphing instead of simple cross-fading.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

IMAGE_SIZE = 32
LATENT_DIM = 128
HIDDEN_DIM = 2048  # Expansion factor: 16x

DNA_FILE = 'tutorial_01_dna.npy'
LATENTS_FILE = 'tutorial_02_latents.npy'


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load DNA and latent codes from previous tutorials."""
    print("📦 Loading training data...")
    
    if not all(os.path.exists(f) for f in [DNA_FILE, LATENTS_FILE]):
        print("❌ Missing required files. Please run tutorials 01 and 02 first.")
        exit(1)
    
    emoji_dna = np.load(DNA_FILE)
    latent_codes = np.load(LATENTS_FILE)
    
    print(f"   ✓ DNA library: {emoji_dna.shape}")
    print(f"   ✓ Latent codes: {latent_codes.shape}\n")
    
    return emoji_dna, latent_codes


# =============================================================================
# ACTIVATION FUNCTION
# =============================================================================

def tanh_activation(x):
    """
    Hyperbolic Tangent: The 'decisiveness' function.
    
    Why Tanh instead of ReLU:
    - ReLU: max(0, x) → deletes negative values
    - Tanh: (e^x - e^-x) / (e^x + e^-x) → symmetric [-1, 1]
    
    For image data centered at 0.5:
    - Black pixels are negative (0.0 - 0.5 = -0.5)
    - White pixels are positive (1.0 - 0.5 = +0.5)
    
    Tanh preserves both, ReLU would turn all blacks to grey.
    
    The Saturation Property:
    - Small inputs (|x| < 1): Linear-like response
    - Large inputs (|x| > 2): Saturates at ±1
    - This creates 'hard' decisions instead of soft blends
    """
    return np.tanh(x)


def visualize_activation_function():
    """Show why Tanh is better than ReLU for our use case."""
    x = np.linspace(-3, 3, 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Tanh
    ax1.plot(x, np.tanh(x), color='#00FFCC', linewidth=2)
    ax1.axhline(0, color='#333', linewidth=1, linestyle='--')
    ax1.axvline(0, color='#333', linewidth=1, linestyle='--')
    ax1.set_facecolor('#0f0f0f')
    ax1.set_title('Tanh Activation\n(Preserves negative values)', color='#00FFCC')
    ax1.set_xlabel('Input', color='#666')
    ax1.set_ylabel('Output', color='#666')
    ax1.grid(True, alpha=0.1)
    ax1.tick_params(colors='#444')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#222')
    
    # ReLU for comparison
    relu = np.maximum(0, x)
    ax2.plot(x, relu, color='#FF6666', linewidth=2)
    ax2.axhline(0, color='#333', linewidth=1, linestyle='--')
    ax2.axvline(0, color='#333', linewidth=1, linestyle='--')
    ax2.set_facecolor('#0f0f0f')
    ax2.set_title('ReLU (Would lose darkness)', color='#FF6666')
    ax2.set_xlabel('Input', color='#666')
    ax2.set_ylabel('Output', color='#666')
    ax2.grid(True, alpha=0.1)
    ax2.tick_params(colors='#444')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#222')
    
    plt.tight_layout()
    plt.savefig('tutorial_04_activation.png', dpi=150, facecolor='#0a0a0a')
    print("💾 Saved activation comparison: tutorial_04_activation.png")
    plt.show()


# =============================================================================
# LAYER 1: RANDOM EXPANSION (THE "HIDDEN" LAYER)
# =============================================================================

def initialize_hidden_layer(latent_dim, hidden_dim, gain=1.5, seed=42):
    """
    Create the first layer with high-gain random weights.
    
    The Extreme Learning Machine Trick:
    - Layer 1: Random, never trained
    - Layer 2: Solved analytically (like tutorial 02)
    
    Why this works:
    - Random projections preserve information (Johnson-Lindenstrauss)
    - High dimensionality (2048) provides capacity
    - Non-linearity (Tanh) creates complex feature space
    
    The High-Gain Trick:
    Multiply weights by 1.5 to force saturation.
    
    Example:
    - Low gain (0.1): tanh(0.1) ≈ 0.099 (linear-like)
    - High gain (1.5): tanh(1.5) ≈ 0.905 (binary-like)
    
    This makes the network "opinionated" - it makes hard decisions
    about which features to activate, reducing the ghosting effect.
    
    Args:
        latent_dim: Input size (128)
        hidden_dim: Hidden layer size (2048)
        gain: Multiplicative factor for initialization
        seed: Random seed
    
    Returns:
        np.ndarray: Random weight matrix (128, 2048)
    """
    print(f"🔮 Initializing hidden layer with high-gain ({gain}x)...")
    
    np.random.seed(seed)
    W_hidden = np.random.randn(latent_dim, hidden_dim) * gain
    
    print(f"   ✓ Weight matrix: {W_hidden.shape}")
    print(f"   ✓ Value range: [{W_hidden.min():.2f}, {W_hidden.max():.2f}]")
    print(f"   ✓ This ensures strong neuron saturation\n")
    
    return W_hidden


# =============================================================================
# LAYER 2: ANALYTICAL SOLUTION (THE "OUTPUT" LAYER)
# =============================================================================

def train_output_layer(latent_codes, W_hidden, pixel_data):
    """
    Solve for the output layer weights using pseudoinverse.
    
    The Process:
    1. Forward pass: latents → hidden activations
    2. Solve: hidden @ W_output = pixels
    
    This is identical to tutorial 02, but now we're working in
    a non-linear feature space (the hidden activations) instead
    of the raw latent space.
    
    Args:
        latent_codes: Shape (10, 128)
        W_hidden: Shape (128, 2048)
        pixel_data: Shape (10, 3072)
    
    Returns:
        tuple: (hidden_activations, W_output)
    """
    print("🧠 Training output layer...")
    
    # Forward pass through hidden layer
    hidden_pre = latent_codes @ W_hidden
    hidden_activations = tanh_activation(hidden_pre)
    
    print(f"   Hidden activations: {hidden_activations.shape}")
    print(f"   Activation range: [{hidden_activations.min():.3f}, {hidden_activations.max():.3f}]")
    
    # Solve for output weights
    centered_pixels = pixel_data - 0.5
    W_output = np.linalg.pinv(hidden_activations, rcond=1e-5) @ centered_pixels
    
    print(f"   ✓ Output weights: {W_output.shape}")
    
    # Calculate training error
    reconstructed = (hidden_activations @ W_output) + 0.5
    error = np.mean((reconstructed - pixel_data) ** 2)
    print(f"   ✓ Training error (MSE): {error:.6f}\n")
    
    return hidden_activations, W_output


# =============================================================================
# NON-LINEAR DECODER
# =============================================================================

def decode_nonlinear(latent_vector, W_hidden, W_output, image_size=32):
    """
    Two-layer decoder with Tanh non-linearity.
    
    The Forward Pass:
    1. Linear projection: latent @ W_hidden → pre-activation
    2. Non-linear activation: tanh(pre-activation) → hidden state
    3. Linear projection: hidden @ W_output → pixels
    4. Shift and clip: add 0.5, clip to [0, 1]
    
    This creates a non-linear manifold where interpolation
    produces smooth feature morphs instead of ghosting.
    
    Args:
        latent_vector: Shape (128,)
        W_hidden: Shape (128, 2048)
        W_output: Shape (2048, 3072)
        image_size: Output dimension
    
    Returns:
        np.ndarray: RGB image (32, 32, 3)
    """
    # Layer 1: Expansion + Activation
    hidden_pre = latent_vector @ W_hidden
    hidden_state = tanh_activation(hidden_pre)
    
    # Layer 2: Projection to pixels
    flat_pixels = (hidden_state @ W_output) + 0.5
    
    # Reshape and clip
    image = flat_pixels.reshape(image_size, image_size, 3)
    return np.clip(image, 0, 1)


# =============================================================================
# COMPARISON VISUALIZATION
# =============================================================================

def compare_linear_vs_nonlinear(pairs, latent_codes, 
                                 W_linear, W_hidden, W_output, labels):
    """
    Side-by-side comparison of linear vs. non-linear interpolation.
    
    For each pair, show:
    - Parent A
    - Linear 50% (ghosting)
    - Non-linear 50% (cleaner)
    - Parent B
    """
    num_pairs = len(pairs)
    
    fig, axes = plt.subplots(num_pairs, 4, figsize=(14, num_pairs * 3.5))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('Linear vs. Non-Linear Interpolation @ 50%',
                 color='#00FFCC', fontsize=16, y=0.98)
    
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for row, (idx_a, idx_b) in enumerate(pairs):
        vec_a = latent_codes[idx_a]
        vec_b = latent_codes[idx_b]
        vec_mid = 0.5 * vec_a + 0.5 * vec_b
        
        # Linear decoder
        img_linear_mid = decode_linear(vec_mid, W_linear)
        
        # Non-linear decoder
        img_nonlinear_mid = decode_nonlinear(vec_mid, W_hidden, W_output)
        
        # Parent images (using non-linear for consistency)
        img_a = decode_nonlinear(vec_a, W_hidden, W_output)
        img_b = decode_nonlinear(vec_b, W_hidden, W_output)
        
        # Display
        images = [
            (img_a, f"Parent A\n{labels[idx_a]}", '#00FFCC'),
            (img_linear_mid, "LINEAR\n(Ghosting)", '#FF6666'),
            (img_nonlinear_mid, "NON-LINEAR\n(Clean)", '#00FF00'),
            (img_b, f"Parent B\n{labels[idx_b]}", '#00FFCC')
        ]
        
        for col, (img, title, color) in enumerate(images):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(title, color=color, fontsize=10, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('tutorial_04_comparison.png', dpi=150, facecolor='#0a0a0a')
    print("💾 Saved comparison: tutorial_04_comparison.png")
    plt.show()


def decode_linear(latent_vector, weight_matrix, image_size=32):
    """Linear decoder from tutorial 02 (for comparison)."""
    flat_pixels = (latent_vector @ weight_matrix) + 0.5
    return np.clip(flat_pixels.reshape(image_size, image_size, 3), 0, 1)


# =============================================================================
# EDUCATIONAL SUMMARY
# =============================================================================

def print_summary(W_hidden, W_output):
    """Display model architecture and statistics."""
    total_params = W_hidden.size + W_output.size
    
    print("=" * 70)
    print("📊 NON-LINEAR DECODER SUMMARY")
    print("=" * 70)
    print(f"Architecture:           2-Layer ELM with Tanh")
    print(f"  Layer 1 (Hidden):     {W_hidden.shape[0]} → {W_hidden.shape[1]}")
    print(f"  Activation:           Tanh (high-gain 1.5x)")
    print(f"  Layer 2 (Output):     {W_output.shape[0]} → {W_output.shape[1]}")
    print(f"\nTotal parameters:       {total_params:,}")
    print(f"Model size:             {(W_hidden.nbytes + W_output.nbytes) / 1024:.2f} KB")
    print(f"Inference cost:         2 matrix multiplications + 1 tanh")
    print("\nKey Improvement:        Non-linear feature manifold")
    print("Result:                 Smooth morphs, no ghosting")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TUTORIAL 04: NON-LINEAR DECODER")
    print("=" * 70 + "\n")
    
    # Load data
    emoji_dna, latent_codes = load_data()
    num_emojis = len(emoji_dna)
    flat_pixels = emoji_dna.reshape(num_emojis, -1)
    
    # Labels
    emoji_labels = [
        "0: Alien", "1: Ghost", "2: Skull", "3: Swords", "4: Gem",
        "5: Fire", "6: Eye", "7: Clover", "8: Potion", "9: Vortex"
    ]
    
    # Show activation function
    print("📈 Visualizing activation function...")
    visualize_activation_function()
    print()
    
    # Initialize hidden layer
    W_hidden = initialize_hidden_layer(LATENT_DIM, HIDDEN_DIM, gain=1.5)
    
    # Train output layer
    hidden_activations, W_output = train_output_layer(
        latent_codes, W_hidden, flat_pixels
    )
    
    # Display summary
    print_summary(W_hidden, W_output)
    
    # Load linear weights for comparison
    if os.path.exists('tutorial_02_weights.npy'):
        W_linear = np.load('tutorial_02_weights.npy')
        
        # Compare on test pairs
        test_pairs = [
            (1, 2),  # Ghost + Skull
            (5, 4),  # Fire + Gem
            (0, 6),  # Alien + Eye
        ]
        
        compare_linear_vs_nonlinear(
            test_pairs, latent_codes,
            W_linear, W_hidden, W_output,
            emoji_labels
        )
    else:
        print("⚠️  Linear weights not found, skipping comparison")
    
    # Save model
    np.save('tutorial_04_hidden_weights.npy', W_hidden)
    np.save('tutorial_04_output_weights.npy', W_output)
    print("\n💾 Saved model: tutorial_04_hidden_weights.npy, tutorial_04_output_weights.npy")
    
    print("\n✅ Non-linear decoder trained!")
    print("\n🎓 Key Takeaway:")
    print("   Adding a single hidden layer with non-linear activation")
    print("   transforms cross-fading into true feature morphing.")
    print("   The 2048-dimensional hidden space creates a richer manifold.")
    print("\n🎓 Next: tutorial_05_disentanglement.py")
    print("   (Separate shape from color for ultimate control!)\n")
