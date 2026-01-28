"""
Tutorial 05: Disentanglement - Neural Surgery for Feature Control
==================================================================

This is the final evolution: training TWO specialized decoders that
learn shape and color independently. This enables "impossible" hybrids
like a red diamond or blue fire.

Key Concepts:
- Feature disentanglement (separating concepts into independent dimensions)
- Multi-decoder architecture (parallel specialists, each still linear)
- YUV color space (luminance vs. chrominance separation)
- Vector arithmetic (mix and match features freely)
- Non-linear combination (multiplication of decoder outputs)

The Architecture:
- Structure decoder: latent_shape @ W_structure → grayscale (linear)
- Color decoder: latent_color @ W_color → RGB (linear)
- Final output: structure * color (non-linear combination!)

Why This Works:
- Each decoder is linear (Ridge Regression, instant training)
- But COMBINATION via multiplication is non-linear
- This enables independent control without gradient descent
- Trade-off: Simple training, but limited to predefined feature splits

Connection to Production AI:
- StyleGAN: Learns disentangled style codes automatically
- Stable Diffusion + ControlNet: Separates content from structure
- Your Tutorial: Manually split features, same core principle
- Production: Learns what features to split from data

Educational Goal:
Understanding how modern AI (StyleGAN, Stable Diffusion, ControlNet)
achieves fine-grained control by learning independent feature
representations. You've built the foundation manually!
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
# DATA PREPARATION: SEPARATING STRUCTURE FROM STYLE
# =============================================================================

def extract_structure_channel(images):
    """
    Extract grayscale luminance (the "structure").
    
    Why grayscale:
    - Contains shape, edges, texture information
    - Color-blind representation
    - Formula: Y = 0.299R + 0.587G + 0.114B (human perception-weighted)
    
    This is what we use to train the "Structure Brain."
    
    Args:
        images: Shape (N, H, W, 3) RGB images
    
    Returns:
        np.ndarray: Shape (N, H, W) grayscale images
    """
    print("🔍 Extracting structure channel (grayscale)...")
    
    # Standard luminance formula
    grayscale = np.dot(images, [0.299, 0.587, 0.114])
    
    print(f"   ✓ Structure shape: {grayscale.shape}")
    print(f"   ✓ Value range: [{grayscale.min():.3f}, {grayscale.max():.3f}]\n")
    
    return grayscale


def extract_color_palette(images):
    """
    Extract average color of each emoji (the "style").
    
    Why average color:
    - Represents the dominant hue/tint
    - Position-independent (we only care about the palette, not where)
    - Simple but effective for style transfer
    
    Algorithm:
    1. Mask out black/dark pixels (background)
    2. Average remaining RGB values
    3. Fallback to white if emoji is pure black
    
    This is what we use to train the "Color Brain."
    
    Args:
        images: Shape (N, H, W, 3) RGB images
    
    Returns:
        np.ndarray: Shape (N, 3) average RGB colors
    """
    print("🎨 Extracting color palettes...")
    
    num_images = len(images)
    palettes = []
    
    for i, img in enumerate(images):
        # Create mask: pixels that aren't near-black
        is_foreground = np.any(img > 0.1, axis=-1)
        
        if np.any(is_foreground):
            # Average color of foreground pixels only
            avg_color = img[is_foreground].mean(axis=0)
        else:
            # Fallback for failed renders
            avg_color = np.array([1.0, 1.0, 1.0])
        
        palettes.append(avg_color)
    
    palettes = np.array(palettes)
    
    print(f"   ✓ Palette shape: {palettes.shape}")
    print(f"   ✓ Example colors:")
    for i in range(min(3, num_images)):
        r, g, b = palettes[i]
        print(f"      Emoji {i}: RGB({r:.3f}, {g:.3f}, {b:.3f})")
    print()
    
    return palettes


# =============================================================================
# TRAINING: TWO SPECIALIZED DECODERS
# =============================================================================

def train_structure_decoder(latent_codes, grayscale_data):
    """
    Train decoder: Latent → Grayscale Image
    
    This brain learns:
    - Shapes (circles, diamonds, stars)
    - Edges (boundaries, outlines)
    - Textures (details, patterns)
    
    But NOT colors!
    
    Args:
        latent_codes: Shape (N, 128)
        grayscale_data: Shape (N, H, W)
    
    Returns:
        np.ndarray: Weight matrix (128, H*W)
    """
    print("🧠 Training STRUCTURE decoder...")
    
    num_samples = len(grayscale_data)
    flat_structure = grayscale_data.reshape(num_samples, -1)
    
    print(f"   Input: {latent_codes.shape}")
    print(f"   Output: {flat_structure.shape}")
    
    # Center the data
    centered_structure = flat_structure - 0.5
    
    # Solve for weights
    W_structure = np.linalg.pinv(latent_codes) @ centered_structure
    
    # Calculate error
    reconstructed = (latent_codes @ W_structure) + 0.5
    error = np.mean((reconstructed - flat_structure) ** 2)
    
    print(f"   ✓ Weight matrix: {W_structure.shape}")
    print(f"   ✓ Training error: {error:.6f}\n")
    
    return W_structure


def train_color_decoder(latent_codes, color_palettes):
    """
    Train decoder: Latent → RGB Color Vector
    
    This brain learns:
    - Hues (red, blue, green, etc.)
    - Saturation (vibrancy)
    - Overall tint
    
    But NOT shapes or positions!
    
    Args:
        latent_codes: Shape (N, 128)
        color_palettes: Shape (N, 3)
    
    Returns:
        np.ndarray: Weight matrix (128, 3)
    """
    print("🧠 Training COLOR decoder...")
    
    print(f"   Input: {latent_codes.shape}")
    print(f"   Output: {color_palettes.shape}")
    
    # Center the data
    centered_colors = color_palettes - 0.5
    
    # Solve for weights
    W_color = np.linalg.pinv(latent_codes) @ centered_colors
    
    # Calculate error
    reconstructed = (latent_codes @ W_color) + 0.5
    error = np.mean((reconstructed - color_palettes) ** 2)
    
    print(f"   ✓ Weight matrix: {W_color.shape}")
    print(f"   ✓ Training error: {error:.6f}\n")
    
    return W_color


# =============================================================================
# DISENTANGLED DECODER
# =============================================================================

def decode_disentangled(latent_structure, latent_color, 
                        W_structure, W_color, image_size=32):
    """
    Frankenstein decoder: Mix structure from one emoji, color from another.
    
    The Process:
    1. Decode structure (grayscale) from latent_structure
    2. Decode color (RGB tint) from latent_color
    3. Multiply: final = structure * color
    
    Why multiplication:
    - Structure provides intensity map (0 = black, 1 = white)
    - Color provides RGB tint
    - Multiply → colored structure
    
    This allows impossible combinations:
    - Diamond shape + Fire color = Red Diamond
    - Fire shape + Diamond color = Blue Fire
    
    Args:
        latent_structure: Shape (128,) - controls WHAT
        latent_color: Shape (128,) - controls HOW IT LOOKS
        W_structure: Shape (128, 1024)
        W_color: Shape (128, 3)
        image_size: Output dimension
    
    Returns:
        np.ndarray: RGB image (H, W, 3)
    """
    # Decode structure (grayscale intensity map)
    flat_structure = (latent_structure @ W_structure) + 0.5
    structure_map = flat_structure.reshape(image_size, image_size)
    structure_map = np.clip(structure_map, 0, 1)
    
    # Decode color (single RGB tint)
    color_tint = (latent_color @ W_color) + 0.5
    color_tint = np.clip(color_tint, 0, 1)
    
    # Broadcast: (H, W) * (3,) → (H, W, 3)
    # Structure controls "how much" of each pixel
    # Color controls "what color" each pixel is
    final_image = structure_map[..., np.newaxis] * color_tint
    
    return final_image


# =============================================================================
# VISUALIZATION: THE IMPOSSIBLE HYBRIDS
# =============================================================================

def visualize_frankenstein_gallery(pairs, latent_codes,
                                    W_structure, W_color,
                                    labels, original_images):
    """
    Create a gallery of impossible emoji combinations.
    
    For each pair:
    - Left: Source of structure
    - Middle: The Frankenstein hybrid
    - Right: Source of color
    """
    num_pairs = len(pairs)
    
    fig, axes = plt.subplots(num_pairs, 3, figsize=(10, num_pairs * 3))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle("🧪 Neural Surgery: Disentangled Hybrids",
                 color='#00FFCC', fontsize=16, y=0.98)
    
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for row, (idx_structure, idx_color) in enumerate(pairs):
        # Get latent codes
        latent_struct = latent_codes[idx_structure]
        latent_col = latent_codes[idx_color]
        
        # Generate hybrid
        hybrid = decode_disentangled(
            latent_struct, latent_col,
            W_structure, W_color
        )
        
        # Display
        # Left: Structure source
        axes[row, 0].imshow(original_images[idx_structure])
        axes[row, 0].set_title(
            f"Structure from:\n{labels[idx_structure]}",
            color='#00FFCC', fontsize=9
        )
        axes[row, 0].axis('off')
        
        # Middle: Hybrid
        axes[row, 1].imshow(hybrid)
        axes[row, 1].set_title(
            "⚗️ HYBRID",
            color='#FFDD00', fontsize=11, fontweight='bold'
        )
        axes[row, 1].axis('off')
        
        # Right: Color source
        axes[row, 2].imshow(original_images[idx_color])
        axes[row, 2].set_title(
            f"Color from:\n{labels[idx_color]}",
            color='#FF00FF', fontsize=9
        )
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('tutorial_05_disentanglement.png', dpi=150, facecolor='#0a0a0a')
    print("💾 Saved gallery: tutorial_05_disentanglement.png")
    plt.show()


# =============================================================================
# VISUALIZATION: FEATURE SEPARATION
# =============================================================================

def visualize_feature_separation(idx, latent_codes, W_structure, W_color,
                                  labels, original_images):
    """
    Show how one emoji is decomposed into structure + color.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle(f"Feature Decomposition: {labels[idx]}",
                 color='#00FFCC', fontsize=14)
    
    # Original
    axes[0].imshow(original_images[idx])
    axes[0].set_title("ORIGINAL", color='#00FFCC')
    axes[0].axis('off')
    
    # Structure only (grayscale)
    latent = latent_codes[idx]
    structure = (latent @ W_structure) + 0.5
    structure_img = structure.reshape(IMAGE_SIZE, IMAGE_SIZE)
    axes[1].imshow(structure_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("STRUCTURE\n(Shape only)", color='#FFFFFF')
    axes[1].axis('off')
    
    # Color only (solid patch)
    color = (latent @ W_color) + 0.5
    color_patch = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3)) * color
    axes[2].imshow(np.clip(color_patch, 0, 1))
    axes[2].set_title("COLOR\n(Palette only)", color='#FF00FF')
    axes[2].axis('off')
    
    # Reconstructed (structure × color)
    reconstructed = decode_disentangled(latent, latent, W_structure, W_color)
    axes[3].imshow(reconstructed)
    axes[3].set_title("RECONSTRUCTED\n(Structure × Color)", color='#00FF00')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'tutorial_05_decomposition_{idx}.png', dpi=150, facecolor='#0a0a0a')
    print(f"💾 Saved decomposition: tutorial_05_decomposition_{idx}.png")
    plt.show()


# =============================================================================
# EDUCATIONAL SUMMARY
# =============================================================================

def print_summary(W_structure, W_color):
    """Display disentangled model statistics."""
    print("=" * 70)
    print("📊 DISENTANGLED DECODER SUMMARY")
    print("=" * 70)
    print(f"Architecture:           Dual-Decoder (Parallel Specialists)")
    print(f"  Structure Decoder:    {W_structure.shape}")
    print(f"  Color Decoder:        {W_color.shape}")
    print(f"\nTotal parameters:       {W_structure.size + W_color.size:,}")
    print(f"Model size:             {(W_structure.nbytes + W_color.nbytes) / 1024:.2f} KB")
    print(f"\nCapabilities:")
    print(f"  ✓ Mix shape from emoji A with color from emoji B")
    print(f"  ✓ Create impossible combinations (red diamond, blue fire)")
    print(f"  ✓ Independent control over structure and style")
    print(f"\nReal-world equivalent:")
    print(f"  - StyleGAN: Controls pose, expression, lighting separately")
    print(f"  - Stable Diffusion: ControlNet separates content from style")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TUTORIAL 05: DISENTANGLEMENT")
    print("=" * 70 + "\n")
    
    # Load data
    emoji_dna, latent_codes = load_data()
    
    # Labels
    emoji_labels = [
        "0: Alien", "1: Ghost", "2: Skull", "3: Swords", "4: Gem",
        "5: Fire", "6: Eye", "7: Clover", "8: Potion", "9: Vortex"
    ]
    
    # Step 1: Extract features
    structure_data = extract_structure_channel(emoji_dna)
    color_data = extract_color_palette(emoji_dna)
    
    # Step 2: Train two decoders
    W_structure = train_structure_decoder(latent_codes, structure_data)
    W_color = train_color_decoder(latent_codes, color_data)
    
    # Display summary
    print_summary(W_structure, W_color)
    
    # Step 3: Visualize decomposition
    print("🔬 Showing feature decomposition for Fire emoji...")
    visualize_feature_separation(5, latent_codes, W_structure, W_color,
                                  emoji_labels, emoji_dna)
    print()
    
    # Step 4: Create impossible hybrids
    print("⚗️  Creating impossible hybrid combinations...")
    frankenstein_pairs = [
        (4, 5),  # Diamond structure + Fire color = RED DIAMOND
        (5, 4),  # Fire structure + Diamond color = BLUE FIRE
        (1, 6),  # Ghost structure + Eye color = FLESH GHOST
        (6, 2),  # Eye structure + Skull color = WHITE EYE
    ]
    
    visualize_frankenstein_gallery(
        frankenstein_pairs,
        latent_codes,
        W_structure, W_color,
        emoji_labels,
        emoji_dna
    )
    
    # Save models
    np.save('tutorial_05_structure_weights.npy', W_structure)
    np.save('tutorial_05_color_weights.npy', W_color)
    print("\n💾 Saved decoders: tutorial_05_structure_weights.npy, tutorial_05_color_weights.npy")
    
    print("\n✅ Disentanglement complete!")
    print("\n🎓 Key Takeaway:")
    print("   By training separate decoders for shape and color, we achieve")
    print("   surgical control over features. This is the foundation of modern")
    print("   style transfer and controllable generation systems.")
    print("\n🏆 Tutorial Series Complete!")
    print("   You've built a neural emoji lab from scratch, learning:")
    print("   1. Feature engineering (DNA extraction)")
    print("   2. Linear algebra (analytical training)")
    print("   3. Model limitations (linearity trap)")
    print("   4. Non-linearity (deep networks)")
    print("   5. Disentanglement (feature control)")
    print("\n   These concepts power modern generative AI! 🚀\n")
