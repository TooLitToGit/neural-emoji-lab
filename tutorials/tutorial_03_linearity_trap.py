"""
Tutorial 03: The Linearity Trap - Understanding Model Limitations
==================================================================

This tutorial demonstrates a fundamental limitation of linear models:
they can only perform "cross-fades," not true morphing. By freezing
interpolation at the 50% midpoint, we reveal the "ghosting" effect.

Key Concepts:
- Linear superposition (A + B = blended transparency, not morphing)
- Midpoint analysis (diagnostic tool for understanding limitations)
- Visual artifacts (ghosting effect from weighted averaging)
- Why non-linearity is necessary (motivation for deep learning)

The Fundamental Problem:
- Linear models: decode(A + B) = decode(A) + decode(B)
- This means 50% blend = 50% transparent A + 50% transparent B
- No feature recombination, no conditional logic ("IF this AND that")
- Cannot learn: XOR, hierarchical features, or true morphing

Why Deep Learning Exists:
Non-linear activation functions break this constraint, enabling:
- Feature composition ("has eyes AND nose AND mouth = face")
- True morphing (smooth shape transitions, not just opacity blending)
- Conditional operations ("IF roundness > 0.5 THEN activate center")

Educational Goal:
Understanding limitations is as important as understanding capabilities.
This tutorial proves we have a continuous manifold, but shows why GANs
and Diffusion Models MUST use non-linear architectures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

IMAGE_SIZE = 32
DNA_FILE = 'tutorial_01_dna.npy'
LATENTS_FILE = 'tutorial_02_latents.npy'
WEIGHTS_FILE = 'tutorial_02_weights.npy'


# =============================================================================
# LOAD TRAINED MODEL
# =============================================================================

def load_model():
    """Load the DNA and trained model from previous tutorials."""
    print("📦 Loading trained model...")
    
    if not all(os.path.exists(f) for f in [DNA_FILE, LATENTS_FILE, WEIGHTS_FILE]):
        print("❌ Missing required files. Please run tutorials 01 and 02 first.")
        exit(1)
    
    emoji_dna = np.load(DNA_FILE)
    latent_codes = np.load(LATENTS_FILE)
    weight_matrix = np.load(WEIGHTS_FILE)
    
    print(f"   ✓ DNA library: {emoji_dna.shape}")
    print(f"   ✓ Latent codes: {latent_codes.shape}")
    print(f"   ✓ Weight matrix: {weight_matrix.shape}\n")
    
    return emoji_dna, latent_codes, weight_matrix


# =============================================================================
# DECODER (Same as Tutorial 02)
# =============================================================================

def decode_latent(latent_vector, weight_matrix, image_size=32):
    """Reconstruct image from latent code."""
    flat_pixels = (latent_vector @ weight_matrix) + 0.5
    image = flat_pixels.reshape(image_size, image_size, 3)
    return np.clip(image, 0, 1)


# =============================================================================
# THE MIDPOINT TEST
# =============================================================================

def analyze_midpoint(latent_a, latent_b, weight_matrix):
    """
    Freeze interpolation at t=0.5 and analyze the result.
    
    The Hypothesis:
    If the model were truly "morphing," the 50% blend would show
    a coherent hybrid with merged features.
    
    The Reality:
    Due to linear superposition, we get A + B at 50% opacity,
    creating a "ghosting" effect (two semi-transparent overlays).
    
    Why This Happens (The Math):
    z_mid = 0.5 * z_a + 0.5 * z_b              (blend latent codes)
    decode(z_mid) = (0.5 * z_a + 0.5 * z_b) @ W  (linear operation)
                  = 0.5 * (z_a @ W) + 0.5 * (z_b @ W)  (distributive property)
                  = 0.5 * decode(z_a) + 0.5 * decode(z_b)
    
    This is LITERAL pixel averaging - a cross-fade, not a morph!
    
    Why Can't We Solve This Directly?
    - For non-linear systems (with tanh, relu): No closed-form solution exists
    - Would need gradient descent (1000s of iterations) to find weights
    - The non-linearity breaks the linear algebra - can't isolate weights
    - But the payoff: true feature morphing instead of ghosting
    
    Returns:
        tuple: (image_a, image_mid, image_b, latent_mid)
    """
    # Compute midpoint in latent space
    latent_mid = 0.5 * latent_a + 0.5 * latent_b
    
    # Decode all three points
    image_a = decode_latent(latent_a, weight_matrix)
    image_b = decode_latent(latent_b, weight_matrix)
    image_mid = decode_latent(latent_mid, weight_matrix)
    
    return image_a, image_mid, image_b, latent_mid


# =============================================================================
# VISUALIZATION: SIDE-BY-SIDE COMPARISON
# =============================================================================

def visualize_linearity_trap(pairs, latent_codes, weight_matrix, labels):
    """
    Display the midpoint analysis for multiple emoji pairs.
    
    Layout:
    - Row 1: Ghost + Skull (similar shapes)
    - Row 2: Fire + Gem (contrasting colors)
    - Row 3: Alien + Eye (different features)
    
    For each pair:
    - Col 1: Parent A + Brain State
    - Col 2: 50% Hybrid + Brain State
    - Col 3: Parent B + Brain State
    """
    num_pairs = len(pairs)
    
    # Create grid: 3 rows × 6 columns
    # Columns: [Img A | Brain A | Img Hybrid | Brain Hybrid | Img B | Brain B]
    fig = plt.figure(figsize=(18, num_pairs * 3))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('The Linearity Trap: Frozen at 50% Interpolation',
                 color='#00FFCC', fontsize=16, y=0.98)
    
    gs = gridspec.GridSpec(
        num_pairs, 6,
        width_ratios=[1, 1.5, 1, 1.5, 1, 1.5],
        wspace=0.3,
        hspace=0.4
    )
    
    for row, (idx_a, idx_b) in enumerate(pairs):
        # Get latent vectors
        vec_a = latent_codes[idx_a]
        vec_b = latent_codes[idx_b]
        
        # Run midpoint analysis
        img_a, img_mid, img_b, vec_mid = analyze_midpoint(
            vec_a, vec_b, weight_matrix
        )
        
        # Package for iteration
        stages = [
            (f"Parent A\n{labels[idx_a]}", img_a, vec_a, '#00FFCC'),
            ("50% HYBRID\n(Linear Blend)", img_mid, vec_mid, '#FFDD00'),
            (f"Parent B\n{labels[idx_b]}", img_b, vec_b, '#00FFCC')
        ]
        
        for col_group, (title, img, vec, color) in enumerate(stages):
            # Image subplot
            ax_img = fig.add_subplot(gs[row, col_group * 2])
            ax_img.imshow(img)
            ax_img.set_title(title, color=color, fontsize=9, fontweight='bold')
            ax_img.axis('off')
            
            # Brain state subplot
            ax_brain = fig.add_subplot(gs[row, col_group * 2 + 1])
            ax_brain.set_facecolor('#0f0f0f')
            ax_brain.set_ylim(-3, 3)
            ax_brain.set_xlim(0, 128)
            
            # Plot bars
            colors_list = ['#00FFCC' if v > 0 else '#FF00FF' for v in vec]
            bars = ax_brain.bar(range(128), vec, width=1.0, color=colors_list, alpha=0.6)
            
            # Zero line
            ax_brain.axhline(0, color='#333', linewidth=1)
            ax_brain.axis('off')
            
            # Label the midpoint brain
            if col_group == 1:
                ax_brain.text(
                    64, 3.5, 
                    '← Exact average of A and B →',
                    ha='center', va='bottom',
                    color='#FFDD00', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='#0a0a0a', edgecolor='#FFDD00')
                )
    
    plt.tight_layout()
    plt.savefig('tutorial_03_linearity_trap.png', dpi=150, facecolor='#0a0a0a')
    print("💾 Saved visualization: tutorial_03_linearity_trap.png")
    plt.show()


# =============================================================================
# ANALYSIS: MATHEMATICAL PROOF
# =============================================================================

def prove_linear_superposition(latent_a, latent_b, weight_matrix):
    """
    Numerically verify the linear superposition property.
    
    Test: decode(A + B) == decode(A) + decode(B)
    
    If this holds true (within floating point error), our model
    is purely linear and cannot create non-linear morphs.
    """
    print("🔬 Testing Linear Superposition Property")
    print("=" * 70)
    
    # Method 1: Decode the blended latent
    latent_blend = 0.5 * latent_a + 0.5 * latent_b
    img_method1 = decode_latent(latent_blend, weight_matrix)
    
    # Method 2: Decode separately then blend
    img_a = decode_latent(latent_a, weight_matrix)
    img_b = decode_latent(latent_b, weight_matrix)
    img_method2 = 0.5 * img_a + 0.5 * img_b
    
    # Calculate difference
    difference = np.abs(img_method1 - img_method2)
    max_error = np.max(difference)
    mean_error = np.mean(difference)
    
    print(f"Method 1: decode(0.5*A + 0.5*B)")
    print(f"Method 2: 0.5*decode(A) + 0.5*decode(B)")
    print(f"\nMaximum pixel difference: {max_error:.10f}")
    print(f"Mean pixel difference:    {mean_error:.10f}")
    
    if max_error < 1e-6:
        print("\n✅ CONFIRMED: Model obeys linear superposition")
        print("   This explains the 'ghosting' effect in hybrids.")
    else:
        print("\n⚠️  Non-linear behavior detected")
    
    print("=" * 70 + "\n")


# =============================================================================
# ANALYSIS: FEATURE BREAKDOWN
# =============================================================================

def analyze_ghosting_effect(img_a, img_b, img_hybrid):
    """
    Analyze what "ghosting" looks like in pixel space.
    
    We'll look at:
    - Brightness (should be exactly average)
    - Color distribution (should be mathematically blended)
    - Spatial patterns (features from both parents visible)
    """
    print("👻 Analyzing the Ghosting Effect")
    print("=" * 70)
    
    # Brightness analysis
    brightness_a = np.mean(img_a)
    brightness_b = np.mean(img_b)
    brightness_hybrid = np.mean(img_hybrid)
    brightness_expected = (brightness_a + brightness_b) / 2
    
    print(f"Average Brightness:")
    print(f"   Parent A:        {brightness_a:.4f}")
    print(f"   Parent B:        {brightness_b:.4f}")
    print(f"   Hybrid (actual): {brightness_hybrid:.4f}")
    print(f"   Expected (A+B)/2: {brightness_expected:.4f}")
    print(f"   Difference:      {abs(brightness_hybrid - brightness_expected):.6f}")
    
    # Color analysis
    color_a = np.mean(img_a, axis=(0, 1))
    color_b = np.mean(img_b, axis=(0, 1))
    color_hybrid = np.mean(img_hybrid, axis=(0, 1))
    color_expected = (color_a + color_b) / 2
    
    print(f"\nAverage RGB Color:")
    print(f"   Parent A:  [{color_a[0]:.3f}, {color_a[1]:.3f}, {color_a[2]:.3f}]")
    print(f"   Parent B:  [{color_b[0]:.3f}, {color_b[1]:.3f}, {color_b[2]:.3f}]")
    print(f"   Hybrid:    [{color_hybrid[0]:.3f}, {color_hybrid[1]:.3f}, {color_hybrid[2]:.3f}]")
    print(f"   Expected:  [{color_expected[0]:.3f}, {color_expected[1]:.3f}, {color_expected[2]:.3f}]")
    
    print("\n📊 Conclusion: The hybrid is a perfect mathematical average.")
    print("   No feature 'morphing' - just transparency blending.")
    print("=" * 70 + "\n")


# =============================================================================
# EDUCATIONAL COMPARISON TABLE
# =============================================================================

def print_architecture_comparison():
    """
    Compare linear vs. non-linear architectures.
    """
    print("📚 WHY DEEP LEARNING EXISTS")
    print("=" * 70)
    print("\n┌─────────────────────┬──────────────────┬─────────────────────┐")
    print("│ Feature             │ Linear Model     │ Deep Non-Linear     │")
    print("├─────────────────────┼──────────────────┼─────────────────────┤")
    print("│ Math                │ y = Wx           │ y = f(f(...f(Wx)))  │")
    print("│ Layers              │ 1                │ Many (10-100+)      │")
    print("│ Training            │ Analytical       │ Gradient Descent    │")
    print("│ Speed               │ Instant          │ Hours/Days          │")
    print("│ Interpolation       │ Cross-fade       │ Feature Morph       │")
    print("│ Artifacts           │ Ghosting         │ Minimal             │")
    print("│ Use Case            │ Compression      │ Generation          │")
    print("└─────────────────────┴──────────────────┴─────────────────────┘\n")
    print("Our linear model is perfect for understanding fundamentals,")
    print("but real generative AI (GANs, Diffusion) needs non-linearity")
    print("to create smooth, artifact-free transitions.")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TUTORIAL 03: THE LINEARITY TRAP")
    print("=" * 70 + "\n")
    
    # Load model
    emoji_dna, latent_codes, weight_matrix = load_model()
    
    # Labels
    emoji_labels = [
        "0: Alien", "1: Ghost", "2: Skull", "3: Swords", "4: Gem",
        "5: Fire", "6: Eye", "7: Clover", "8: Potion", "9: Vortex"
    ]
    
    # Test pairs (chosen to highlight different artifacts)
    test_pairs = [
        (1, 2),  # Ghost + Skull (similar shapes → soft ghosting)
        (5, 4),  # Fire + Gem (red + blue → purple math)
        (0, 6),  # Alien + Eye (distinct features → obvious overlay)
    ]
    
    # Main visualization
    visualize_linearity_trap(test_pairs, latent_codes, weight_matrix, emoji_labels)
    
    # Mathematical proof
    prove_linear_superposition(
        latent_codes[1],  # Ghost
        latent_codes[2],  # Skull
        weight_matrix
    )
    
    # Detailed analysis of one pair
    img_a, img_mid, img_b, _ = analyze_midpoint(
        latent_codes[5],  # Fire
        latent_codes[4],  # Gem
        weight_matrix
    )
    analyze_ghosting_effect(img_a, img_b, img_mid)
    
    # Educational comparison
    print_architecture_comparison()
    
    print("✅ Analysis complete!")
    print("\n🎓 Key Takeaway:")
    print("   Linear models can create 'hybrids' through superposition,")
    print("   but they cannot perform true feature-level morphing.")
    print("   This limitation motivated the development of deep networks.")
    print("\n🎓 Next: tutorial_04_nonlinear_decoder.py")
    print("   (We'll fix the ghosting with a 2-layer network!)\n")
