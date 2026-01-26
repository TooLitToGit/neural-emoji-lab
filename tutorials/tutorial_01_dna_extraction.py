"""
Tutorial 01: DNA Extraction - Establishing the Genetic Ground Truth
=====================================================================

Before a neural network can learn, it needs clean, consistent training data.
This tutorial demonstrates how to extract "pure" emoji representations from
a font file, ensuring perfect alignment and normalization.

Key Concepts:
- Strike discovery (finding native bitmap resolution)
- Alpha-based cropping (removing empty space)
- Symmetric normalization (centering with consistent padding)
- Tensor conversion (images → numbers)

Educational Goal:
Understanding that AI training quality depends on data preprocessing.
"Garbage in, garbage out" - we must eliminate positional noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import subprocess


# =============================================================================
# CONFIGURATION
# =============================================================================

IMAGE_SIZE = 32  # Final normalized size (32x32 pixels)
FONT_PATH = "NotoColorEmoji.ttf"
FONT_URL = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"

# The 10 emojis that will serve as our "Genetic Landmarks"
# These are chosen to align with digits 0-9 for π navigation
EMOJI_LANDMARKS = [
    "👾",  # 0: Alien
    "👻",  # 1: Ghost
    "💀",  # 2: Skull
    "⚔️",  # 3: Swords
    "💎",  # 4: Gem
    "🔥",  # 5: Fire
    "🧿",  # 6: Eye
    "🍀",  # 7: Clover
    "🧪",  # 8: Potion
    "🌀",  # 9: Vortex
]

EMOJI_LABELS = [
    "0: Alien", "1: Ghost", "2: Skull", "3: Swords", "4: Gem",
    "5: Fire", "6: Eye", "7: Clover", "8: Potion", "9: Vortex"
]


# =============================================================================
# STEP 0: FONT DOWNLOAD
# =============================================================================

def download_font_if_needed():
    """
    Download the Noto Color Emoji font if not already present.
    
    Why we need this font:
    - System fonts vary across platforms (Windows, Mac, Linux)
    - Noto Color Emoji is consistent and open-source
    - Contains high-quality bitmap strikes for rendering
    
    This ensures everyone gets identical results regardless of their OS.
    """
    if os.path.exists(FONT_PATH):
        print(f"✅ Font already exists: {FONT_PATH}\n")
        return
    
    print(f"📥 Downloading Noto Color Emoji font...")
    print(f"   URL: {FONT_URL}")
    print(f"   Size: ~10 MB (this may take a moment)\n")
    
    try:
        # Try using curl (common on Linux/Mac)
        subprocess.check_call(
            ["curl", "-L", "-o", FONT_PATH, FONT_URL],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✅ Font downloaded successfully: {FONT_PATH}\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to urllib (pure Python, works everywhere)
        try:
            print("   (Using Python urllib as fallback)")
            import urllib.request
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
            print(f"✅ Font downloaded successfully: {FONT_PATH}\n")
        except Exception as e:
            print(f"❌ Font download failed: {e}")
            print(f"\nPlease download manually from:")
            print(f"   {FONT_URL}")
            print(f"And save it as: {FONT_PATH}")
            sys.exit(1)


# =============================================================================
# STEP 1: STRIKE DISCOVERY
# =============================================================================

def discover_native_resolution(font_path):
    """
    Find the native bitmap resolution of the emoji font.
    
    Why this matters:
    - Emoji fonts contain pre-rendered bitmaps called "strikes"
    - Using the native resolution avoids blurry interpolation
    - Common sizes: 109px (Noto), 128px (Apple)
    
    Returns:
        tuple: (font_object, native_size)
    """
    print("🔍 Discovering native font resolution...")
    
    # Try common strike sizes
    for strike_size in [109, 128, 136]:
        try:
            font = ImageFont.truetype(font_path, strike_size)
            print(f"   ✓ Found native strike: {strike_size}px")
            return font, strike_size
        except Exception:
            continue
    
    # Fallback to default
    print("   ⚠ Using fallback size: 128px")
    return ImageFont.truetype(font_path, 128), 128


# =============================================================================
# STEP 2: ALPHA-BASED CROPPING
# =============================================================================

def render_and_crop_emoji(emoji_char, font, strike_size):
    """
    Render emoji and crop to its visual bounds using alpha channel.
    
    Why alpha-based cropping:
    - Font metadata (bounding box) is often asymmetric
    - We only care about visible pixels, not the "container"
    - This eliminates positional noise in the training data
    
    Args:
        emoji_char: Single emoji character
        font: PIL Font object
        strike_size: Native resolution of the font
    
    Returns:
        PIL.Image: Cropped RGBA image
    """
    # Render at 2x size for better quality (oversampling)
    temp_size = strike_size * 2
    canvas = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Draw emoji at center
    draw.text(
        (temp_size // 2, temp_size // 2),
        emoji_char,
        font=font,
        anchor="mm",  # Middle-middle alignment
        embedded_color=True  # Enable color emoji rendering
    )
    
    # Find the bounding box of non-transparent pixels
    bbox = canvas.getbbox()
    
    if bbox is None:
        # Emoji failed to render - return empty image
        print(f"   ⚠ Failed to render: {emoji_char}")
        return Image.new('RGBA', (strike_size, strike_size), (0, 0, 0, 0))
    
    # Crop to visual bounds only
    cropped = canvas.crop(bbox)
    return cropped


# =============================================================================
# STEP 3: SYMMETRIC NORMALIZATION
# =============================================================================

def normalize_to_square(emoji_image, target_size, padding_percent=0.1):
    """
    Center the emoji on a square canvas with symmetric padding.
    
    Why symmetric normalization:
    - Ensures all emojis occupy the same "conceptual space"
    - Prevents the network from learning positional artifacts
    - Creates consistent "breathing room" around each entity
    
    Args:
        emoji_image: Cropped PIL Image (RGBA)
        target_size: Final output size (e.g., 32)
        padding_percent: Percentage of padding (default 10%)
    
    Returns:
        PIL.Image: Normalized square image
    """
    # Calculate padding
    max_dimension = max(emoji_image.size)
    padding = int(max_dimension * padding_percent)
    canvas_size = max_dimension + (padding * 2)
    
    # Create square transparent canvas
    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    
    # Calculate center position
    paste_x = (canvas_size - emoji_image.width) // 2
    paste_y = (canvas_size - emoji_image.height) // 2
    
    # Paste emoji at center
    canvas.paste(emoji_image, (paste_x, paste_y))
    
    # Resize to target size using high-quality Lanczos filter
    normalized = canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return normalized


# =============================================================================
# STEP 4: TENSOR CONVERSION
# =============================================================================

def image_to_tensor(pil_image):
    """
    Convert PIL Image to normalized NumPy tensor.
    
    Why tensors:
    - Neural networks operate on numbers, not pixels
    - Normalizing to [0, 1] range makes learning more stable
    - RGB conversion removes the alpha channel
    
    Args:
        pil_image: PIL Image (RGBA)
    
    Returns:
        np.ndarray: Shape (height, width, 3), values in [0, 1]
    """
    # Convert RGBA to RGB (white background)
    rgb_image = pil_image.convert('RGB')
    
    # Convert to NumPy array and normalize
    tensor = np.array(rgb_image) / 255.0
    
    return tensor


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def extract_emoji_dna(emoji_list, target_size=32):
    """
    Complete DNA extraction pipeline.
    
    This is the "Feature Engineering" step that transforms raw font data
    into clean, normalized training tensors.
    
    Args:
        emoji_list: List of emoji characters
        target_size: Output image size
    
    Returns:
        np.ndarray: Shape (num_emojis, height, width, 3)
    """
    # Step 1: Discover native resolution
    font, strike_size = discover_native_resolution(FONT_PATH)
    
    print(f"\n🧬 Extracting DNA from {len(emoji_list)} emojis...")
    
    dna_tensors = []
    
    for i, emoji in enumerate(emoji_list):
        # Step 2: Render and crop
        cropped = render_and_crop_emoji(emoji, font, strike_size)
        
        # Skip failed renders
        if cropped.size[0] < 10:
            print(f"   ⚠ Skipping {emoji} (too small)")
            continue
        
        # Step 3: Normalize
        normalized = normalize_to_square(cropped, target_size)
        
        # Step 4: Convert to tensor
        tensor = image_to_tensor(normalized)
        
        dna_tensors.append(tensor)
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"   ⚡ Processed {i + 1}/{len(emoji_list)}...")
    
    print(f"   ✅ Extraction complete: {len(dna_tensors)} tensors\n")
    
    # Stack into single array
    return np.array(dna_tensors)


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_dna_library(dna_array, labels):
    """
    Display the extracted DNA library for verification.
    
    This visual check ensures:
    - All emojis are centered
    - Padding is consistent
    - Colors are preserved
    - No rendering artifacts
    """
    num_samples = len(dna_array)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 3))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('Extracted DNA Library (Genetic Landmarks)', 
                 color='#00FFCC', fontsize=14, y=0.98)
    
    for i in range(num_samples):
        axes[i].imshow(dna_array[i])
        axes[i].set_title(labels[i], color='#00FFCC', fontsize=9, pad=8)
        axes[i].axis('off')
        
        # Add grid lines to verify centering
        axes[i].axvline(IMAGE_SIZE // 2, color='#333', linewidth=0.5, alpha=0.3)
        axes[i].axhline(IMAGE_SIZE // 2, color='#333', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tutorial_01_output.png', dpi=150, facecolor='#0a0a0a')
    print("💾 Saved visualization: tutorial_01_output.png")
    plt.show()


# =============================================================================
# EDUCATIONAL SUMMARY
# =============================================================================

def print_summary(dna_array):
    """Print key statistics about the extracted data."""
    print("=" * 70)
    print("📊 DNA EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Number of landmarks:    {len(dna_array)}")
    print(f"Tensor shape:           {dna_array.shape}")
    print(f"Data type:              {dna_array.dtype}")
    print(f"Value range:            [{dna_array.min():.3f}, {dna_array.max():.3f}]")
    print(f"Memory footprint:       {dna_array.nbytes / 1024:.2f} KB")
    print(f"Per-emoji compression:  {IMAGE_SIZE}x{IMAGE_SIZE}x3 = {IMAGE_SIZE**2 * 3} values")
    print("=" * 70)
    print("\n✅ Ground truth established! Ready for neural training.\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TUTORIAL 01: DNA EXTRACTION")
    print("=" * 70 + "\n")
    
    # Download font if needed
    download_font_if_needed()
    
    # Extract DNA
    emoji_dna = extract_emoji_dna(EMOJI_LANDMARKS, target_size=IMAGE_SIZE)
    
    # Display statistics
    print_summary(emoji_dna)
    
    # Visualize results
    visualize_dna_library(emoji_dna, EMOJI_LABELS)
    
    # Save for next tutorial
    np.save('tutorial_01_dna.npy', emoji_dna)
    print("💾 Saved DNA tensors: tutorial_01_dna.npy")
    print("\n🎓 Next: tutorial_02_linear_decoder.py\n")
