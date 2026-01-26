#!/usr/bin/env python3
"""
Neural Emoji Lab - Training Script
Generates the latent space and trains Ridge Regression models for emoji synthesis.
"""

import base64
import io
import json
import sys
import subprocess
import os
import numpy as np
import time
import unicodedata
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# --- CONFIGURATION ---
IMG_SIZE = 64
LATENT_DIM = 512
LAMBDA = 5.0
DECIMALS = 4

FONT_FILENAME = "NotoColorEmoji.ttf"
FONT_URL = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"

OUTPUT_DIR = "public/data"

# --- TOP POPULAR EMOJIS ---
POPULAR_EMOJIS = """
😂❤️🤣👍😭🙏😘🥰😍✨🔥😊💀🤔🎉😁🥺😅✔️♥️💪✨
⚡👀🙌😎✌️😉🌹👏💖🦋😢💋🌹🎈💙👋🤦😆🤝😁💕🤔
😡🥳😠🔫🍀🤗💜☠️😔✨😏☀️😩👋💐💓🍵🌞🤭😤👁️
💃👉⌚👊🍃🤤🤪🐔👽🌊🍓⭐🚫🤬✝️🤷🌹🥀🙈😈🎶💞
💔☹️💣😴💤🤢🤧🤠🤲😷🤒🤕🤑🤠🥴🥵🥶🥳🤯🥰🥱
🧐💀☠️👽👾🤖🎃😺😸😹😻😼😽🙀😿😾👋🤚🖐️✋🖖
👌🤏✌️🤞🤟🤘🤙👈👉👆👇👍👎👊🤛🤜👏🙌👐🤲🤝
🙏💅🤳💪🦾🦿🦵🦶👂🦻👃🧠🦷🦴👀👁️👅👄💋🩸
💘💝💖💗💓💞💕💟❣️💔❤️🧡💛💚💙💜🤎🖤🤍💯
💢💥💫💦💨🕳️💣💬👁️‍🗨️💤👋🤚🖐️✋🖖👌🤏✌️🤞
🤟🤘🤙👈👉👆👇👍👎👊🤛🤜👏🙌👐🤲🤝🙏💅🤳
💪🦾🦿🦵🦶👂🦻👃🧠🦷🦴👀👁️👅👄💋🩸🤢🤮🤧
🥵🥶🥴🤯🤠🥳😎🤓🧐😕😟🙁☹️😮😯😲😳🥺😦😧
😨😰😥😢😭😱😖😣😞😓😩😫🥱😤😡😠🤬😈👿💀
☠️💩🤡👹👺👻👽👾🤖😺😸😹😻😼😽🙀😿😾🙈🙉
🙊💋💌💘💝💖💗💓💞💕💟❣️💔❤️🧡💛💚💙💜
🤎🖤🤍💯💢💥💫💦💨🕳️💣💬👁️‍🗨️💤👋🤚🖐️✋
"""

POPULAR_LIST = sorted(list(set(POPULAR_EMOJIS.replace("\n", "").replace(" ", ""))))


def log(msg):
    """Log with timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def download_font():
    """Download Noto Color Emoji font if not present"""
    if os.path.exists(FONT_FILENAME):
        log(f"✅ Font already exists: {FONT_FILENAME}")
        return
    
    log(f"⬇️  Downloading Font from {FONT_URL}...")
    try:
        subprocess.check_call(["curl", "-L", "-o", FONT_FILENAME, FONT_URL])
        log("✅ Font downloaded successfully")
    except Exception as e:
        log(f"❌ Font download failed with curl, trying urllib: {e}")
        try:
            import urllib.request
            urllib.request.urlretrieve(FONT_URL, FONT_FILENAME)
            log("✅ Font downloaded successfully")
        except Exception as e2:
            log(f"❌ Font download failed: {e2}")
            raise


def load_font():
    """Load the emoji font with appropriate size"""
    try:
        return ImageFont.truetype(FONT_FILENAME, 109), 109
    except Exception:
        try:
            return ImageFont.truetype(FONT_FILENAME, 128), 128
        except Exception as e:
            log(f"❌ Failed to load font: {e}")
            return None, None


def bleed_colors(img):
    """Create infinite color bleed effect for training"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    r, g, b, a = img.split()
    
    # Create smeared color version
    smear = img.resize((8, 8)).resize(img.size, resample=Image.Resampling.BILINEAR)
    smear = smear.filter(ImageFilter.GaussianBlur(radius=8))
    smear.putalpha(255)
    
    # Composite original over smear
    final = Image.composite(img, smear, a).convert('RGB')
    
    # Enhance color saturation
    return ImageEnhance.Color(final).enhance(1.4)


def extract_texture(img_gray):
    """Extract high-pass texture information"""
    blurred = img_gray.filter(ImageFilter.GaussianBlur(radius=3))
    arr_orig = np.array(img_gray) / 255.0
    arr_blur = np.array(blurred) / 255.0
    return np.clip((arr_orig - arr_blur) + 0.5, 0, 1)


def get_emoji_name(char):
    """Get Unicode name for emoji character"""
    try:
        return unicodedata.name(char).title()
    except Exception:
        return f"U+{ord(char):X}"


def create_dataset():
    """Generate training dataset from emoji list"""
    font, strike_size = load_font()
    if not font:
        log("❌ Font failure - cannot continue")
        return None
    
    d_sil, d_tex, d_col = [], [], []
    ui_b64, ui_lbl = [], []
    collected = 0
    
    log(f"🌟 Processing {len(POPULAR_LIST)} popular emojis...")
    
    for char in POPULAR_LIST:
        # Render at 2x size for better quality
        temp_res = strike_size * 2
        img = Image.new('RGBA', (temp_res, temp_res), (0, 0, 0, 0))
        
        try:
            ImageDraw.Draw(img).text(
                (temp_res // 2, temp_res // 2),
                char,
                font=font,
                anchor="mm",
                embedded_color=True
            )
        except Exception:
            continue
        
        # Crop to content
        bbox = img.getbbox()
        if not bbox:
            continue
        
        img = img.crop(bbox)
        
        # Skip tiny renders
        if img.size[0] < 20:
            continue
        
        # Add padding
        max_s = max(img.size)
        pad = int(max_s * 0.05)
        final_s = max_s + (pad * 2)
        bg = Image.new('RGBA', (final_s, final_s), (0, 0, 0, 0))
        bg.paste(img, ((final_s - img.size[0]) // 2, (final_s - img.size[1]) // 2))
        
        # Resize to target size
        small = bg.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Extract three feature channels
        alpha = np.clip(np.array(small.split()[-1]) / 255.0 * 2.0, 0, 1)
        tex = extract_texture(small.convert('L'))
        col = np.array(bleed_colors(small)) / 255.0
        
        d_sil.append(alpha)
        d_tex.append(tex)
        d_col.append(col)
        
        # Generate base64 preview
        buff = io.BytesIO()
        small.save(buff, format="PNG")
        b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        ui_b64.append(f"data:image/png;base64,{b64}")
        ui_lbl.append(f"{char} {get_emoji_name(char)}")
        
        collected += 1
        if collected % 50 == 0:
            log(f"⚡ {collected} emojis processed...")
    
    log(f"✅ Dataset created: {collected} emojis")
    return np.array(d_sil), np.array(d_tex), np.array(d_col), ui_lbl, ui_b64


def solve_ridge(X, Y, lam=1.0):
    """Solve Ridge Regression using closed-form solution"""
    XTX = X.T @ X
    idxs = np.diag_indices_from(XTX)
    XTX[idxs] += lam
    return np.linalg.inv(XTX) @ X.T @ Y


def train_models(sil, tex, col):
    """Train the three Ridge Regression models"""
    N = len(sil)
    log(f"🧠 Training {N} items with latent dimension {LATENT_DIM}...")
    
    # Generate random latent codes
    np.random.seed(42)
    latents = np.random.randn(N, LATENT_DIM)
    
    # Train three separate models
    log(f"📐 Applying Ridge Regression (Lambda={LAMBDA})...")
    W_sil = solve_ridge(latents, sil.reshape(N, -1) - 0.5, LAMBDA)
    W_tex = solve_ridge(latents, tex.reshape(N, -1) - 0.5, LAMBDA)
    W_col = solve_ridge(latents, col.reshape(N, -1) - 0.5, LAMBDA)
    
    log("✅ Training complete")
    return latents, W_sil, W_tex, W_col


def precision_round(matrix):
    """Round matrix to specified decimal places"""
    return [[round(float(x), DECIMALS) for x in row] for row in matrix]


def export_data(latents, W_sil, W_tex, W_col, labels, b64s):
    """Export training data to JSON files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log(f"📦 Packaging data ({DECIMALS} decimal precision)...")
    
    # Metadata and previews
    meta = {
        "labels": labels,
        "b64": b64s,
        "size": IMG_SIZE,
        "latentDim": LATENT_DIM,
        "lambda": LAMBDA
    }
    
    with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f)
    
    # Latent vectors
    with open(f"{OUTPUT_DIR}/latents.json", "w") as f:
        json.dump(precision_round(latents), f)
    
    # Weight matrices
    with open(f"{OUTPUT_DIR}/weights_sil.json", "w") as f:
        json.dump(precision_round(W_sil), f)
    
    with open(f"{OUTPUT_DIR}/weights_tex.json", "w") as f:
        json.dump(precision_round(W_tex), f)
    
    with open(f"{OUTPUT_DIR}/weights_col.json", "w") as f:
        json.dump(precision_round(W_col), f)
    
    log(f"✅ Data exported to {OUTPUT_DIR}/")


def main():
    """Main training pipeline"""
    log("🧬 Neural Emoji Lab - Training Pipeline")
    log("=" * 50)
    
    # Download font if needed
    download_font()
    
    # Create dataset
    result = create_dataset()
    if result is None:
        log("❌ Dataset creation failed")
        return 1
    
    sil, tex, col, labels, b64s = result
    
    # Train models
    latents, W_sil, W_tex, W_col = train_models(sil, tex, col)
    
    # Export data
    export_data(latents, W_sil, W_tex, W_col, labels, b64s)
    
    log("=" * 50)
    log("🚀 Training complete! Ready to serve with http-server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
