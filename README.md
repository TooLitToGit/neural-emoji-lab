# Neural Emoji Lab

> **Baked by Python. Served by JavaScript.**

An educational experiment that uses Python to pre-calculate the "Latent Space" of emojis, allowing your browser to splice their **Shape**, **Texture**, and **Color** in real-time.

## 🌐 Try It Live

**👉 [Launch Neural Emoji Lab](https://toolittogit.github.io/neural-emoji-lab/) 👈**

### What Does It Do?

Neural Emoji Lab lets you **remix emojis by blending their features independently**. Select two parent emojis and use three sliders to control:

- **🎭 Silhouette** - The shape and boundary (Cookie cutter)
- **✨ Texture** - Internal details and lighting (Relief map)
- **🎨 Color** - RGB color palette (Infinite bleed)

Want the body of a 👻 ghost but the texture and color of 🔥 fire? Or perhaps a 💎 diamond shape with the ghostly colors of 👻? You can create that! Each slider independently blends between your two parent emojis. The app demonstrates **disentangled representation learning** - a core concept in modern AI systems like Stable Diffusion and VAEs.

Save your creations, hit random for inspiration, and explore the mathematical magic of latent spaces in your browser.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the Python training script to generate the neural network data:

```bash
python train.py
```

This will:

- Download the Noto Color Emoji font
- Process ~200+ popular emojis
- Train three Ridge Regression models (Silhouette, Texture, Color)
- Export training data to `public/data/` directory

### 3. Serve the Web App

Use Node's http-server (or any static file server):

```bash
npx http-server public -p 3000
```

Or with Python:

```bash
python -m http.server 3000 --directory public
```

### 4. Open in Browser

Navigate to: http://localhost:3000

## 📐 Architecture

This project uses a **"Compute-Once, Run-Anywhere"** architecture:

```
┌─────────────────┐
│  Python         │  Train three Ridge Regression models
│  train.py       │  • Silhouette (alpha channel)
│                 │  • Texture (high-pass filter)
│                 │  • Color (infinite bleed)
└────────┬────────┘
         │ Exports JSON
         ▼
┌─────────────────┐
│  public/data/   │  Pre-computed training data
│  • meta.json    │  • Emoji metadata & previews
│  • latents.json │  • Random latent codes
│  • weights_*.json  Learned weight matrices
└────────┬────────┘
         │ Loads in browser
         ▼
┌─────────────────┐
│  JavaScript     │  Real-time inference
│  app.js         │  • Matrix multiplication
│                 │  • Feature interpolation
│                 │  • Canvas rendering
└─────────────────┘
```

## 🧠 How It Works

### Three Independent Feature Channels

1. **Silhouette** - The shape/boundary (from alpha channel)
2. **Texture** - Internal details/lighting (high-pass filtered)
3. **Color** - RGB values with infinite bleed effect

### Training

Uses Ridge Regression (closed-form solution) instead of gradient descent:

```
W = (X^T X + λI)^(-1) X^T Y
```

Where:

- `X` = Random latent codes (N × 512)
- `Y` = Extracted features (N × 4096)
- `λ` = Regularization parameter (5.0)

### Inference

In the browser, we:

1. Interpolate between two latent vectors
2. Multiply by weight matrices (`z @ W`)
3. Composite the three channels into final image

## 🎨 Usage

1. Select two parent emojis (A and B)
2. Adjust three sliders:
   - **Silhouette**: Morph the shape between A and B
   - **Texture**: Blend internal details
   - **Color**: Mix color palettes
3. Click **Save** to download your creation
4. Click **Random** to discover new combinations

## 📁 Project Structure

```
neural-emoji-lab/
├── train.py              # Python training script
├── requirements.txt      # Python dependencies
├── public/
│   ├── index.html       # Web interface
│   ├── app.js           # Browser inference engine
│   └── data/            # Generated training data (created by train.py)
│       ├── meta.json
│       ├── latents.json
│       ├── weights_sil.json
│       ├── weights_tex.json
│       └── weights_col.json
└── README.md
```

## 🔬 Technical Details

- **Image Size**: 64×64 pixels
- **Latent Dimension**: 512
- **Regularization**: λ = 5.0
- **Precision**: 4 decimal places
- **Dataset**: ~200+ popular emojis from Unicode

## 💡 Key Concepts

### Disentangled Representation

By training three separate models, we force the network to learn independent features. This allows surgical control—you can take the shape of a 👻 ghost, the texture of a 💎 diamond, and the color of 🔥 fire.

### Ridge Regression

Instead of backpropagation, we use a closed-form solution that's instant and deterministic. Perfect for educational demonstrations.

### Infinite Color Bleed

We pre-process training images to "smear" colors into empty space, ensuring color information exists everywhere. This prevents black artifacts when morphing shapes.

## 🛠️ Development

### Modify Emoji List

Edit the `POPULAR_EMOJIS` string in [train.py](train.py) to include your own emoji selection.

### Adjust Parameters

In [train.py](train.py):

- `IMG_SIZE`: Resolution (default: 64)
- `LATENT_DIM`: Latent space dimensions (default: 512)
- `LAMBDA`: Ridge regression regularization (default: 5.0)
- `DECIMALS`: JSON precision (default: 4)

### Re-train

After changes, re-run:

```bash
python train.py
```

## 📝 License

MIT

## 🙏 Credits

- Font: [Noto Color Emoji](https://github.com/googlefonts/noto-emoji) by Google
- Inspired by modern generative AI concepts (VAEs, Stable Diffusion)
