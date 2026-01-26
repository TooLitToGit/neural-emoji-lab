# 🚀 Quick Start Guide

Follow these steps to get the Neural Emoji Lab up and running:

## Prerequisites

This project includes a **Dev Container** configuration that's ready to go! If you're using VS Code with the Dev Containers extension, the environment is already set up with:

- Python 3.11
- Node.js 20
- All necessary system dependencies

Just open the project in VS Code and reopen in the container when prompted.

## Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- Pillow (image processing)
- NumPy (numerical computation)

## Step 2: Train the Model

Run the training script to generate the neural network data:

```bash
python train.py
```

This will:

- Download the Noto Color Emoji font (~10MB)
- Process ~200+ popular emojis
- Train three Ridge Regression models
- Export training data to `public/data/` directory

Expected output:

```
[HH:MM:SS] 🧬 Neural Emoji Lab - Training Pipeline
[HH:MM:SS] ✅ Font already exists: NotoColorEmoji.ttf
[HH:MM:SS] 🌟 Processing 200+ popular emojis...
[HH:MM:SS] ✅ Dataset created: XXX emojis
[HH:MM:SS] 🧠 Training XXX items with latent dimension 512...
[HH:MM:SS] 📐 Applying Ridge Regression (Lambda=5.0)...
[HH:MM:SS] ✅ Training complete
[HH:MM:SS] 📦 Packaging data (4 decimal precision)...
[HH:MM:SS] ✅ Data exported to public/data/
[HH:MM:SS] 🚀 Training complete! Ready to serve with http-server
```

## Step 3: Serve the Web Application

Use Node's http-server to serve the static files:

```bash
npx http-server public -p 3000
```

Or with Python's built-in server:

```bash
python -m http.server 3000 --directory public
```

## Step 4: Open in Browser

Navigate to: **http://localhost:3000**

You should see the Neural Emoji Lab interface with:

- Two emoji selectors (Parent A and B)
- A canvas showing the neural composite
- Three sliders to control Silhouette, Texture, and Color

## 🎨 Usage Tips

1. **Select Parents**: Choose two different emojis from the dropdown menus
2. **Adjust Sliders**:
   - **Silhouette**: Controls the shape (A ← → B)
   - **Texture**: Controls internal details and lighting (A ← → B)
   - **Color**: Controls the color palette (A ← → B)
3. **Random**: Click the 🎲 button to randomly select emojis
4. **Save**: Click the 💾 button to download your creation as PNG

## 🔧 Troubleshooting

### Training fails with font error

- Make sure you have internet connection for the initial font download
- The font will be saved as `NotoColorEmoji.ttf` in the project root

### Web app shows "Failed to load neural network data"

- Make sure you've run `python train.py` first
- Check that `public/data/` contains JSON files

### Port 3000 already in use

- Use a different port: `npx http-server public -p 8080`
- Or stop the process using port 3000

## 📚 Next Steps

- Read [README.md](README.md) for detailed technical information
- Modify the emoji list in `train.py` to include your favorites
- Experiment with different slider combinations
- Share your creations!
