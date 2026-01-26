/**
 * Neural Emoji Lab - Browser Inference Engine
 * Loads pre-trained weights and performs real-time emoji synthesis
 */

// Global state
const APP = {
  meta: null,
  latents: null,
  W_sil: null,
  W_tex: null,
  W_col: null,
  ctx: null,
  imgData: null,
  size: 64,
  pixelCount: 0,
};

// DOM elements
const elements = {
  loading: null,
  app: null,
  emojiCount: null,
  imgA: null,
  imgB: null,
  selectA: null,
  selectB: null,
  sliderSil: null,
  sliderTex: null,
  sliderCol: null,
  canvas: null,
};

/**
 * Initialize the application
 */
async function init() {
  console.log("🧬 Neural Emoji Lab - Initializing...");

  // Get DOM elements
  elements.loading = document.getElementById("loading");
  elements.app = document.getElementById("app");
  elements.emojiCount = document.getElementById("emojiCount");
  elements.imgA = document.getElementById("imgA");
  elements.imgB = document.getElementById("imgB");
  elements.selectA = document.getElementById("selectA");
  elements.selectB = document.getElementById("selectB");
  elements.sliderSil = document.getElementById("sliderSil");
  elements.sliderTex = document.getElementById("sliderTex");
  elements.sliderCol = document.getElementById("sliderCol");
  elements.canvas = document.getElementById("canvas");

  // Setup canvas
  APP.ctx = elements.canvas.getContext("2d", { willReadFrequently: true });

  try {
    // Load all data files
    await loadData();

    // Setup UI
    setupUI();

    // Initial render
    randomize();

    // Show app
    elements.loading.style.display = "none";
    elements.app.style.display = "block";

    console.log("✅ Initialization complete");
  } catch (error) {
    console.error("❌ Initialization failed:", error);
    elements.loading.innerHTML = `
            <div style="color: #ff4444;">
                <div style="font-size: 48px; margin-bottom: 20px;">❌</div>
                <div>Failed to load neural network data</div>
                <div style="font-size: 12px; margin-top: 10px; color: #666;">${error.message}</div>
                <div style="font-size: 12px; margin-top: 10px; color: #666;">Make sure you've run: python train.py</div>
            </div>
        `;
  }
}

/**
 * Load all JSON data files
 */
async function loadData() {
  console.log("📦 Loading training data...");

  const baseURL = "data/";

  // Load metadata
  const metaResponse = await fetch(baseURL + "meta.json");
  if (!metaResponse.ok) throw new Error("Failed to load meta.json");
  APP.meta = await metaResponse.json();

  // Load latents
  const latentsResponse = await fetch(baseURL + "latents.json");
  if (!latentsResponse.ok) throw new Error("Failed to load latents.json");
  APP.latents = await latentsResponse.json();

  // Load weight matrices
  const [silResponse, texResponse, colResponse] = await Promise.all([
    fetch(baseURL + "weights_sil.json"),
    fetch(baseURL + "weights_tex.json"),
    fetch(baseURL + "weights_col.json"),
  ]);

  if (!silResponse.ok || !texResponse.ok || !colResponse.ok) {
    throw new Error("Failed to load weight matrices");
  }

  APP.W_sil = await silResponse.json();
  APP.W_tex = await texResponse.json();
  APP.W_col = await colResponse.json();

  APP.size = APP.meta.size;
  APP.pixelCount = APP.size * APP.size;
  APP.imgData = APP.ctx.createImageData(APP.size, APP.size);

  console.log(`✅ Loaded ${APP.meta.labels.length} emojis`);
  console.log(`   Latent dimension: ${APP.latents[0].length}`);
  console.log(`   Image size: ${APP.size}x${APP.size}`);
}

/**
 * Setup UI elements and event listeners
 */
function setupUI() {
  // Populate selects
  const fragment = document.createDocumentFragment();
  APP.meta.labels.forEach((label, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = label;
    fragment.appendChild(option);
  });

  elements.selectA.appendChild(fragment.cloneNode(true));
  elements.selectB.appendChild(fragment);

  // Update emoji count
  elements.emojiCount.textContent = `(${APP.meta.labels.length} emojis)`;

  // Event listeners
  elements.selectA.addEventListener("change", render);
  elements.selectB.addEventListener("change", render);
  elements.sliderSil.addEventListener("input", render);
  elements.sliderTex.addEventListener("input", render);
  elements.sliderCol.addEventListener("input", render);
}

/**
 * Matrix multiplication: vec @ W
 * Returns the result of multiplying a vector by a weight matrix
 */
function predict(vec, W) {
  const outDim = W[0].length;
  const latDim = W.length;
  const out = new Float32Array(outDim).fill(0);

  // Optimized matrix multiplication
  for (let r = 0; r < latDim; r++) {
    const v = vec[r];
    if (Math.abs(v) < 0.01) continue; // Skip near-zero values

    const row = W[r];
    for (let c = 0; c < outDim; c++) {
      out[c] += v * row[c];
    }
  }

  return out;
}

/**
 * Linear interpolation between two values
 */
function lerp(a, b, t) {
  return (1 - t) * a + t * b;
}

/**
 * Render the composite emoji
 */
function render() {
  const indexA = parseInt(elements.selectA.value);
  const indexB = parseInt(elements.selectB.value);

  // Update preview images
  elements.imgA.src = APP.meta.b64[indexA];
  elements.imgB.src = APP.meta.b64[indexB];

  // Get interpolation weights
  const tSil = elements.sliderSil.value / 100;
  const tTex = elements.sliderTex.value / 100;
  const tCol = elements.sliderCol.value / 100;

  // Interpolate latent vectors
  const latentDim = APP.latents[0].length;
  const zSil = new Float32Array(latentDim);
  const zTex = new Float32Array(latentDim);
  const zCol = new Float32Array(latentDim);

  const latA = APP.latents[indexA];
  const latB = APP.latents[indexB];

  for (let i = 0; i < latentDim; i++) {
    zSil[i] = lerp(latA[i], latB[i], tSil);
    zTex[i] = lerp(latA[i], latB[i], tTex);
    zCol[i] = lerp(latA[i], latB[i], tCol);
  }

  // Run inference
  const outSil = predict(zSil, APP.W_sil);
  const outTex = predict(zTex, APP.W_tex);
  const outCol = predict(zCol, APP.W_col);

  // Composite the image
  const data = APP.imgData.data;
  const bias = 0.5;

  for (let i = 0; i < APP.pixelCount; i++) {
    // Silhouette (alpha channel)
    let a = Math.max(0, Math.min(1, outSil[i] + bias));
    // Enhance contrast
    a = (a - 0.2) * 1.5;
    a = Math.max(0, Math.min(1, a));

    // Texture (lighting)
    let t = Math.max(0, Math.min(1, (outTex[i] + bias - 0.5) * 2.0 + 0.5));

    // Color (RGB)
    let r = outCol[i * 3] + bias;
    let g = outCol[i * 3 + 1] + bias;
    let b = outCol[i * 3 + 2] + bias;

    // Apply texture as lighting
    r *= t * 2.0;
    g *= t * 2.0;
    b *= t * 2.0;

    // Write to image data
    data[i * 4] = Math.min(255, r * 255);
    data[i * 4 + 1] = Math.min(255, g * 255);
    data[i * 4 + 2] = Math.min(255, b * 255);
    data[i * 4 + 3] = a * 255;
  }

  APP.ctx.putImageData(APP.imgData, 0, 0);
}

/**
 * Randomize emoji selection
 */
function randomize() {
  const count = APP.meta.labels.length;
  elements.selectA.value = Math.floor(Math.random() * count);
  elements.selectB.value = Math.floor(Math.random() * count);
  render();
}

/**
 * Save the current composite as PNG
 */
function saveImage() {
  const getName = (selectElement) => {
    const text = selectElement.options[selectElement.selectedIndex].text;
    // Extract name after emoji character
    return text
      .substring(text.indexOf(" ") + 1)
      .replace(/\s+/g, "_")
      .replace(/[^a-zA-Z0-9_]/g, "");
  };

  const nameA = getName(elements.selectA);
  const nameB = getName(elements.selectB);
  const s = elements.sliderSil.value;
  const t = elements.sliderTex.value;
  const c = elements.sliderCol.value;

  const filename = `${nameA}_${nameB}_S${s}_T${t}_C${c}.png`;

  const link = document.createElement("a");
  link.download = filename;
  link.href = elements.canvas.toDataURL("image/png");
  link.click();

  console.log(`💾 Saved: ${filename}`);
}

// Start the app when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

// Expose functions to global scope for HTML onclick handlers
window.randomize = randomize;
window.saveImage = saveImage;
