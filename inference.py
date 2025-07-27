# inference_clean_v2.py  ──────────────────────────────────────────────
# Stand‑alone script: smooth heat‑map, no hot‑stripe, low‑count visible
# --------------------------------------------------------------------
import cv2, torch, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from model import MSDCNet  
from torchvision import transforms
from PIL import Image
import scipy.io as sio

# ─── USER SETTINGS ──────────────────────────────────────────
CHECKPOINT_PATH = "fine_tune_last.pth"
IMAGE_FOLDER  = "ShanghaiTechA\\part_B\\test_Data\\images"
GT_FOLDER = "ShanghaiTechA\\part_B\\test_Data\\ground_truth"
OUTPUT_DIR    = "testB"
HEATMAP_ALPHA      = 0.4            # Transparency of heat map 1-opauque and 0-invisible
EMPTY_THRESHOLD  = 0.2               # <0.25 ⇒ skip heat overlay
GAMMA_CORRECTION      = 0.65            # <1 amplifies low activations
# ────────────────────────────────────────────────────────────
def normalize_heatmap(density_map: np.ndarray) -> np.ndarray:
    """Blurs, clips, and gamma-corrects the density map for clear visualization."""
    lo_pct = 5
    hi_pct = 75
    knee = 0.65
    highlight_gamma = 2.0
    blurred = cv2.GaussianBlur(density_map, (0, 0), sigmaX=5)

    # 2) percentile stretch
    lo = np.percentile(blurred, lo_pct)
    hi = np.percentile(blurred, hi_pct)
    if hi - lo < 1e-6:                 # avoid /0
        return np.zeros_like(blurred)

    stretched = np.clip(blurred, lo, hi)
    stretched = (stretched - lo) / (hi - lo)      # → [0,1]

    # 3) soft‑knee highlight compression
    below = stretched <= knee
    above = ~below
    compressed = np.empty_like(stretched)
    compressed[below] = stretched[below]          # linear mid‑tones
    # gamma remap for highlights
    compressed[above] = knee + (stretched[above] - knee) ** highlight_gamma / \
                        (1 - knee) ** (highlight_gamma - 1)
    return compressed

def create_overlay(image_np: np.ndarray, density_map: np.ndarray, gt_count: int, pred_count: float) -> plt.Figure:
    """Creates a matplotlib figure with a heatmap overlay and predicted count."""
    final_image = image_np.copy()
    if pred_count >= EMPTY_THRESHOLD:
        norm_map = normalize_heatmap(density_map)
        cmap = plt.get_cmap('jet')
        heatmap_rgb = cmap(norm_map)[:, :, :3]
        alpha_mask = (norm_map * HEATMAP_ALPHA).clip(0, 1)
        final_image = (final_image * (1 - alpha_mask[..., None]) +
                       heatmap_rgb * alpha_mask[..., None])

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(np.clip(final_image, 0, 1))
    ax.axis('off')
    ax.text(15, 35, f"Actual Count: {gt_count}\nPredicted Count: {pred_count:.0f}",
            color='white', weight='bold', fontsize=16,
            bbox=dict(facecolor='black', alpha=0.7, pad=8, edgecolor='none'))    
    return fig

# --------------------------------------------------------------------------
#  MAIN INFERENCE FUNCTION
# --------------------------------------------------------------------------
@torch.inference_mode()
def run_inference():
    """Loads a model and runs inference on all images in a folder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    # --- Load Model ---
    model = MSDCNet()
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        return
    model.load_state_dict(ckpt, strict=False)
    model.use_mfe = any(k.startswith("mfenet.") for k in ckpt)
    model.to(device).eval()
    print(f"Model '{CHECKPOINT_PATH}' loaded successfully.")

    # --- Define Image Transforms ---
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Process each image file ---
    image_paths = list(Path(IMAGE_FOLDER).glob('*[.png,.jpg,.jpeg]'))
    if not image_paths:
        print(f"ERROR: No images found in '{IMAGE_FOLDER}'")
        return
        
    print(f"Found {len(image_paths)} images. Starting inference...")
    
    for img_path in image_paths:
        # Load and transform the image
        img_pil = Image.open(img_path).convert('RGB')
        original_image_for_viz = img_pil.copy()
        normalized_img = img_transform(img_pil).to(device)

        gt_count = 0
        # Assumes ground truth file is named 'GT_{image_name}.mat'
        gt_path = Path(GT_FOLDER) / f"GT_{img_path.stem}.mat"
        if gt_path.exists():
            mat = sio.loadmat(str(gt_path))
            # This data structure is specific to the ShanghaiTech dataset format
            gt_count = len(mat['image_info'][0, 0][0, 0][0])
        else:
            print(f"  WARNING: Ground truth for {img_path.name} not found at {gt_path}")

        
        # --- Prediction ---
        pred_density = model(normalized_img.unsqueeze(0)).squeeze().cpu().numpy()
        pred_count = pred_density.sum().item()
        print(f"  > Processing {img_path.name}: Predicted Count={pred_count:.2f} Actual Count = {gt_count}")

        # --- Generate and Save Overlay Image ---
        original_image_np = np.array(original_image_for_viz) / 255.0
        pred_density_resized = cv2.resize(pred_density, (original_image_np.shape[1], original_image_np.shape[0]))
        fig = create_overlay(original_image_np, pred_density_resized, gt_count, pred_count)
        
        output_path = Path(OUTPUT_DIR) / f"{img_path.stem}_prediction.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"\nInference complete! All results saved to the '{OUTPUT_DIR}' folder.")

# --------------------------------------------------------------------------
#  RUN SCRIPT
# --------------------------------------------------------------------------
if __name__ == '__main__':
    run_inference()
