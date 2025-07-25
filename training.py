import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import MSDCNet
from torchvision.models import vgg16
from torch.optim.lr_scheduler import StepLR

# 1. Dataset to load .pt files
class PtCrowdDataset(Dataset):
    """
    Simple loader for the *_orig.pt, *_hflip.pt, *_vflip.pt, *_rnd*.pt files
    produced by our new preprocessing pipeline.

    Each .pt file stores:
        {"img": Tensor[C,H,W]  float32 0‑1,
         "den": Tensor[1,H,W]  float32}
    """
    def __init__(self, pt_dir, transform=None):
        self.files = sorted(Path(pt_dir).glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found in {pt_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # --- load sample (guaranteed contiguous, positive‑stride tensors) ---
        sample = torch.load(self.files[idx], map_location="cpu")
        img = sample["img"]          # [C,H,W], float32 0‑1
        den = sample["den"]          # [1,H,W]

        # --- safety guard for odd legacy files -----------------------------
        if img.dim() == 2:
            img = img.unsqueeze(0)           # 1×H×W
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)        # grey → RGB

        if den.dim() == 2:
            den = den.unsqueeze(0)           # 1×H×W

        # --- optional in‑memory transform (e.g., normalisation) ------------
        if self.transform is not None:
            img = self.transform(img)

        return {"img": img, "den": den}
    

def freeze(module, flag=False):
    for p in module.parameters():
        p.requires_grad = flag

def make_optimizer(params, lr, wd=5e-3):
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)

def evaluate(model, loader, device):
    model.eval()
    total_mae = total_mse = n = 0
    for batch in loader:
        img = batch['img'].to(device)
        den = batch['den'].to(device)
        pred = model(img)

        pred_cnt = pred.flatten(1).sum(dim=1)
        true_cnt = den.flatten(1).sum(dim=1)
        err      = pred_cnt - true_cnt
        total_mae += err.abs().sum().item()
        total_mse += (err**2).sum().item()
        n += img.size(0)
    mae  = total_mae / n
    rmse = (total_mse / n) ** 0.5
    return mae, rmse

def configure_stage(model, stage):
    # Stage 1 ▸ back‑end warm‑up
    if stage == 1:
        print("Running Stage 1")
        model.use_mfe = False     # bypass flag implemented inside MSDCNet.forward
        freeze(model.front_end)
        freeze(model.backend, True)
        return make_optimizer(model.backend.parameters(), lr=2e-3, wd=0)

    # Stage 2 ▸ front‑end fine‑tune
    if stage == 2:
        print("Running Stage 2")
        model.use_mfe = False
        freeze(model.backend)   # MAKE TRUE OF THERE IS NO PROGRESS IN TRAINING
        freeze(model.front_end, True)
        '''
        THIS NEEDS TO BE DONE IF BOTH ARE UNFROZEN
        return torch.optim.Adam([
            {'params': model.front_end.parameters(), 'lr': 1e-3},
            {'params': model.backend.parameters(),   'lr': 1e-4},
        ], weight_decay=0)
        '''
        return make_optimizer(model.front_end.parameters(), lr=1e-3, wd=0)
        
    # Stage 3 ▸ MFENet column‑wise
    if stage == 3:
        print("Running Stage 3")
        model.use_mfe = True
        freeze(model.front_end)
        freeze(model.backend)
        # iterate later inside training loop
        return None
    print("Running Stage 4")
    # Stage 4 ▸ global fine‑tune
    freeze(model, True)           # unfreeze all
    return make_optimizer(model.parameters(), lr=8e-5, wd=5e-4)

# 4. Single‑epoch training -----------------------------------------
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs = batch['img'].to(device)
        dens = batch['den'].to(device)
        pred = model(imgs)
        loss = criterion(pred, dens)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()        # sum over batch
    return total_loss / len(loader.dataset)               # average over images

# ---------------- main entry -------------------------------------
def main():
    stage = 1
    train_root = "ShanghaiTechA/part_B/train_data/pt"
    val_root = "ShanghaiTechA/part_B/test_data/pt"
    batch = 8
    out_dir = 'Run1'
    epochs = 100
    workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    # ► load / build model
    model = MSDCNet()
    ## CODE TO FIX THE PLATEAU ISSUE
    vgg = vgg16(weights='IMAGENET1K_V1')
    fe_conv_layers = [m for m in model.front_end if isinstance(m, torch.nn.Conv2d)]
    vgg_conv_layers = [m for m in vgg.features if isinstance(m, torch.nn.Conv2d)]
    for src, dst in zip(vgg_conv_layers[:10], fe_conv_layers[:10]):   # first‑10 convs
        dst.weight.data.copy_(src.weight.data)
        if dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)
            
    model.front_end.load_state_dict(vgg.features[:23].state_dict(), strict=False)
    model.to(device)    
    normalize = lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) \
                      / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # data loaders
    train_loader = DataLoader(PtCrowdDataset(train_root, transform = normalize), batch_size=batch,
                              shuffle=True, num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(PtCrowdDataset(val_root, transform = normalize),   batch_size=batch,
                              shuffle=False, num_workers=workers, pin_memory=True)
    # If we're in stage 2, 3 or 4 load the best ckpt from stage‑1,‑2,‑3
    if stage > 1:
        prev_ckpt = Path(out_dir) / f"stage{stage-1}_best.pth"
        print(f"Loading checkpoint {prev_ckpt} …")
        model.load_state_dict(torch.load(prev_ckpt, map_location=device), strict=False)
    opt = configure_stage(model, stage)
    if opt is not None:
        scheduler = StepLR(opt, step_size=8, gamma=0.0001)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
        
    #criterion = nn.MSELoss(reduction='mean')
    criterion = nn.MSELoss(reduction='sum')
    best_rmse = float('inf'); best_path = None 

    if stage == 3:
        # 1) Freeze everything up‑front
        freeze(model.mfenet)  

        # ——————————————
        # 2) Train only column 1
        # ——————————————
        col1 = model.mfenet.columns[0]
        freeze(col1, True)                       # unfreeze col1 only
        col1_opt = make_optimizer(col1.parameters(), lr=1e-4)
        best_rmse = float('inf')
        patience = 10
        patience_counter = 0
        for ep in range(1, epochs+1):
            tr_loss   = train_epoch(model, train_loader, col1_opt, criterion, device)
            val_mae, val_rmse = evaluate(model, val_loader, device)
            print(f"[Stage 3‑Col1] ep{ep:02d} train loss {tr_loss:.6f} "
                f"val MAE {val_mae:.2f} RMSE {val_rmse:.2f}")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), Path(out_dir)/"stage3_col1_best.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {ep} for column 1")
                    break
        # ——————————————
        # 3) Copy col1 weights → cols 2–4
        # ——————————————
        best1 = Path(out_dir)/"stage3_col1_best.pth"
        state = torch.load(best1, map_location=device)
        model.load_state_dict(state, strict=False)     # load only the layers that exist
        with torch.no_grad():
            # assume first layer is at index 0 in each column
            ref_conv = model.mfenet.columns[0][0]
            for col in model.mfenet.columns[1:]:
                col_conv = col[0]
                col_conv.weight.copy_(ref_conv.weight)
                col_conv.bias.copy_(ref_conv.bias)

        # ————————————————————————————————————
        # 4) Train columns 2, 3, and 4 one by one
        # ————————————————————————————————————
        for idx, col in enumerate(model.mfenet.columns[1:], start=2):
            prev = Path(out_dir)/f"stage3_col{idx-1}_best.pth"
            prev_state = torch.load(prev, map_location=device)
            model.load_state_dict(prev_state, strict=False)
            freeze(model.mfenet)                # freeze all again
            freeze(col, True)                   # unfreeze only this column
            col_opt = make_optimizer(col.parameters(), lr=1e-4)
            best_rmse = float('inf')
            patience = 10
            patience_counter = 0
            for ep in range(1, epochs+1):
                tr_loss   = train_epoch(model, train_loader, col_opt, criterion, device)
                val_mae, val_rmse = evaluate(model, val_loader, device)
                print(f"[Stage 3‑Col{idx}] ep{ep:02d} train loss {tr_loss:.6f} "
                    f"val MAE {val_mae:.2f} RMSE {val_rmse:.2f}")

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    torch.save(model.state_dict(), Path(out_dir)/f"stage3_col{idx}_best.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {ep} for column {idx}")
                        break
    else:  # stages 1, 2, 4
        patience = 10
        patience_counter = 0
        for ep in range(1, epochs+1):
            tr_loss = train_epoch(model, train_loader, opt, criterion, device)
            val_mae, val_rmse = evaluate(model, val_loader, device)
            print(f"[Stage {stage}] ep{ep:02d} train loss {tr_loss:.6f} "
                  f"val MAE {val_mae:.2f} RMSE {val_rmse:.2f}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_path = Path(out_dir)/f"stage{stage}_best.pth"
                torch.save(model.state_dict(), best_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {ep} for stage {stage}")
                    break
            scheduler.step()

    # save final (last‑epoch) weights too
    final_path = Path(out_dir) / f"stage{stage}_last.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\nStage {stage} finished.  Best RMSE {best_rmse:.2f} "
          f"→ saved to {best_path if best_path else final_path}")

# ---------------- run ---------------------------------------------
if __name__ == "__main__":
    main()

