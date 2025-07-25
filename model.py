import torch
import torch.nn as nn
import torch.nn.functional as F

class SPP2(nn.Module):
    """
    Spatial Pyramid Pooling module replacing each 2×2 max-pool.
    Stacks three max-pools (kernels 2×2/2×2/3×3, strides 2/1/1), crops to the first output’s size,
    and sums them for scale‑invariant feature fusion.
    """
    def __init__(self):
        super(SPP2, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p2 = F.adaptive_max_pool2d(p2, output_size=(p1.size(2), p1.size(3)))
        p3 = self.pool3(x)
        p3 = F.adaptive_max_pool2d(p3, output_size=(p1.size(2), p1.size(3)))
        return p1 + p2 + p3


class MFENet(nn.Module):
    """
    Multi-Scale Feature Extraction Network:
    Four parallel ‘bottleneck’ columns with receptive fields of 1×1, 3×3, 5×5, 7×7
    implemented via stacks of 3×3 convs. Each column reduces channels 512→128.
    Outputs are concatenated back to 512 channels.
    """
    def __init__(self, in_channels=512):
        super(MFENet, self).__init__()
        # Column 1: 1×1 conv
        self.col1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Column 2: 1×1 → 3×3
        self.col2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Column 3: 1×1 → 3×3 → 3×3
        self.col3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Column 4: 1×1 → 3×3 → 3×3 → 3×3
        self.col4 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.columns = nn.ModuleList([self.col1, self.col2, self.col3, self.col4])

    def forward(self, x):
        c1 = self.col1(x)
        c2 = self.col2(x)
        c3 = self.col3(x)
        c4 = self.col4(x)
        # Concatenate outputs → back to 512 channels
        return torch.cat([c1, c2, c3, c4], dim=1)


class MSDCNet(nn.Module):
    """
    MSDCNet: Front-end (truncated VGG16 + SPP2), MFENet, and back-end (dilated convs + up-sampling).
    Produces a full-size density map (same H×W as input).
    """
    def __init__(self):
        super(MSDCNet, self).__init__()

        # --- Front-end: first 10 conv layers of VGG16 with SPP2 instead of max-pool ---
        self.front_end = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            SPP2(),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            SPP2(),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            SPP2(),
            # Block 4 (no pooling)
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        # --- Core MFENet ---
        self.mfenet = MFENet(in_channels=512)

        # --- Back-end: 4 dilated 3×3 convs → 1×1 conv ---
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, dilation=1, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
            #nn.ReLU()
        )

    def forward(self, x):
        # x: (B,3,H,W)
        feat = self.front_end(x)        # → (B,512,H/8,W/8)
        if hasattr(self, 'use_mfe') and not self.use_mfe:
            mfeat = feat
        else:
            mfeat = self.mfenet(feat)
        dens = self.backend(mfeat)       # → (B,1,H/8,W/8)
        torch.clamp(dens, min=0.0)
        # Upsample back to input resolution
        dens = F.interpolate(dens, scale_factor=8, mode='bilinear', align_corners=True)
        return dens


if __name__ == "__main__":
    # Sanity check
    model = MSDCNet()
    inp = torch.randn(1, 3, 512, 512)
    out = model(inp)
    print("Input:", inp.shape, "Output:", out.shape)  # Expect [1,1,512,512]
