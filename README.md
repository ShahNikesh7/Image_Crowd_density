# MSDCNet Crowd-Counting

A minimal, production-ready PyTorch implementation of **MSDCNet** for dense crowd counting.

## Repository Structure

.
├── model.py                # MSDCNet architecture  
├── pre_process.py          # Convert ShanghaiTech dataset to .pt tensors  
├── inference.py            # Generate heat-map overlays and count predictions  
├── requirements.txt        # Python dependencies  
├── stage2_best.pth         # Trained model 1 (slight better in certain scenarios)  
└── stage4_best.pth         # Trained model 2  

## 1. Installation

### Quick Start

There is a requirements.txt that you can install through pip.
    pip install -r requirements.txt

### Running Inference
<ol>
<li>Configure paths in inference.py</li>
<li>Run inference.py</li>
</ol>

### Notes

<ul>
<li>Create a images folder and link it to IMAGE_FOLDER in inference.py</li>
<li>Create a output directory and link it to OUTPUT_DIR</li>
</ul>