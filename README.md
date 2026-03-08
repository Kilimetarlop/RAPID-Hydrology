RAPID-Hydrology

This repository contains the **core model architecture and trainer code** for the paper "[RAPID-Hydrology: Pre-training Hydro-Meteorological Representations with Diffusion for Enhanced Causal Prediction]", currently under review at IEEE Transactions on Geoscience and Remote Sensing (TGRS).

## 📦 Current Content (For Peer Review Only)
To protect the intellectual property of this work during the review process, we currently provide the following core components to verify the feasibility of our proposed framework:
- `model.py`: Implementation of the **encoder-evolver-decoder architecture** (the core cross-domain fusion innovation of this paper), corresponding to Section X of the manuscript.
- `trainer.py`: Core training logic and optimization setup, corresponding to Section Y of the manuscript.
- `mini_test/`: A minimal test dataset (a few MB) to verify the forward pass of the model and 1-2 steps of training.
- `requirements.txt`: Dependencies required to run the code.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the minimal test: `python test_mini.py` (or follow the step-by-step instructions in `docs/run_guide.md` if provided)

## 📅 Full Open-Source Plan
Upon **acceptance of this manuscript**, we will immediately release the complete content, including:
- Full training and inference code
- Full training/test datasets (or detailed instructions for data acquisition)
- Pre-trained model weights
- Complete data preprocessing, visualization, and distributed training scripts
- A comprehensive documentation for full reproducibility

## 🙏 Acknowledgment
Thank you for your time and understanding during the review process. If you have any questions about the code, please feel free to contact us via the manuscript's correspondence channel.
