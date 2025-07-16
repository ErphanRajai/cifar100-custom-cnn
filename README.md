# cifar100-custom-cnn
A deep convolutional neural network built **from scratch** in PyTorch, trained on the CIFAR-100 dataset with no pretrained models.


# CIFAR-100 Custom CNN

> 🧠 A deep convolutional neural network built **from scratch** in PyTorch, trained on the CIFAR-100 dataset with no pretrained models.

## 🚀 Highlights
- 🔨 Custom-designed CNN architecture (no ResNet, no transfer learning)
- 🎯 Achieves **~90% training accuracy** and **~70% validation accuracy**
- 🧪 Label smoothing, OneCycleLR scheduler, and Dropout2d regularization
- 📈 Clean modular training loop with model checkpointing and metric plotting
- 🔍 Fully reproducible and easy to adapt to other image classification tasks

## 🧱 Model Architecture
- 4 × `ConvBlock` (each with Conv2d → BatchNorm → ReLU → Dropout2d)
- Dense classifier with Dropout and BatchNorm

## 📊 Final Metrics
| Metric         | Value      |
|----------------|------------|
| Train Accuracy | 90.4%      |
| Val Accuracy   | 70.4%      |
| Val Loss       | ~1.78      |
| Best Epoch     | 93         |

## 📁 Project Structure
├── engine.py # Training utilities (dataloaders, loop, plotting)
├── best_model.pth # best saved model
├── CIFAR100_Custom_CNN.ipynb # Training results, curves, and experimentation
├── requirements.txt
└── README.md

## 📦 Requirements

```bash
torch>=2.0
torchvision>=0.15
matplotlib
tqdm
```

install with:
pip install -r requirements.txt

# Clone the repo
git clone https://github.com/yourusername/cifar100-custom-cnn.git
cd cifar100-custom-cnn

# Run training
python train.py
