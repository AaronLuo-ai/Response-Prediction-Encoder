# Response Prediction Encoder

Predicts **tumor treatment response** of rectal cancer from MRI by reusing the encoder of a U-Net trained for tumor segmentation as a frozen feature extractor.

## How it works

1. **Encoder:** take the ResNet-34 encoder of a segmentation U-Net (`segmentation_models_pytorch`) and freeze it.
2. **Features:** split each MRI volume into 2D slices, resize them to $128 \times 128$, and global-average-pool the encoder output into a 512-d feature vector per slice.
3. **Classifier:** an MLP ($512 \to 256 \to 128 \to 64 \to 32 \to 1$, with BatchNorm, ReLU and dropout) trained with BCE loss to predict **complete response vs. non-response**. Partial responses are grouped with complete responses.
4. Training uses PyTorch Lightning, with early stopping on validation loss and Weights & Biases logging.

## Structure

```
main.py                    # train the response classifier
utils/dataloader.py        # NRRD slices + response labels -> encoder features
utils/RegressionLightening.py  # MLP + Lightning training module
utils/model_extractor.py   # save the U-Net encoder weights
utils/transform.py         # resize / normalize
model_param/               # encoder definition and weights
```

## Run

```bash
pip install torch pytorch-lightning segmentation-models-pytorch torchmetrics pynrrd pandas openpyxl wandb
python main.py
```

Set the data paths (`root_dir`, `batch_path`, `response_dir`) in `utils/dataloader.py` and the encoder path in `main.py`. The imaging data aren't included.
