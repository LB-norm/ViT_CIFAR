# CIFAR-10/100 Classification

Final graded project for the course **AI Systems 2**. I implemented a Vision Transformer (ViT) from scratch and evaluated it with and without pretraining on the CIFAR-10/100 datasets [2]. I then fine-tuned two pretrained ViTs from the original ViT paper [1], using weights available on Hugging Face [3][4].

## Results (CIFAR-10 test accuracy)

| Model / Setting                         | Pretraining          | Notes                         | Accuracy |
|----------------------------------------|----------------------|-------------------------------|---------:|
| ViT from scratch                       | None                 | single model                  | 83.93%   |
| ViT from scratch (ensemble of 5)       | None                 | simple averaging              | 85.52%   |
| ViT from scratch                       | CIFAR-100            | single model                  | 86.93%   |
| ViT from scratch (ensemble of 5)       | CIFAR-100            | simple averaging              | 87.93%   |
| ViT-Base                               | ImageNet-21k         | gradient clipping             | 98.92%   |
| ViT-Base                               | ImageNet-21k         | weight decay                  | 98.94%   |
| ViT-Base                               | ImageNet-21k         | grad clip + weight decay      | 98.98%   |
| ViT-Large                              | ImageNet-21k         | grad clip + weight decay      | 99.13%   |

> Note: “ensemble of 5” = average of model outputs at test time.

## Setup
```bash
# bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```powershell
# powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## References

1. Dosovitskiy, A. et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR. arXiv:2010.11929.
2. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* Technical Report (CIFAR-10/100).
3. Hugging Face model card: `google/vit-base-patch16-224-in21k`.
4. Wightman, R. *timm: PyTorch Image Models.* GitHub.
