# Pytorch-segmenter
Segmenter for domain semantic segmentation continual learning 

# dataset
Cityscapes and ACDC

# code reference
[Segmenter: Transformer for Semantic Segmentation](https://github.com/rstrudel/segmenter/tree/master)

# pip download
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

# Get start with cityscapes
'''
python -m torch.distributed.launch --nproc_per_node=#   main.py

'''
