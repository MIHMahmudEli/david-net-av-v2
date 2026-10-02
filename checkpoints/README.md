# Pretrained Model Checkpoints

Model checkpoints for DAVID-Net (`david_net_best.safetensors` and `david_net_lite.safetensors`) can be downloaded automatically or mounted locally.

### Automatic Download
When running the inference API (`api/app.py`), the weights are automatically fetched from Hugging Face Hub if not present locally:
```bash
python -m api.app
```

### Manual Placement
To use a local checkpoint, place `david_net_best.safetensors` in this folder:
```
checkpoints/
└── david_net_best.safetensors
```
