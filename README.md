# Oil Spill detection with SegFormer

## Training

Training parameters are in configs/train_config.yaml

```bash
python train.py
```

### Creating patches from images
To split the images and store them

```bash
python data/split_into_patches.py -i input_dataset_path -o output_dataset_path
 [-p patch_size -ov overlap -s dataset_split]
 ```

## Inference

Inference parameters are in configs/test_config.yaml
 
### Inference on the whole image

```bash
python inference.py
```

### Inference on smaller patches

```bash
python inference_on_patches.py [--use_ground_truth]
```
