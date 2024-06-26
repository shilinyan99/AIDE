### Training

```
./scripts/train.sh  --data_path [/path/to/train_data] --eval_data_path [/path/to/eval_data] --resnet_path [/path/to/pretrained_resnet_path] --convnext_path [/path/to/pretrained_convnext_path] --output_dir [/path/to/output_dir] [other args]
```

For example, training on ProGAN, run the following command:

```
./scripts/train.sh --data_path dataset/progan/train --eval_data_path dataset/progan/eval --resnet_path pretrained_ckpts/resnet50.pth --convnext_path pretrained_ckpts/open_clip_pytorch_model.bin --output_dir results/progan_train
```

### Inference

Inference using the trained model.
```
./scripts/eval.sh --data_path [/path/to/train_data] --eval_data_path [/path/to/eval_data] --resume [/path/to/progan_train] --eval True --output_dir [/path/to/output_dir]
```

For example, evaluating the progan_train model, run the following command:

```
./scripts/eval.sh --data_path dataset/progan/train --eval_data_path dataset/progan/eval --resume results/progan_train/progan_train.pth --eval True --output_dir results/progan_train
```



