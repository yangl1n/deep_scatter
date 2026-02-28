Experiment 1: 5-block, scattering init
```bash
torchrun --nproc_per_node=8 train.py --dataset imagenet --data-dir /path/to/ImageNet \
  --n-blocks 5 --modulus-type complex_modulus --L 4 --lowpass-last --mixing-horizon 27 \
  --kernel-size 7 --global-batch-size 1024 --lr-epochs 20 --epochs 90 \
  --workers 16 --print-freq 100 \
  --save-dir imagenet_check/5b_cmod_4L_h27_scat
```

Experiment 2: 6-block, scattering init

```bash
torchrun --nproc_per_node=8 train.py --dataset imagenet --data-dir /path/to/ImageNet \
  --n-blocks 6 --modulus-type complex_modulus --L 4 --lowpass-last --mixing-horizon 27 \
  --kernel-size 7 --global-batch-size 1024 --lr-epochs 20 --epochs 90 \
  --workers 16 --print-freq 100 \
  --save-dir imagenet_check/6b_cmod_4L_h27_scat
```

Experiment 3: 7-block, scattering init
```bash
torchrun --nproc_per_node=8 train.py --dataset imagenet --data-dir /path/to/ImageNet \
  --n-blocks 7 --modulus-type complex_modulus --L 4 --lowpass-last --mixing-horizon 27 \
  --kernel-size 7 --global-batch-size 1024 --lr-epochs 20 --epochs 90 \
  --workers 16 --print-freq 100 \
  --save-dir imagenet_check/7b_cmod_4L_h27_scat
```

Experiment 4: 5-block, random init (joint)
```bash
torchrun --nproc_per_node=8 train.py --dataset imagenet --data-dir /path/to/ImageNet \
  --n-blocks 5 --modulus-type complex_modulus --L 4 --lowpass-last --mixing-horizon 27 \
  --kernel-size 7 --random-init --joint --epochs 120 \
  --global-batch-size 1024 --workers 16 --print-freq 100 \
  --save-dir imagenet_check/5b_cmod_4L_h27_rand
```