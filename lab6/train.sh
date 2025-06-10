python3 main.py \
  --mode train \
  --image_dir ~/iclevr \
  --json_path ./train.json \
  --object_path ./objects.json \
  --checkpoint ./ckpt \
  --batch_size 256 \
  --timesteps 1000 \
  --epochs 5000 \
  --image_size 64