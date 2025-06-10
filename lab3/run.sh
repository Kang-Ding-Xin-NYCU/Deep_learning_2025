python inpainting.py --load-transformer-ckpt-path "./transformer_checkpoints/0.6/epoch_185.pt" --sweet-spot 10 --total-iter 10 --mask-func square
cd faster-pytorch-fid/
python fid_score_gpu.py --predicted-path ../test_results --device cuda:0
cd ..
