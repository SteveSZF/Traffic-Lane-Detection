CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --lr 0.01 --workers 4 --epochs 80 --batch-size 16 --gpu-ids 0,1,2,3 --checkname mobilenet --eval-interval 1 --dataset bdd100k --loss-type focal
