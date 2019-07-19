CUDA_VISIBLE_DEVICES=0,1,2,3 python val.py --workers 4 --batch-size 1 --gpu-ids 0 --checkname erfnet --dataset bdd100k  --resume run/bdd100k/erfnet/model_best.pth.tar 
