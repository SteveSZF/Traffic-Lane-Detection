if [ $1 -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m ipdb train.py --lr 0.01 --workers 4 --epochs 80 --batch-size 16 --gpu-ids 0,1,2,3 --checkname erfnet --eval-interval 1 --dataset bdd100k --write-val --resume run/bdd100k/erfnet.bak/experiment_10/checkpoint.pth.tar
else
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --lr 0.01 --workers 4 --epochs 80 --batch-size 16 --gpu-ids 0,1,2,3 --checkname erfnet --eval-interval 1 --dataset bdd100k --write-val --resume run/bdd100k/erfnet.bak/experiment_10/checkpoint.pth.tar
fi
