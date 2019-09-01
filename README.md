# Lane Detection
在BDD100K数据集上进行车道分割检测
## Training
./train_bdd100k.sh  
## Video Test
读取视频帧，并对分割后的结果进行展示  
./video_seg.sh  
![image](https://github.com/SteveSZF/Traffic-Lane-Detection/blob/master/b.jpg) 
![image](https://github.com/SteveSZF/Traffic-Lane-Detection/blob/master/c.jpg)
## Validation
将验证集预测结果和gt结果进行对比展示,并计算单帧前向传播的时间  
./time_val_bdd100k.sh  
![image](https://github.com/SteveSZF/Traffic-Lane-Detection/blob/master/a.jpg)
![image](https://github.com/SteveSZF/Traffic-Lane-Detection/blob/master/d.jpg)  

|模型         | 单帧前传速度（640x480）           | mIoU  |  
|:-------------:|:-------------:|:-----:|  
| erfnet      | 18.98ms | 75.44% |
| resnet101-deeplabv3+      | 26.92ms      |   87.21% |
| mobilenetv2-deeplabv3+ | 17.73ms      |    76.73% |
| mobilenetv2-focal-loss-deeplabv3+ | 17.73ms      |    68.10% |
| resnet18-bisenet | 7.68ms      |    76.32% |
| ENet | 15.14ms      |    81.32% |
