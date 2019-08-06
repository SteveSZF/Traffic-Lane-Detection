from modeling.prune_erfnet_road import *
import torch

model = PERFNet(num_classes_pixel = 3)
model = torch.nn.DataParallel(model).cuda()

erfnet = torch.load('weights_erfnet_road.pth')
for k, v in erfnet.items():
    print(k)

model.load_state_dict(erfnet, strict=False)
for k, v in model.state_dict().items():
    print(k)

torch.save(model.state_dict(), 'prune_erfnet.pth.tar')

