
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
class LossFunction(Function):

    # 必须是staticmethod
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    def forward(ctx, input, sigmoid):
        ctx.save_for_backward(input, sigmoid)
        output = input * 1.0 / (sigmoid ** 2) + torch.log(sigmoid)
        #print(output)
        return output


    @staticmethod
    def backward(ctx, grad_output): 
        input, sigmoid = ctx.saved_tensors
        grad_input = grad_sigmoid = None
        #print(grad_output)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 1.0 / (sigmoid ** 2)
        if ctx.needs_input_grad[1]:
            grad_sigmoid = grad_output * (input * (-2) * 1.0 / (sigmoid ** 3)) + 1.0 / sigmoid
        
        return grad_input, grad_sigmoid

class LossWithUncertainty(nn.Module):
    def __init__(self):
        super(LossWithUncertainty, self).__init__()
        #self.input_loss = SegmentationLosses(cuda=args.cuda).build_loss(mode=args.loss_type)
        #self.output_features = output_features
        #self.input_size = input_size
        #self.target_size = target_size
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # 这个很重要！ Parameters是默认需要梯度的！
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.sigmoid = nn.Parameter(torch.Tensor(1))
        
        # Not a very smart way to initialize weights
        self.sigmoid.data.fill_(1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LossFunction.apply(input, self.sigmoid)


if __name__ == '__main__':
    '''
    from torch.autograd import gradcheck

    input = (Variable(torch.ones(1).double(), requires_grad=True), Variable(torch.ones(1).double(), requires_grad=True),)
    test = gradcheck(LossFunction.apply, input, eps=1e-6, atol=1e-4)
    print(test)
    '''

    input = Variable(torch.ones(1), requires_grad=True)
    sigmoid = Variable(torch.full([1], 5), requires_grad=True)
    z = LossFunction.apply(input, sigmoid)
    z.backward() 
