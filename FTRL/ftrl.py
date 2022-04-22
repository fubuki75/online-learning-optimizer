import torch
from torch.optim.optimizer import Optimizer


class FTRL(Optimizer):
    """ Implements FTRL online learning algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): alpha parameter (default: 1.0)
        beta (float, optional): beta parameter (default: 1.0)
        l1 (float, optional): L1 regularization parameter (default: 1.0)
        l2 (float, optional): L2 regularization parameter (default: 1.0)

    .. _Ad Click Prediction: a View from the Trenches: 
        https://www.eecs.tufts.edu/%7Edsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, params, alpha=1.0, beta=1.0, l1=1.0, l2=1.0):
        #检查超参
        if not 0.0 < alpha:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        if not 0.0 < beta:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= l1:
            raise ValueError("Invalid l1 parameter: {}".format(l1))
        if not 0.0 <= l2:
            raise ValueError("Invalid l2 parameter: {}".format(l2))
        #将defaults打包  
        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super(FTRL, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        #对每个参数循环
        for group in self.param_groups:
            #对每个参数循环
            for p in group["params"]:
                if p.grad is None:
                    continue
                #对每个param求导，这里面的param非常多
                grad = p.grad.data
                #获取这是第几维度,其实是第几个参数
                state = self.state[p]
                #如果是第一维则定义z和n
                if len(state) == 0:
                    #z和n实现累加
                    state["z"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)
                    state['d'] = torch.zeros_like(p.data)

                z, n,dirta = state["z"], state["n"],state['d']
                #几个重要参数的更新 theta是dirta
                theta =( (n + grad ** 2).sqrt()- n.sqrt()) / group["alpha"] 
                z.add_(grad - theta * dirta)
                n.add_(grad ** 2)
                
                #权重的更新
                #origin = p.data #这个origin是我自己加的，想在之前的权重的基础上改
                dirta = (-1/ (group["l2"] + (group["beta"] + n.sqrt()) / group["alpha"])
                    *(z - group["l1"] * z.sign()))
                dirta[z.abs() < group["l1"]] = 0
                p.data=p.data+dirta
                #p.data+=origin，不知道为啥会发生loss-->nan
                #origin

        return loss
'''if __name__=='__main__':
    model=torch.nn.Sequential(torch.nn.Linear(2,10),
                              torch.nn.ReLU(),
                              torch.nn.Linear(10,1)
                              )
    optimizer = FTRL(model.parameters(), alpha=1e-5, beta=1.0, l1=0.1, l2=0.1)
    loss_func=torch.nn.MSELoss()
    ###数据
    x=torch.randn((4,2))
    y=torch.randn((4,1))
    for i in range(1):
        model.train()
        pred=model(x)
        loss=loss_func(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()'''
        
        
        