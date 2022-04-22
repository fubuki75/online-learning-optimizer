# -*- coding: utf-8 -*-
import torch
import time
class SWGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, N=3):
        defaults = dict(lr=lr, N=N)
        super().__init__(params, defaults)
        self.step_num = 0
        self.total_time = 0
    def step(self):
        start = time.time()
        for group in self.param_groups:
            #per-parameter
            for p in group['params']:
                if p.grad is None:
                    continue
                #求当前系数下的梯度
                gt = p.grad.data
                
                #state用于记录所有变量的当前值
                #state将包含window_step,N个历史梯度
                state = self.state[p]
                #如果是第一次更新的话，就会为每个参数创建中间量，并初始化为0
                if len(state) == 0:
                    # State initialization
                    state['window_step'] = 0
                    
                #将state中的量传出来
                ws = state['window_step']
                state[ws] = gt.clone().detach()
                state['window_step'] += 1
                ws = state['window_step']
                #超参数的带入，包括学习率与betas
                niu = group['lr']
                N = group['N']

                #如果到数目了进行判断
                if ws>=N:
                    #判断是否同向
                    k=True
                    mid={}
                    gt_total=0
                    
                    for i in range(N):
                        mid[i]=(state[i]<0)
                        a=mid[0]
                        if not(a.equal(mid[i])):
                            k=False        
                    
                    if k:
                        #同向计算平均
                        for i in range(N):
                            gt_total+=state[i]
                        gt_avg=gt_total/N
                        p.data+=-gt_avg*niu
                        self.state[p]={}
                    else:#方向不一致的话
                        state[(ws-1)%N] = gt.clone().detach()
                
        self.step_num += 1
        self.total_time += time.time() - start
'''
if __name__=='__main__':
    model=torch.nn.Sequential(torch.nn.Linear(1,1),
                              )
    optimizer = SWGD(model.parameters())
    loss_func=torch.nn.MSELoss()
    ###数据
    for i in range(100):
        x=torch.randn((4,1))
        y=x*x
        model.train()
        pred=model(x)
        loss=loss_func(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()'''