# -*- coding: utf-8 -*-
"""
loss function for learned target pdf
"""

import torch
from torch.autograd import Function

def mc_integrate(t_samples, target_range):
    return torch.mean(t_samples * (target_range[1] - target_range[0]), axis=1)

def numeric_integral(function_tensor, delta_t):
    return torch.sum(function_tensor * delta_t, axis=1)

class LearnedPDFMCIntegration(Function):
    
    @staticmethod
    def forward(ctx, y, target_range, *y_range):
        device = y.device
        y_range = torch.stack(y_range, dim=1).to(device)
        Z = mc_integrate(torch.exp(y_range), target_range).to(device)
        ctx.save_for_backward(y, y_range, Z, target_range)
        loss = -y + torch.log(Z)
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        y, y_range, Z, target_range = ctx.saved_tensors
        device = y.device

        y_grad = -grad_output * torch.ones(y.shape).to(device)
        summand = torch.exp(y_range) * (target_range[1] - target_range[0]) / y_range.shape[1]
        y_range_grad = torch.div(summand, Z.reshape(-1,1))
        
        return y_grad, None, *y_range_grad.T


class LearnedPDFNumericIntegration(Function):
    
    @staticmethod
    def forward(ctx, y, delta_t, *y_range):
        device = y.device
        y_range = torch.stack(y_range, dim=1).to(device)
        Z = numeric_integral(torch.exp(y_range), delta_t).to(device)
        ctx.save_for_backward(y, y_range, Z, delta_t)
        loss = -y + torch.log(Z)
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        y, y_range, Z, delta_t = ctx.saved_tensors
        device = y.device
        y_grad = -grad_output * torch.ones(y.shape).to(device)
        y_range_grad = torch.div(torch.exp(y_range) * delta_t, Z.reshape(-1,1))
        
        return y_grad, None, *y_range_grad.T
    
    
class LearnedPDFImportanceSampling(Function):
    
    @staticmethod
    def forward(ctx, y, q, *y_range):
        device = y.device
        y_range = torch.stack(y_range, dim=1).to(device)
        MZ = (torch.exp(y_range) / q).sum(dim=1).to(device) #MZ implies Z * # of MC samples (M)
        M = q.shape[1]
        ctx.save_for_backward(y, y_range, q, MZ)
        loss = -y + torch.log(MZ / M)
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        y, y_range, q, MZ = ctx.saved_tensors
        device = y.device

        y_grad = -grad_output * torch.ones(y.shape).to(device)
        summand = torch.exp(y_range) / q * grad_output
        y_range_grad = torch.div(summand, MZ.reshape(-1,1))
        
        return y_grad, None, *y_range_grad.T