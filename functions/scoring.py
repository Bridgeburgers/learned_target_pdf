# -*- coding: utf-8 -*-
"""
prediction and evaluation functions for learned target layers
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from functions.loss_function import LearnedPDFMCIntegration, LearnedPDFNumericIntegration

# =============================================================================
# def learned_pdf_eval(model, loss_function, dataloader, device, delta_t=None,
#                      target_range=None):
#     
#     if delta_t is None:
#         try:
#             delta_t = model.delta_t
#         except AttributeError:
#             raise ValueError('delta_t must be provided, or an attribute of model')
#             
#     if target_range is None:
#         try:
#             target_range = model.target_range
#         except AttributeError:
#             raise ValueError('target_range must be provided, or an attribute of model')
#     
#     model.eval()
#     epoch_loss = 0.0
#     for x, t in dataloader:
#         x, t = x.to(device), t.to(device)
#         with torch.no_grad():
#             y, y_range = model(x,t)
#         y, y_range = y.to(device), y_range.to(device)
#         if loss_function == LearnedPDFMCIntegration.apply:
#             loss = loss_function(y, target_range, *y_range.T)
#         elif loss_function == LearnedPDFNumericIntegration.apply:
#             loss = loss_function(y, delta_t, *y_range.T)
#         else:
#             raise ValueError('loss_function must be derived from' 
#                              ' LearnedPDFMCIntegration or '
#                              'LearnedPDFNumericIntegration')
#         batch_loss = loss.item()
#         epoch_loss += batch_loss
#         
#     epoch_loss /= len(dataloader)
#     model.train(True)
#     return(epoch_loss)
# =============================================================================

def learned_pdf_eval(model, loss_function, dataloader, device, delta_t=None,
                     target_range=None):
    
    if delta_t is None:
        try:
            delta_t = model.delta_t
        except AttributeError:
            raise ValueError('delta_t must be provided, or an attribute of model')
            
    if target_range is None:
        try:
            target_range = model.target_range
        except AttributeError:
            raise ValueError('target_range must be provided, or an attribute of model')
    
    model.eval()
    epoch_loss = 0.0
    for data_dict in dataloader:
        for k,v in data_dict.items():
            if torch.is_tensor(v):
                data_dict[k] = v.to(device)
        with torch.no_grad():
            y, y_range = model(**data_dict)
        y, y_range = y.to(device), y_range.to(device)
        if loss_function == LearnedPDFMCIntegration.apply:
            loss = loss_function(y, target_range, *y_range.T)
        elif loss_function == LearnedPDFNumericIntegration.apply:
            loss = loss_function(y, delta_t, *y_range.T)
        else:
            raise ValueError('loss_function must be derived from' 
                             ' LearnedPDFMCIntegration or '
                             'LearnedPDFNumericIntegration')
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
    epoch_loss /= len(dataloader)
    model.train(True)
    return(epoch_loss)

# =============================================================================
# def learned_pdf_predict(model, test_dataset, batch_size=1024):
#     dataloader = DataLoader(test_dataset, batch_size=batch_size)
#     y = np.array([])
#     y_range = None
#     device = model.device
#     for x, t in dataloader:
#         x,t = x.to(device), t.to(device)
#         with torch.no_grad():
#             model.eval()
#             out_y, out_yrange = model.integration_forward(x,t)
#         model.train(True)
#         out_y, out_yrange = out_y.detach().cpu().numpy(), out_yrange.detach().cpu().numpy()
#         y = np.concatenate([y, out_y])
#         y_range = np.concatenate([y_range, out_yrange], axis=0) if y_range is not None else out_yrange
#         
#     t = test_dataset[:][1].detach().cpu().numpy()
#         
#     return y, y_range, t
# =============================================================================


def learned_pdf_predict(model, test_dataset, batch_size=1024):
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    y = np.array([])
    y_range = None
    device = model.device
    for data_dict in dataloader:
        for k,v in data_dict.items():
            if torch.is_tensor(v):
                data_dict[k] = v.to(device)
        with torch.no_grad():
            model.eval()
            out_y, out_yrange = model.integration_forward(**data_dict)
        model.train(True)
        out_y, out_yrange = out_y.detach().cpu().numpy(), out_yrange.detach().cpu().numpy()
        y = np.concatenate([y, out_y])
        y_range = np.concatenate([y_range, out_yrange], axis=0) if y_range is not None else out_yrange
        
    t = test_dataset[:][1].detach().cpu().numpy()
        
    return y, y_range, t

def learned_pdf_score(model, test_dataset, batch_size=1024, 
                      t_range=None, delta_t=None):
    
    if t_range is None:
        try:
            t_range = model.t_range
        except AttributeError:
            raise ValueError('t_range must be provided, or an attribute of model')
            
    if delta_t is None:
        try:
            delta_t = model.delta_t
        except AttributeError:
            raise ValueError('delta_t must be provided, or an attribute of model')
            
    if torch.is_tensor(delta_t):
        delta_t = delta_t.detach().cpu().numpy()
    if torch.is_tensor(t_range):
        t_range = t_range.detach().cpu().numpy()
    
    y, y_range, t = learned_pdf_predict(model, test_dataset, batch_size)
    e_yrange = np.exp(y_range)
    Z = (e_yrange * delta_t).sum(axis=1)
    p_dist = e_yrange/Z.reshape(-1,1)
    p = np.exp(y) / Z
    t_ev = (e_yrange * delta_t * t_range).sum(axis=1) / Z
    
    return p, p_dist, t, t_ev
