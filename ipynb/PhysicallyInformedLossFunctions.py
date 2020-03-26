import torch
import torch.nn as nn
import torch.nn.functional as F

class PCE_Loss(nn.Module):
    """
    This class contains loss functions that use a mean-squared-error loss for reasonable predictions.
    They inherit from torch.nn.Module just like the custom model. For physically unreasonable conditions,
    prediction loss is more severely calculated. What qualifies as reasonable is based on empirically
    gathered datasets and literature reported boundaries of performance in P3HT:PCBM OPV devices.
    
    For the following Power Conversion Efficiency predictions that are improbable, the loss is penalized:
    - PCE < 0%
    - PCE > 6%
    """

    def __init__(self):
        super(PCE_Loss, self).__init__()

    @staticmethod   
#     def forward(ctx, predictions, labels):
    def forward(predictions, labels):
        for x, y in predictions, labels:

            if x < 0:
                #Exponential MSE
                result = F.mse_loss(x,y)
                result = torch.pow(result, 2)


            elif x > 6:
                #exponential MSE
                result = F.mse_loss(x,y)
                result = result.pow(loss, 2)

            else:
                result = F.mse_loss()

            #tell contex object to save operation tensor for autograd.backward
            ctx.save_for_backward(result)


#                 result.requires_grad = True
#                 result.retain_grad()

            return result

    @staticmethod        
#     def backward(ctx, grad_output):
    def backward(grad_output):

        result, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)



        return grad_input, grad_weight, grad_bias


class Voc_Loss(nn.Module):
    """
    This class contains loss functions that use a mean-squared-error loss for reasonable predictions.
    They inherit from torch.nn.Module just like the custom model. For physically unreasonable conditions,
    prediction loss is more severely calculated. What qualifies as reasonable is based on empirically
    gathered datasets and literature reported boundaries of performance in P3HT:PCBM OPV devices.
    
     For the following open-circuitt voltage predictions that are improbable, the loss is penalized:
    - Voc < 0
    - Voc > 1.0
    """

    def __init__(self):
        super(Voc_Loss, self).__init__()

    @staticmethod   
    def forward(predictions, labels):
        for x, y in predictions, labels:

            if x < 0:
                #Exponential MSE
                result = F.mse_loss(x,y)
                result = torch.pow(result, 2)


            elif x > 1.0:
                #exponential MSE
                result = F.mse_loss(x,y)
                result = result.pow(loss, 2)

            else:
                result = F.mse_loss()

            #tell contex object to save operation tensor for autograd.backward
            ctx.save_for_backward(result)


#                 result.requires_grad = True
#                 result.retain_grad()

            return result

    @staticmethod        
    def backward(grad_output):

        result, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)


        return grad_input, grad_weight, grad_bias


class Jsc_Loss(nn.Module):
    """
    This class contains loss functions that use a mean-squared-error loss for reasonable predictions.
    They inherit from torch.nn.Module just like the custom model. For physically unreasonable conditions,
    prediction loss is more severely calculated. What qualifies as reasonable is based on empirically
    gathered datasets and literature reported boundaries of performance in P3HT:PCBM OPV devices.
    
     For the following short-circuit current predictions that are improbable, the loss is penalized:
    - Jsc < 0
    - Jsc > 10
    """

    def __init__(self):
        super(Jsc_Loss, self).__init__()

    @staticmethod   
    def forward(predictions, labels):
        for x, y in predictions, labels:

            if x < 0:
                #Exponential MSE
                result = F.mse_loss(x,y)
                result = torch.pow(result, 2)


            elif x > 10:
                #exponential MSE
                result = F.mse_loss(x,y)
                result = result.pow(loss, 2)

            else:
                result = F.mse_loss()

            #tell contex object to save operation tensor for autograd.backward
            ctx.save_for_backward(result)


#                 result.requires_grad = True
#                 result.retain_grad()

            return result

    @staticmethod        
    def backward(grad_output):

        result, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)



        return grad_input, grad_weight, grad_bias

class FF_Loss(nn.Module):
    """
    This class contains loss functions that use a mean-squared-error loss for reasonable predictions.
    They inherit from torch.nn.Module just like the custom model. For physically unreasonable conditions,
    prediction loss is more severely calculated. What qualifies as reasonable is based on empirically
    gathered datasets and literature reported boundaries of performance in P3HT:PCBM OPV devices.
    
     For the following Fill Factor predictions that are improbable, the loss is penalized:
    - FF < 10
    - FF > 85
    """

    def __init__(self):
        super(FF_Loss, self).__init__()

    @staticmethod   
    def forward(predictions, labels):
        for x, y in predictions, labels:

            if x < 10:
                #Exponential MSE
                result = F.mse_loss(x,y)
                result = torch.pow(result, 2)


            elif x > 85:
                #exponential MSE
                result = F.mse_loss(x,y)
                result = result.pow(loss, 2)

            else:
                result = F.mse_loss()

            #tell contex object to save operation tensor for autograd.backward
            ctx.save_for_backward(result)


#                 result.requires_grad = True
#                 result.retain_grad()

            return result

    @staticmethod        
    def backward(grad_output):

        result, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
        
        