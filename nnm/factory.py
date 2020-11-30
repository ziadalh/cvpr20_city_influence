import torch
import torch.optim as optim


def get_optimizer(params, opts):
    """
    Returns an optimizer object
    """
    optimizer_name = getattr(opts, 'optimizer', 'SGD')
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(params,
                              lr=getattr(opts, 'lr', 0.001),
                              momentum=getattr(opts, 'momentum', 0),
                              weight_decay=getattr(opts, 'weight_decay', 0),
                              nesterov=getattr(opts, 'nesterov', False)
                              )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(params,
                               lr=getattr(opts, 'lr', 0.001),
                               betas=(getattr(opts, 'adam_beta1', 0.9),
                                      getattr(opts, 'adam_beta2', 0.999)),
                               weight_decay=getattr(opts, 'weight_decay', 0),
                               amsgrad=getattr(opts, 'adam_amsgrad', False)
                               )
    else:
        optimizer_fn = getattr(optim, optimizer_name)
        logger.warning('only the learning rate option is supported for <{}>'.
                       format(optimizer_name))
        optimizer = optimizer_fn(params,
                                 lr=getattr(opts, 'lr', 0.001),
                                 )
    return optimizer
