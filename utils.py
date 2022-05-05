def liner_warmup(cur_step, t_step, warmup):
    progress = cur_step / t_step
    if progress < warmup:
        return progress / warmup
    return max((progress - 1.) / (warmup - 1.), 0.)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            # print(param.shape)
            if param.grad == None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)
            