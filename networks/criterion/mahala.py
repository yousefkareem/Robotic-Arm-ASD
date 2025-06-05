import torch

def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)

    return diff, mu

def cov_v(diff, num):
    var = torch.matmul(diff.t(), diff) / num
    return var

def mahalanobis(u, v, cov_x, use_precision=False, reduction=True):
    num, dim = v.size()
    if use_precision == True:
        inv_cov = cov_x
    else:
        inv_cov = torch.inverse(cov_x)
    delta = torch.sub(u, v)
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())

    if reduction:
        return torch.sum(m_loss)/num
    else:
        return m_loss, num

def loss_function_mahala(recon_x, x, block_size, cov=None, is_source_list=None, is_target_list=None, update_cov=False, use_precision=False, reduction=True):
    ### Modified mahalanobis loss###
    if update_cov == False:
        # Reshape to match the input dimensions
        recon_x_reshaped = recon_x.view(-1, block_size)
        x_reshaped = x.view(-1, block_size)
        loss = mahalanobis(recon_x_reshaped, x_reshaped, cov, use_precision, reduction=reduction)
        return loss
    else:
        diff = x - recon_x
        # Calculate the actual block size based on the input dimensions
        actual_block_size = diff.size(1)
        cov_diff_source, _ = cov_v_diff(in_v=diff[is_source_list].view(-1, actual_block_size))

        cov_diff_target = None
        is_calc_cov_target = any(is_target_list)
        if is_calc_cov_target:
            cov_diff_target, _ = cov_v_diff(in_v=diff[is_target_list].view(-1, actual_block_size))

        loss = diff**2
        if reduction:
            loss = torch.mean(loss, dim=1)
        
        return loss, cov_diff_source, cov_diff_target

def loss_reduction_mahala(loss):
    return torch.mean(loss)

def calc_inv_cov(model, device="cpu"):
    inv_cov_source=None
    inv_cov_target=None
    
    # Add regularization to prevent singular matrices
    eps = 1e-6
    reg = torch.eye(model.cov_source.size(0)).to(device) * eps
    
    cov_x_source = model.cov_source.data
    cov_x_source = cov_x_source.to(device).float()
    cov_x_source = cov_x_source + reg  # Add regularization
    inv_cov_source = torch.inverse(cov_x_source)
    inv_cov_source = inv_cov_source.to(device).float()
    
    cov_x_target = model.cov_target.data
    cov_x_target = cov_x_target.to(device).float()
    cov_x_target = cov_x_target + reg  # Add regularization
    inv_cov_target = torch.inverse(cov_x_target)
    inv_cov_target = inv_cov_target.to(device).float()
    
    return inv_cov_source, inv_cov_target
