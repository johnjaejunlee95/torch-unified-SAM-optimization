import torch
import torch.nn.functional as F

class ESAM(torch.optim.Optimizer):
    
    def __init__(self, 
                 params, 
                 base_optimizer, 
                 model,
                 rho=0.05, 
                 beta=0.5, 
                 gamma=0.5, 
                 adaptive=False, 
                 **kwargs):
      
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 < beta <= 1.0, f"Invalid beta, should be in (0, 1]: {beta}"
        assert 0.0 < gamma <= 1.0, f"Invalid gamma, should be in (0, 1]: {gamma}"

        defaults = dict(rho=rho, beta=beta, gamma=gamma, adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.model = model
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            swp_scale = scale / group["beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                
                if group["beta"] < 1.0:
                    mask = (torch.rand_like(p) < group["beta"]).float()
                else:
                    mask = 1.0

                adaptive_factor = torch.pow(p, 2) if group["adaptive"] else 1.0
                e_w = mask * swp_scale.to(p) * adaptive_factor * p.grad
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: 
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    
    def step(self, inputs, targets, loss_fct):
        self.model.train()
        
        logits = self.model(inputs)
        losses = F.cross_entropy(logits, targets, reduction='none')
        loss = losses.mean()
        
        l_before = losses.clone().detach()
        loss.backward()        
        self.first_step(zero_grad=True)

        with torch.no_grad():
            self.model.eval()
            logits_adv = self.model(inputs)
            l_after = F.cross_entropy(logits_adv, targets, reduction='none')
            
            instance_sharpness = l_after - l_before
            
            gamma = self.param_groups[0]['gamma']
            batch_size = len(targets)
            
            if gamma >= 1.0:
                indices = range(batch_size)
            else:
                k = int(batch_size * gamma)
                _, topk_indices = torch.topk(instance_sharpness, k)
                indices = topk_indices
        
        loss_subset = loss_fct(self.model(inputs[indices]), targets[indices])
        loss_mean = loss_subset.mean()
        loss_mean.backward()
        
        self.second_step(zero_grad=True)
        
        return loss_subset, logits