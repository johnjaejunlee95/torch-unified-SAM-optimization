import torch
from src.utils.bypass_bn import disable_running_stats, enable_running_stats

class FisherSAM(torch.optim.Optimizer):
    """
    Fisher SAM: Information Geometry and Sharpness Aware Minimization"
    """
    def __init__(self, 
                 params, 
                 base_optimizer,
                 model, 
                 gamma=0.1, 
                 eta=1.0, 
                 **kwargs):

        assert gamma >= 0.0, f"Invalid gamma, should be non-negative: {gamma}"
        assert eta >= 0.0, f"Invalid eta, should be non-negative: {eta}"

        defaults = dict(gamma=gamma, eta=eta, **kwargs)
        super().__init__(params, defaults)

        self.model = model
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):

        denom = self._compute_denominator()

        for group in self.param_groups:
            gamma = group["gamma"]
            eta = group["eta"]

            scale = gamma / (denom + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                self.state[p]["old_p"] = p.data.clone()

                fisher_diag = p.grad.pow(2)

                inv_fisher_diag = 1.0 / (1.0 + eta * fisher_diag)

                e_w = inv_fisher_diag * p.grad * scale.to(p)
                
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):

        assert closure is not None, "FisherSAM requires closure, but it was not provided."
        closure = torch.enable_grad()(closure)

        logits, loss = closure()
        
        self.first_step(zero_grad=True)
        disable_running_stats(self.model)
        closure()
        self.second_step(zero_grad=True)

        enable_running_stats(self.model)
        
        return logits, loss

    def _compute_denominator(self) -> torch.Tensor:

        shared_device = self.param_groups[0]["params"][0].device
        total = torch.tensor(0.0, device=shared_device)

        for group in self.param_groups:
            eta = group["eta"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                fisher_diag = p.grad.pow(2)
                
                inv_fisher_diag = 1.0 / (1.0 + eta * fisher_diag)

                contribution = torch.sum(p.grad * inv_fisher_diag * p.grad)
                total = total + contribution.to(shared_device)

        return torch.sqrt(total)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
