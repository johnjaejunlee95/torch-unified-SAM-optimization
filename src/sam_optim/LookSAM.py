import torch
from typing import Optional, Callable, Dict, Any


class LookSAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        k: int = 5,
        alpha: float = 0.7,
        eps: float = 1e-12,
        layerwise: bool = False,
        **kwargs,
    ):
        assert rho >= 0.0, "rho must be non-negative"
        assert k >= 1, "k must be >= 1"
        assert alpha >= 0.0, "alpha must be non-negative"
        assert eps > 0.0, "eps must be positive"

        defaults = dict(rho=rho, k=k, alpha=alpha, eps=eps, layerwise=layerwise, **kwargs)
        super().__init__(params, defaults)

        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['rho', 'k', 'alpha', 'eps', 'layerwise']}
        self.base_optimizer = base_optimizer(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self._step_count = 0

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        if self._get_group_value("layerwise"):
            self._perturb_layerwise()
        else:
            self._perturb_global()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        assert closure is not None, "LookSAM requires closure (full forward-backward)."
        closure = torch.enable_grad()(closure)

        self._step_count += 1
        k = self._get_group_value("k")
        refresh = (self._step_count % k == 0)

        if refresh:
            loss = closure()
            self._save_current_grads_as("g_ref")

            self.first_step(zero_grad=True)
            closure()
            
            self._compute_and_store_gv_from_saved_g("g_ref")

            self.second_step(zero_grad=False)
            self.base_optimizer.step()
            self.zero_grad(set_to_none=True)
        else:
            loss = closure()
            self._apply_looksam_reuse_update()

        return loss

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    @torch.no_grad()
    def _perturb_global(self):
        grad_norm = self._grad_norm_global()
        eps = self._get_group_value("eps")

        for group in self.param_groups:
            rho = group.get("rho", self._get_group_value("rho"))
            scale = rho / (grad_norm + eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad, alpha=scale.to(p).item())

    @torch.no_grad()
    def _perturb_layerwise(self):
        eps = self._get_group_value("eps")
        
        for group in self.param_groups:
            rho = group.get("rho", self._get_group_value("rho"))
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                self.state[p]["old_p"] = p.data.clone()
                
                param_norm = p.data.norm(p=2)
                grad_norm = p.grad.norm(p=2)
                adaptive_rate = param_norm / (grad_norm + eps)
                
                grad_normalized = p.grad / (p.grad.norm(p=2) + eps)
                
                perturbation = rho * adaptive_rate * grad_normalized
                p.add_(perturbation)

    def _get_group_value(self, key: str):
        return self.param_groups[0].get(key, self.defaults.get(key))

    def _iter_params_with_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    yield p

    @torch.no_grad()
    def _save_current_grads_as(self, name: str):
        for p in self._iter_params_with_grad():
            self.state[p][name] = p.grad.detach().clone()

    @torch.no_grad()
    def _compute_and_store_gv_from_saved_g(self, g_name: str):
        eps = self._get_group_value("eps")
        device = self.param_groups[0]["params"][0].device

        dot = torch.zeros((), device=device)
        g_norm_sq = torch.zeros((), device=device)
        gs_norm_sq = torch.zeros((), device=device)

        for p in self._iter_params_with_grad():
            g = self.state[p].get(g_name, None)
            if g is None:
                continue
            gs = p.grad.detach()
            
            dot = dot + torch.sum(g * gs)
            g_norm_sq = g_norm_sq + torch.sum(g * g)
            gs_norm_sq = gs_norm_sq + torch.sum(gs * gs)

        g_norm = torch.sqrt(g_norm_sq + eps)
        gs_norm = torch.sqrt(gs_norm_sq + eps)
        
        cos_theta = dot / (g_norm * gs_norm + eps)

        gv_norm_sq = torch.zeros((), device=device)
        for p in self._iter_params_with_grad():
            g = self.state[p].get(g_name, None)
            gs = p.grad.detach()
            
            if g is None:
                gv = gs.clone()
            else:
                proj_magnitude = gs_norm * cos_theta
                g_normalized = g / (g_norm + eps)
                gv = gs - proj_magnitude.to(gs) * g_normalized
            
            self.state[p]["gv"] = gv
            gv_norm_sq = gv_norm_sq + torch.sum(gv * gv)

        self.state["gv_norm"] = torch.sqrt(gv_norm_sq + eps)

    @torch.no_grad()
    def _apply_looksam_reuse_update(self):
        alpha = self._get_group_value("alpha")
        eps = self._get_group_value("eps")

        gv_norm = self.state.get("gv_norm", None)
        if gv_norm is None or not torch.isfinite(gv_norm) or gv_norm.item() < eps:
            self.base_optimizer.step()
            self.zero_grad(set_to_none=True)
            return

        g_norm = self._grad_norm_global()
        scale = alpha * (g_norm / (gv_norm + eps))

        for p in self._iter_params_with_grad():
            gv = self.state[p].get("gv", None)
            if gv is None:
                continue
            p.grad.add_(gv.to(p.grad), alpha=scale.to(p.grad).item())

        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)

    def _grad_norm_global(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norm_sq = torch.zeros((), device=device)
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norm_sq = norm_sq + torch.sum(p.grad.detach() ** 2)
        
        return torch.sqrt(norm_sq)

    def _grad_norm_group(self, group) -> torch.Tensor:
        device = group["params"][0].device
        norm_sq = torch.zeros((), device=device)
        
        for p in group["params"]:
            if p.grad is None:
                continue
            norm_sq = norm_sq + torch.sum(p.grad.detach() ** 2)
        
        return torch.sqrt(norm_sq)


# LookLayerSAM: LookSAM with layer-wise perturbation enforced
class LookLayerSAM(LookSAM):
    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        k: int = 5,
        alpha: float = 0.7,
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(
            params=params,
            base_optimizer=base_optimizer,
            rho=rho,
            k=k,
            alpha=alpha,
            eps=eps,
            layerwise=True,
            **kwargs,
        )