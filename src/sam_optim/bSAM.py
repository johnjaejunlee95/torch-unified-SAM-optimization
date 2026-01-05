import torch
import torch.nn.functional as F
from src.utils.bypass_bn import disable_running_stats, enable_running_stats

class BayesianSAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        model=None,              
        lr: float = 1e-3,
        rho: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        wdecay: float = 0.0,
        msharpness: int = 1,
        Ndata: int = 50000,
        s_init: float = 1.0,
        damping: float = 1e-8,
        eps: float = 1e-12,
        bn_update_once: bool = True,
    ):
        assert lr > 0
        assert 0.0 <= beta1 < 1.0
        assert 0.0 <= beta2 < 1.0
        assert rho >= 0.0
        assert msharpness >= 1
        assert Ndata >= 1
        assert s_init > 0
        assert damping >= 0
        assert eps > 0

        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, wdecay=wdecay, rho=rho,
            msharpness=msharpness, Ndata=Ndata, s_init=s_init, damping=damping,
            eps=eps, bn_update_once=bn_update_once
        )
        super().__init__(params, defaults)

        self.model = model

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    st = self.state[p]
                    st["gm"] = torch.zeros_like(p)
                    st["s"]  = torch.ones_like(p) * s_init
                    st["old_p"] = torch.empty_like(p)

    @torch.no_grad()
    def _stash_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                self.state[p]["old_p"].copy_(p.data)

    @torch.no_grad()
    def _restore_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                p.data.copy_(self.state[p]["old_p"])

    @torch.no_grad()
    def _add_noise_sample(self):
        for group in self.param_groups:
            Ndata = group["Ndata"]
            eps   = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                s = self.state[p]["s"]
                scale = torch.sqrt(1.0 / (Ndata * s + eps))
                p.add_(torch.randn_like(p) * scale)

    @torch.no_grad()
    def _add_perturbation(self, g_snapshot):
        """w <- w + rho * g / s   (elementwise)"""
        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                s = self.state[p]["s"]
                g = g_snapshot[p]
                p.add_(rho * g / (s + eps))

    @torch.no_grad()
    def step(self, x, y, lrfactor: float = 1.0):
        assert x is not None and y is not None
        
        self._stash_params()

        m = self.param_groups[0]["msharpness"]
        
        if x.size(0) < m:
            xs, ys = [x], [y]
            real_m = 1
        else:
            xs = torch.chunk(x, m, dim=0)
            ys = torch.chunk(y, m, dim=0)
            real_m = m

        logits_lists = []
        loss_lists = []
        
        g_eps_sum = {}
        gs_sum = {}    
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    g_eps_sum[p] = torch.zeros_like(p)
                    gs_sum[p] = torch.zeros_like(p)

        # for the first time, update BN stats only once
        if self.model is not None:
            enable_running_stats(self.model)
            self.model(x)                   
            disable_running_stats(self.model)
        
        for k in range(real_m):
            xk, yk = xs[k], ys[k]

            self._restore_params()
            self.zero_grad(set_to_none=True)

            self._add_noise_sample()
            
            with torch.enable_grad():
                logits = self.model(xk)
                loss1 = F.cross_entropy(logits, yk)
                loss1.backward()
            
            logits_lists.append(logits.detach())
            
            g_snapshot = {}
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    
                    g = p.grad.detach().clone()
                    g_snapshot[p] = g
                    
                    s = self.state[p]["s"]
                    term = torch.sqrt(s * (g.pow(2)) + 1e-12)
                    gs_sum[p].add_(term)

            self._restore_params()
            self.zero_grad(set_to_none=True)
            self._add_perturbation(g_snapshot)

            with torch.enable_grad():
                logits2 = self.model(xk)
                loss2 = F.cross_entropy(logits2, yk)
                loss2.backward()
            
            loss_lists.append(loss2.detach())
            
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    g_eps_sum[p].add_(p.grad.detach())

        logits_lists = torch.cat(logits_lists, dim=0)
        loss_lists = torch.stack(loss_lists)
        
        self._restore_params()

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    g_eps_sum[p].div_(real_m)
                    gs_sum[p].div_(real_m)

        for group in self.param_groups:
            lr = group["lr"] * lrfactor
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            wdecay = group["wdecay"]
            damping = group["damping"]
            eps = group["eps"]

            for p in group["params"]:
                if not p.requires_grad: continue
                
                st = self.state[p]
                gm = st["gm"]
                s = st["s"]

                g_eps = g_eps_sum[p] 
                gs = gs_sum[p]      

                gm.mul_(beta1).add_((1.0 - beta1) * (g_eps + wdecay * p.data))
                p.data.add_(-lr * gm / (s + eps))

                s.mul_(beta2).add_((1.0 - beta2) * (gs + damping + wdecay))

        self.zero_grad(set_to_none=True)
        
        if self.model is not None:
            enable_running_stats(self.model) 

        return logits_lists, loss_lists, lr