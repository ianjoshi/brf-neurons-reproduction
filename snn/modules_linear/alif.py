import torch
from .. import functional
from .linear_layer import LinearMask

################################################################
# Neuron update functional
################################################################

# default values for time constants
DEFAULT_ALIF_TAU_M = 20.
DEFAULT_ALIF_TAU_ADP = 20.

# base threshold
DEFAULT_ALIF_THETA = 0.01

DEFAULT_ALIF_BETA = 1.8


def alif_update(
        x: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        a: torch.Tensor,
        alpha: torch.Tensor,
        rho: torch.Tensor,
        beta: float = DEFAULT_ALIF_BETA,
        theta: float = DEFAULT_ALIF_THETA
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # adapt spike accumulator.
    a = a.mul(rho) + z.mul(1.0 - rho)

    # determine dynamic threshold.
    theta_t = theta + a.mul(beta)

    # membrane potential update
    u = u.mul(alpha) + x.mul(1.0 - alpha)

    # generate spike.
    z = functional.StepDoubleGaussianGrad.apply(u - theta_t)

    # reset membrane potential.
    # soft reset (keeps remaining membrane potential)
    u = u - z.mul(theta_t)

    return z, u, a


################################################################
# Layer classes
################################################################

class ALIFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            adaptive_tau_mem_mean: float,
            adaptive_tau_mem_std: float,
            adaptive_tau_adp_mean: float,
            adaptive_tau_adp_std: float,
            tau_mem: float = DEFAULT_ALIF_TAU_M,
            adaptive_tau_mem: bool = True,
            tau_adp: float = DEFAULT_ALIF_TAU_ADP,
            adaptive_tau_adp: bool = True,
            bias: bool = False,
            mask_prob: float = 0.,
            pruning: bool = False,
            use_linear_decay: bool = True  # [MOD] Added toggle
    ) -> None:
        super(ALIFCell, self).__init__()

        self.use_linear_decay = use_linear_decay  # [MOD]

        self.linear = torch.nn.Linear(input_size, layer_size, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear.weight)

        tau_mem = tau_mem * torch.ones(layer_size)
        if adaptive_tau_mem:
            self.tau_mem = torch.nn.Parameter(tau_mem)
            torch.nn.init.normal_(self.tau_mem, mean=adaptive_tau_mem_mean, std=adaptive_tau_mem_std)
        else:
            self.register_buffer("tau_mem", tau_mem)

        tau_adp = tau_adp * torch.ones(layer_size)
        if adaptive_tau_adp:
            self.tau_adp = torch.nn.Parameter(tau_adp)
            torch.nn.init.normal_(self.tau_adp, mean=adaptive_tau_adp_mean, std=adaptive_tau_adp_std)
        else:
            self.register_buffer("tau_adp", tau_adp)

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z, u, a = state
        in_sum = self.linear(x)

        tau_mem = torch.abs(self.tau_mem)
        tau_adp = torch.abs(self.tau_adp)

        if self.use_linear_decay:  # [MOD]
            alpha = 1.0 - (1.0 / tau_mem)  # [MOD]
            rho = 1.0 - (1.0 / tau_adp)  # [MOD]
            alpha = torch.clamp(alpha, 0.0, 1.0)  # [MOD]
            rho = torch.clamp(rho, 0.0, 1.0)  # [MOD]
        else:
            alpha = torch.exp(-1.0 / tau_mem)  # [MOD]
            rho = torch.exp(-1.0 / tau_adp)  # [MOD]

        z, u, a = alif_update(x=in_sum, z=z, u=u, a=a, alpha=alpha, rho=rho)
        return z, u, a



class ALIFCellBP(ALIFCell):
    def __init__(
            self,
            *args,
            bit_precision: int = 32,
            use_linear_decay: bool = True,  # [MOD] New toggle
            **kwargs
    ) -> None:
        super().__init__(*args, use_linear_decay=use_linear_decay, **kwargs)  # [MOD]

        self.bit_precision = bit_precision
        self.use_linear_decay = use_linear_decay  # [MOD]

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, u, a = state

        in_sum = self.linear(x)

        tau_mem = torch.abs(self.tau_mem)
        tau_adp = torch.abs(self.tau_adp)

        if self.use_linear_decay:  # [MOD]
            alpha = 1.0 - (1.0 / tau_mem)
            rho = 1.0 - (1.0 / tau_adp)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            rho = torch.clamp(rho, 0.0, 1.0)
        else:
            alpha = torch.exp(-1.0 / tau_mem)
            rho = torch.exp(-1.0 / tau_adp)

        alpha = functional.quantize_tensor(alpha, self.bit_precision)
        rho = functional.quantize_tensor(rho, self.bit_precision)

        z, u, a = alif_update(
            x=in_sum,
            z=z,
            u=u,
            a=a,
            alpha=alpha,
            rho=rho,
        )

        return z, u, a



