import torch
from emle.models import ANI2xEMLE, MACEEMLE

from typing import Optional
from emle._units import (
    _NANOMETER_TO_ANGSTROM,
    _NANOMETER_TO_BOHR,
    _BOHR_TO_ANGSTROM,
    _HARTREE_TO_KJ_MOL,
)

class GmxEMLEModel(torch.nn.Module):
    def __init__(self, flavour: str, **kwargs):
        super().__init__()

        if flavour == 'ani2x':
            self.model = ANI2xEMLE(**kwargs)
            self.is_nnpops = self.model._is_nnpops
        elif flavour == 'mace':
            self.model = MACEEMLE(**kwargs)
            self.is_nnpops = False

        self.length_conversion = _NANOMETER_TO_ANGSTROM
        self.energy_conversion = _HARTREE_TO_KJ_MOL

    def forward(self, atomic_numbers, positions_nn, positions_mm, charges_mm, qm_charge: Optional[int]=None):
        device = positions_nn.device
        # convert units 
        positions_nn = positions_nn * self.length_conversion
        positions_mm = positions_mm * self.length_conversion

        if not self.is_nnpops:
            positions_nn = positions_nn.unsqueeze(0)
            positions_mm = positions_mm.unsqueeze(0)
            atomic_numbers = atomic_numbers.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)

        if qm_charge is None:
            qm_charge = 0

        # Compute the total energy and gradients
        E = self.model(atomic_numbers, charges_mm, positions_nn, positions_mm, qm_charge)
        E_tot = E.sum() * self.energy_conversion
        dE_dxyz_nn, dE_dxyz_mm = torch.autograd.grad(
            [E_tot], [positions_nn, positions_mm], allow_unused=True
        )

        if dE_dxyz_nn is None:
            forces_nn = torch.zeros_like(positions_nn, device=device)
        else:
            forces_nn = -1. * dE_dxyz_nn * self.length_conversion
        if dE_dxyz_mm is None:
            forces_mm = torch.zeros_like(positions_mm, device=device)
        else:
            forces_mm = -1. * dE_dxyz_mm * self.length_conversion
        
        if not self.is_nnpops:
            forces_nn = forces_nn.squeeze(0)
            forces_mm = forces_mm.squeeze(0)

        return E_tot, forces_nn, forces_mm
    

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # atomic numbers needed for nnpops
    # sequence should be the same as in the .gro file
    # e.g. alanine dipeptide
    # atomic_numbers = torch.tensor([1,6,1,1,6,8,7,1,6,1,6,1,1,1,6,8,7,1,6,1,1,1])
    atomic_numbers = None

    model = GmxEMLEModel(flavour='mace', atomic_numbers=atomic_numbers, dtype=torch.float32, device=device)

    # export model
    # check for torch extension library
    ext_lib = []
    ext_lib.append('/nethome/mullender/anaconda3/envs/emle/lib/python3.12/site-packages/torchani/cuaev.cpython-312-x86_64-linux-gnu.so')
    for lib in torch.ops.loaded_libraries:
        if lib:
            ext_lib.append(lib)
    ext_lib = ":".join(ext_lib)
    print("loaded extension libraries: ", ext_lib)
    
    # save model
    save_path = 'models/mace_emle.pt'
    extra_files = {}
    if ext_lib:
        extra_files['extension_libs'] = ext_lib
    torch.jit.script(model).save(save_path, _extra_files=extra_files)
