import torch
import numpy as np
import argparse as ap
import os
from models.gmx_ani import GmxANIModel
from models.gmx_mace import GmxMACEModel
try:
    from models.gmx_emle import GmxEMLEModel
    have_emle = True
except ImportError:
    have_emle = False
    

_map_atom_number = {
    "h": 1,
    "c": 6,
    "n": 7,
    "o": 8,
    "f": 9,
    "s": 16,
    "cl": 17,
}

def atomNumberFromLine(line):
    atype = line.split()[1].lower()
    # Sort keys by length descending to match 'cl' before 'c'
    for key in sorted(_map_atom_number.keys(), key=len, reverse=True):
        if atype.startswith(key):
            return _map_atom_number[key]
    raise ValueError(f"Atom number couldn't be inferred from line: {line}")

def getAtomNumbers(grofile, ndxfile=None, group=None):
    """
    Get the atomic numbers from the GROMACS index and coordinate files.
    """

    indices = []
    if ndxfile is not None:
        # Read the GROMACS index file
        with open(ndxfile, 'r') as f:
            lines = f.readlines()

        # Find the group of interest
        if group is None:
            group = "non-Water"
            print("Group not specified, using 'non-Water' group.")
        else:
            print(f"Reading group {group} from index file..")

        group_lines = [line for line in lines if line.startswith(f"[ {group} ]")]
        if not group_lines:
            raise ValueError(f"Group '{group}' not found in index file.")
        start_index = lines.index(group_lines[0]) + 1
        # Find the end of the group, i.e. next group or end of file
        end_index = start_index
        while end_index < len(lines) and not lines[end_index].startswith("["):
            end_index += 1
        indices = [int(num) for line in lines[start_index:end_index] for num in line.split()]

    # Read the GROMACS coordinate file
    with open(grofile, 'r') as f:
        lines = f.readlines()[2:]  # Skip the first two lines (title and number of atoms)

    # Get atomic numbers based on indices
    atomic_numbers = []
    if indices:
        for idx in indices:
            for line in lines:
                if int(line.split()[2]) == idx:
                    atomic_numbers.append(atomNumberFromLine(line))
                    break
    else:
        # If not, use all atoms until the first water or EOF
        for line in lines:
            if "SOL" in line or len(line.split()) <= 3:
                break
            # get the atom type and infer number, assuming the first part is the atom type
            atomic_numbers.append(atomNumberFromLine(line))
    assert atomic_numbers, "Atom numbers could not be read from the coordinate file."
    print(f"Read atom numbers from coordinate file: {' '.join(map(str, atomic_numbers))}")
    return torch.tensor(atomic_numbers, dtype=torch.int32)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Wrap a NNP model for use in GROMACS.")
    parser.add_argument("fname", type=str, default="model.pt", help="Model to wrap")
    parser.add_argument("--out", type=str, default="model_gmx.pt", help="Output file name")
    parser.add_argument("--use_opt", type=str, default=None, help="Use optimization (cuaev, nnpops)")
    parser.add_argument("--group", type=str, default=None, help="Index group name")
    parser.add_argument("--ndxfile", type=str, default=None, help="GROMACS Index file")
    parser.add_argument("--grofile", type=str, default=None, help="GROMACS Coordinate file")
    parser.add_argument("--model_index", type=int, default=None, help="Model index")

    # Parse the arguments
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the atomic numbers from the gromacs index and coordinate files
    if args.grofile is not None:
        atomic_numbers = getAtomNumbers(args.grofile, args.ndxfile, args.group)
    else:
        atomic_numbers = None

    # Get the model
    if args.fname.endswith(".pt"):
        # TODO: handle errors and torchscript models
        model = torch.load(args.fname, map_location=device)

    elif "ani" in args.fname:
        if "emle" in args.fname:
            model = GmxEMLEModel(flavor="ani2x", atomic_numbers=atomic_numbers, model_index=args.model_index, dtype=torch.float32, device=device)
            print(f"Saving ANI2x-EMLE model with {args.use_opt} optimization to {args.out}")
        elif args.fname == "ani1x":
            model = GmxANIModel(use_opt=args.use_opt, atomic_numbers=atomic_numbers, model_index=args.model_index, version=1, device=device)
            print(f"Saving ANI-1x model with {args.use_opt} optimization to {args.out}")
        elif args.fname == "ani2x":
            model = GmxANIModel(use_opt=args.use_opt, atomic_numbers=atomic_numbers, model_index=args.model_index, version=2, device=device)
            print(f"Saving ANI-2x model with {args.use_opt} optimization...")
        else:
            raise ValueError("Invalid model name for ANI: {}".format(args.fname))

    elif "mace" in args.fname:
        if "emle" in args.fname:
            model = GmxEMLEModel(flavor="mace", atomic_numbers=atomic_numbers, model_index=args.model_index, dtype=torch.float32, device=device)
            print(f"Saving MACE-EMLE model with {args.use_opt} optimization to {args.out}")
        else:
            raise ValueError("Invalid flavor for MACE model: {}".format(args.fname))

    # TODO: handle MACE models
    else:
        raise ValueError("Invalid model name: {}".format(args.fname))
    
    # Check for extensions
    ext_lib = []
    for lib in torch.ops.loaded_libraries:
        if lib:
            ext_lib.append(lib)
    ext_lib = ":".join(ext_lib)
    print("Loaded extension libraries: ", ext_lib)
    extra_files = {}
    if ext_lib:
        extra_files['extension_libs'] = ext_lib
    
    # Save the model
    assert args.out.endswith(".pt"), "Output file name must end with .pt"
    torch.jit.script(model).save(args.out, _extra_files=extra_files)
    print(f"Saved wrapped model to {args.out}.")