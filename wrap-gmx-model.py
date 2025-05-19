import torch
import numpy as np
import argparse as ap
import os
from models import *


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Wrap a NNP model for use in GROMACS.")
    parser.add_argument("fname", type=str, help="Model to wrap")
    parser.add_argument("oname", type=str, help="Output file name")
    parser.add_argument("--use_opt", type=str, default=None, help="Use optimization")
    parser.add_argument("--group", type=str, default=None, help="Index group name")
    parser.add_argument("--ndxfile", type=int, default=None, help="GROMACS Index file")
    parser.add_argument("--grofile", type=int, default=None, help="GROMACS Coordinate file")
    parser.add_argument("--model_index", type=int, default=None, help="Model index")
