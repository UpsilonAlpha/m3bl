from __future__ import annotations

import argparse
import os
from typing import List, Set, Dict, Any, Tuple
from openmm.app import PDBFile

from preprocess import (
    save_result,
    load_section,
)

from ash import *


def run(args) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    qmmm = load_section(args.input, "qmmm")

    pdbfile = qmmm["results"]["lastframe"]

    numcores=args.cores
    numimages=8
    
    ################################################
    # Defining reactant and product ASH fragments
    #################################################
    react=Fragment(xyzfile="react.xyz", charge=0, mult=1)
    prod=Fragment(xyzfile="prod.xyz", charge=0, mult=1)

    #Theory to use for NEB
    xtbcalc = xTBTheory(xtbmethod='GFN2', runmode='library', numcores=1)

    #Run NEB to find saddlepoint. Returns an ASH Results object
    NEB_result = NEB(reactant=react, product=prod, theory=xtbcalc, images=numimages, runmode='parallel', numcores=numcores)
    print(NEB_result)

    #Optional NumFreq job on saddlepoint to confirm that a saddlepoint was found.
    NumFreq(theory=xtbcalc, fragment=NEB_result.saddlepoint_fragment)
