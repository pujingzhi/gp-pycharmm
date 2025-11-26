# Calculate single-point energy and forces at AM1/MM 
#

#Import Modules
import os
import sys
import numpy as np

try:
    charmm_lib_dir = os.environ['CHARMM_LIB_DIR']
    charmm_data_dir = os.environ['CHARMM_DATA_DIR']
except KeyError:
    print('please set environment variables',
          'CHARMM_LIB_DIR and CHARMM_DATA_DIR',
          file=sys.stderr)
    sys.exit(1)

import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.lingo as stream
import pycharmm.select as select
from pycharmm.lib import charmm as libcharmm

####################
### CHARMM BLOCK ###
####################
# CHARMM Parameters

# Read Topology and parameter files
rtf_fn = 'toppar/top_all27_prot_na.rtf'
read.rtf(rtf_fn)

stream.charmm_script('stream toppar/amm.top')
stream.charmm_script('stream toppar/mecl.top')

prm_fn = 'toppar/par_all27_prot_na.prm'
read.prm(prm_fn)

#old_warn_level = settings.set_warn_level(-2)
#old_bomb_level = settings.set_bomb_level(-3)

#settings.set_warn_level(old_warn_level)
#settings.set_bomb_level(old_bomb_level)

stream.charmm_script('bomblev -2')

# read in the system
read.psf_card('gen1.psf')
read.coor_card('gen1.crd')

# Electrostatics 
# Note that charmm seems to have issues with more than 7 keywords
# use update: True to set all options
# problem seemed to be fixed by recompiling in gnu enviorment without mkl
dict_nbonds = { 'group': True,
        'fswitch': True,
        'noextend': True,
        'cdie': True,
        'vdw': True,
        'vswitch': True,
        'eps': 1.0,
        'cutnb': 14.0,
        'ctofnb': 13.0,
        'ctonnb': 12.0,
        'vgroup': True,
        'WMIN': 1.2,
        'inbf': 25,
        }
nbond_ewald = pycharmm.NonBondedScript(**dict_nbonds)
nbond_ewald.run()

# Initiate MNDO. No pycharmm module presently availible 
stream.charmm_script('define qm sele segid AMM .or. segid MECL end')
stream.charmm_script('quantum sele qm end am1 char 0 nocut')

stream.charmm_script('open unit 23 read unform name men_resd_am1.dcd')
stream.charmm_script('trajectory query unit 23')
stream.charmm_script(f'trajectory iread 23 nread 1 nfile ?NFILE')

# Torsion scan
#-------------
sqm_energy = []
sqm_force = []
crd = []

stream.charmm_script(f'set NFILE ?NFILE')
nfile = stream.get_charmm_variable('NFILE')
for l in range(0,nfile):
    stream.charmm_script('TRAJ READ')

    energy.show()
    sqm_energy.append(stream.get_energy_value('ENER'))

    forces = coor.get_forces()
    sqm_force.append(forces.to_numpy())

    positions = coor.get_positions()
    crd.append(positions.to_numpy())

sqm_energy = np.array(sqm_energy)

np.savez_compressed('sqm_data', sqm_energy=sqm_energy, sqm_force=sqm_force, crd=crd)

exit()



