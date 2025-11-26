# Calculate single-point energy and forces at DFT/MM
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
#------ Launch Gaussian ---------------------------------
ai_env = 'envi g09profile "data/G09PROFILE"\n'
ai_env += f'envi g09exe     "/geode2/soft/hps/rhel8/gaussian/g16/g16"\n' #"{gaussian_dir}/g16"\n'
ai_env += 'envi g09fchk    "/geode2/soft/hps/rhel8/gaussian/g16/formchk"\n'  #"{gaussian_dir}/formchk"\n'
ai_env += 'envi g09cmd     "data/md1_g09.G09CMD"\n'
ai_env += 'envi g09inp     "g1"\n'
ai_env += 'envi GAUSS_SCRDIR "scratch"\n'
stream.charmm_script(ai_env)
#--------------------------------------------------------
stream.charmm_script('gaus nogu remove sele qm end')

stream.charmm_script('open unit 23 read unform name men_resd_am1.dcd')
stream.charmm_script('trajectory query unit 23')
stream.charmm_script(f'trajectory iread 23 nread 1 nfile ?NFILE')

# Torsion scan
#-------------
aiqm_energy = []
aiqm_force = []

stream.charmm_script(f'set NFILE ?NFILE')
nfile = stream.get_charmm_variable('NFILE')
for l in range(0,nfile):
    stream.charmm_script('TRAJ READ')

    energy.show()
    aiqm_energy.append(stream.get_energy_value('ENER'))

    forces = coor.get_forces()
    aiqm_force.append(forces.to_numpy())

aiqm_energy = np.array(aiqm_energy)

np.savez_compressed('aiqm_data', aiqm_energy=aiqm_energy, aiqm_force=aiqm_force)

exit()



