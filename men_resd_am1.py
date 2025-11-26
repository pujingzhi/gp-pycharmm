# RESD RC scan of Menshutkin reaction at AM1/MM 
#

#Import Modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

# Set up PBC
crystal.define_cubic(length=40)
stream.charmm_script('crystal build Noper 0 cutoff 16.0')

stream.charmm_script('image byseg xcen 0.0 ycen 0.0 zcen 0.0 select segid AMM .or. segid MECL end')
stream.charmm_script('image byres xcen 0.0 ycen 0.0 zcen 0.0 select segid SOLV end')

# Electrostatics 
# Note that charmm seems to have issues with more than 7 keywords
# use update: True to set all options
# problem seemed to be fixed by recompiling in gnu enviorment without mkl
dict_nbonds = {'elec': True,
        'group': True,
        'switch': True,
        'cdie': True,
        'eps': 1.0,
        'ewald': True,
        'KAPPA': 0.34,
        'spline': True,
        'PMEWald': True,
        'ORDEr': 6,
        'FFTX': 40,
        'FFTY': 40,
        'FFTZ': 40,
        'vdw': True,
        'vgroup': True,
        'vswitch': True,
        'cutnb': 14.0,
        'ctofnb': 13.0,
        'ctonnb': 12.0,
        'inbfrq': 25,
        'cutim': 14.0,
        'imgfrq': 25, 
        'wmin': 0.5,
        }
nbond_ewald = pycharmm.NonBondedScript(**dict_nbonds)
nbond_ewald.run()

# Initiate MNDO. No pycharmm module presently availible 
stream.charmm_script('define qm sele segid AMM .or. segid MECL end')
stream.charmm_script('mndo remo sele qm end GLNK sele none end sele none end am1 char 0 switch')

# Set up trajectory file
stream.charmm_script(f'open write unit 22 file name men_resd_am1.dcd')
stream.charmm_script(f'traj iwrite 22 nwrite 1 nfile 44 skip 1\n')

# switch shake ... excluding qm atoms
# No pycharmm module for shake presently availible 
stream.charmm_script('shake bonh para tol 1.0e-6 sele all end  sele  (.not. qm .and. hydrogen) end')

# fix a smaller region
stream.charmm_script('cons fix sele .not. ( .byres. ( point 0.0 0.0 0.0 cut 10.0 ) ) end')

# Causes error, must adress, will use charmm_script for the time being
#ic.edit_angle(1,'NZ',2,'C1',2,'CL',180.0)
ic_angle_script = "ic edit\nangle ByNum 1 ByNum 5 ByNum 9 180.0\nend\ncons ic angle 500.0"
stream.charmm_script(ic_angle_script)

# Set Restraints
stream.charmm_script('set atom1 MECL 1 CL')
stream.charmm_script('set atom2 MECL 1 C1')
stream.charmm_script('set atom3 AMM  1 NZ')

# ------------
# Torsion scan
#-------------
rc = []
sqm_energy = []

# Open data file for resd scan and energy
r1 = -2.2
for i in range(0, 44, 1):
    stream.charmm_script('skip none')
    stream.charmm_script('RESDistance  RESET')
    stream.charmm_script(f'RESDistance  KVAL 2000.0  RVAL {r1} 1.0   @atom1  @atom2  -1.0  @atom2  @atom3')
    stream.charmm_script('Mini abnr nsteps 500 tolgrd 0.2 nprint 500')
    stream.charmm_script('print resdistance')
    rc.append(r1)

    stream.charmm_script('skip resd')
    energy.show()
    sqm_energy.append(stream.get_energy_value('ENER'))
    stream.charmm_script('traj write') 

    r1 += 0.1

# print out results
for i in range(len(sqm_energy)):
    print(rc[i], sqm_energy[i])

# -----------
# Plot figure
#------------
rc = np.array(rc)
sqm_energy -= np.min(sqm_energy[0:22])

# plot
fig, ax = plt.subplots()
ax.plot(rc, sqm_energy, 'b')
ax.set(xlabel='r(C-Cl) - r(N-C) (A)', ylabel='Energy (kcal/mol)',
       title='Reaction profile in Menshutkin reaction at AM1/MM')

# turn off grid
ax.grid(False)

# set tick marks inside, and make labels further from axis
ax.tick_params(axis="x",direction="in", pad=7.5)
ax.tick_params(axis="y",direction="in", pad=7.5)
fig.savefig("men_resd_am1.png")
plt.show()

exit()



