# Deploy energy-only GPR to correct AM1/MM for RESD scan of Menshutkin reaction
#

#Import Modules
import os
import sys
sys.path.append("../")
import math
import numpy as np
import tensorflow as tf
import gpflow
import pickle

from gpflow import Parameter
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import positive
from gpflow.utilities.ops import difference_matrix, square_distance
from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData

from tensorflow import convert_to_tensor as Tensor

from dims import nsolu, natm, n_atm_types, nconf, nskip, nfeat
from feature import get_descriptor, pairwise_vector

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

#################
### GPR BLOCK ###
#################
ddtype = tf.float64
gpflow.config.set_default_float(ddtype)
f64 = gpflow.utilities.to_default_float

# Read in GPR Information
npz_sqm = np.load('sqm_data.npz')
coords = Tensor(npz_sqm['crd'], dtype=ddtype)
sqm_energy = Tensor(npz_sqm['sqm_energy'],dtype=ddtype)
sqm_force = Tensor(npz_sqm['sqm_force'],dtype=ddtype)

npz_aiqm = np.load('aiqm_data.npz')
aiqm_energy = Tensor(npz_aiqm['aiqm_energy'],dtype=ddtype)
aiqm_force = Tensor(npz_aiqm['aiqm_force'],dtype=ddtype)

sqm_energy -= tf.reduce_mean(sqm_energy)
aiqm_energy -= tf.reduce_mean(aiqm_energy)

energy_test = aiqm_energy - sqm_energy

gradient = aiqm_force - sqm_force

atom_types = Tensor(np.array([1,0,0,0,2,0,0,0,3]))
atom_types_vec = tf.identity(atom_types)
atom_types = tf.repeat(tf.expand_dims(atom_types, axis=0), len(coords), axis=0)

descriptor = get_descriptor(ani=True, n_atm_types=n_atm_types)

path = './'
npz = np.load(f'{path}/const_k_rbf.npz')
Y = Tensor(npz['Y'], dtype=ddtype)
X = Tensor(npz['X'], dtype=ddtype)
alpha = Tensor(npz['alpha'], dtype=ddtype)

# --- SSS kernel
ka = []
num_kernels = nsolu
for i in range(num_kernels):
    ki = gpflow.kernels.RBF(active_dims=[j+(nfeat*i) for j in range(nfeat)])
    ka.append(ki)
kern =  gpflow.kernels.Sum(ka)

model = gpflow.models.GPR((X,Y), kern)

with open('./params_k_rbf_atom_f64.pkl', 'rb') as f:
    params = pickle.load(f)

gpflow.utilities.multiple_assign(model, params)

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

energy.show()

# Set up PBC
crystal.define_cubic(length=40)
stream.charmm_script('crystal build Noper 0 cutoff 16.0')

stream.charmm_script('image byseg xcen 0.0 ycen 0.0 zcen 0.0 select segid AMM .or. segid MECL end')
stream.charmm_script('image byres xcen 0.0 ycen 0.0 zcen 0.0 select segid SOLV end')

# Electrostatics
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
stream.charmm_script(f'open write unit 22 file name men_deploy_gpr.dcd')
stream.charmm_script(f'traj iwrite 22 nwrite 1 nfile 44 skip 1\n')

# switch shake ... excluding qm atoms
stream.charmm_script('shake bonh para tol 1.0e-6 sele all end  sele  (.not. qm .and. hydrogen) end')

import ctypes
def e_gpr(natoms,
         x_pos, y_pos, z_pos,
         dx, dy, dz):
    x_pos1 = Tensor(x_pos[:natoms],dtype=ddtype)
    y_pos1 = Tensor(y_pos[:natoms],dtype=ddtype)
    z_pos1 = Tensor(z_pos[:natoms],dtype=ddtype)
    crds_ = tf.constant(tf.transpose(tf.stack([x_pos1, y_pos1, z_pos1])))

    rij = tf.expand_dims(crds_[:nsolu], 1) - tf.expand_dims(crds_[nsolu:], 0)
    dij = tf.norm(rij, ord='euclidean', axis=-1)
    args = list((np.argwhere(tf.math.reduce_min(dij,axis=0).numpy() < 6.0)+nsolu).flatten())
    args = [i for i in range(nsolu)] + args
    crds = tf.expand_dims(tf.gather(crds_, args, axis=0),0)

    with tf.GradientTape(
        persistent=False, watch_accessed_variables=False
    ) as tape:
        tape.watch(crds)
        atom_types_s = tf.repeat(tf.expand_dims(atom_types_vec, axis=0), len(crds), axis=0)
        Xs = descriptor(crds, atom_types_s)
        Xs = tf.reshape(Xs, [Xs.shape[0], Xs.shape[1]*Xs.shape[2]])
        ks = model.kernel.K(Xs,X)
        preds = tf.matmul(ks,alpha)
    dedcc = tf.squeeze(tape.gradient(preds, crds))
    dedcx1 = dedcc.numpy()[:,0].flatten()
    dedcy1 = dedcc.numpy()[:,1].flatten()
    dedcz1 = dedcc.numpy()[:,2].flatten()
    dedcx = dedcx1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dedcy = dedcy1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dedcz = dedcz1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    for i in range(len(args)):
        dx[args[i]] += dedcx[i]
        dy[args[i]] += dedcy[i]
        dz[args[i]] += dedcz[i]
    return preds

e_func = pycharmm.EnergyFunc(e_gpr)
energy.show()

# fix a smaller region
stream.charmm_script('cons fix sele .not. ( .byres. ( point 0.0 0.0 0.0 cut 10.0 ) ) end')

# additional co-linear restraint
ic_angle_script = "ic edit\nangle ByNum 1 ByNum 5 ByNum 9 180.0\nend\ncons ic angle 500.0"
stream.charmm_script(ic_angle_script)

# Set Restraints
stream.charmm_script('set atom1 MECL 1 CL')
stream.charmm_script('set atom2 MECL 1 C1')
stream.charmm_script('set atom3 AMM  1 NZ')

# -------
# RC scan
#--------
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

# -----------------
# Retrieve AM1 data
# -----------------
npz_sqm_0 = np.load('sqm_data.npz')
sqm_energy_0 = npz_sqm_0['sqm_energy']

# shift energies for plotting purpose
sqm_energy_0 -= np.min(sqm_energy_0[0:22])

import matplotlib.pyplot as plt

# -----------
# Plot figure
#------------
rc = np.array(rc)
sqm_energy -= np.min(sqm_energy[0:22])

# plot
fig, ax = plt.subplots()
ax.plot(rc, sqm_energy_0, 'r', label='AM1/MM')
ax.plot(rc, sqm_energy, 'b', label='AM1-GPR/MM')

ax.set(xlabel='r(C-Cl) - r(N-C) (A)', ylabel='Energy (kcal/mol)',
       title='Reaction profile in Menshutkin reaction')
ax.legend()

# turn off grid
ax.grid(False)

# set tick marks inside, and make labels further from axis
ax.tick_params(axis="x",direction="in", pad=7.5)
ax.tick_params(axis="y",direction="in", pad=7.5)
fig.savefig("men_deploy_gpr.png")
plt.show()

exit()

