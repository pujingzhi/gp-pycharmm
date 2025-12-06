# Import libraries 

import math
import numpy as np
import tensorflow as tf
import gpflow

from gpflow import Parameter
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import positive
from gpflow.utilities.ops import difference_matrix, square_distance
from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData

from tensorflow import convert_to_tensor as Tensor

from dims import nsolu, natm, n_atm_types, nconf, nskip, nfeat
from feature import get_descriptor, pairwise_vector

ddtype = tf.float64
gpflow.config.set_default_float(ddtype)
f64 = gpflow.utilities.to_default_float

import time
start = time.time()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

npz_sqm = np.load('sqm_data.npz')
coords = Tensor(npz_sqm['crd'], dtype=ddtype)
sqm_energy = Tensor(npz_sqm['sqm_energy'],dtype=ddtype)
sqm_force = Tensor(npz_sqm['sqm_force'],dtype=ddtype)

npz_aiqm = np.load('aiqm_data.npz')
aiqm_energy = Tensor(npz_aiqm['aiqm_energy'],dtype=ddtype)
aiqm_force = Tensor(npz_aiqm['aiqm_force'],dtype=ddtype)

sqm_energy -= tf.reduce_mean(sqm_energy)
aiqm_energy -= tf.reduce_mean(aiqm_energy)

energy = aiqm_energy - sqm_energy
gradient = aiqm_force - sqm_force

# Encode gradient infor as additional energies using finite displacement
# (note: this trick is only effective when energy fitting is highly accurate).
crds_t = []
diff_t = []
diff = energy
dx = 0.01
num_frames = tf.shape(coords)[0]
for i in tf.range(num_frames):
    crds_t.append(tf.identity(coords[i]))
    diff_t.append(tf.identity(diff[i]))
    for j in tf.range(nsolu):
        for k in tf.range(3):
            crd_a = tf.identity(coords[i])
            crd_a = tf.tensor_scatter_nd_add(
                crd_a,
                indices=[[j, k]],
                updates=[dx]
            )
            crds_t.append(crd_a)
            diff_a = diff[i] + gradient[i, j, k] * dx
            diff_t.append(diff_a)
coords = tf.stack(crds_t, axis=0)
diff = tf.stack(diff_t, axis=0)
energy = diff

atom_types = Tensor(np.array([1,0,0,0,2,0,0,0,3]))
atom_types_vec = tf.identity(atom_types)
atom_types = tf.repeat(tf.expand_dims(atom_types, axis=0), len(coords), axis=0)

coords_train = coords[::nskip]
energy_train = tf.expand_dims(energy[::nskip], -1)
grad_train = gradient[::nskip,:nsolu]

descriptor = get_descriptor(ani=True, n_atm_types=n_atm_types)

X = descriptor(coords[::nskip], atom_types[::nskip])
X = tf.reshape(X, [X.shape[0],-1])
Y = Tensor(np.atleast_2d(energy[::nskip]).T,dtype=ddtype)

# SSS Kernel
ka = []
num_kernels = nsolu
for i in range(num_kernels):
    ki = gpflow.kernels.RBF(active_dims=[j+(nfeat*i) for j in range(nfeat)])
    ka.append(ki)
kern =  gpflow.kernels.Sum(ka)

model = gpflow.models.GPR((X,Y), kern)

opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

# Make Predictions
kinv = tf.linalg.inv(model.kernel.K(X) + model.likelihood.variance * tf.eye(tf.shape(model.kernel.K(X))[0], dtype=ddtype))
alpha = tf.matmul(kinv,Y)

mfin = nsolu*3+1
crds = Tensor(coords)
crds = crds[::mfin]
for i in range(nconf):
    crds1 = crds[i*1:1*(i+1),:,:]
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(crds1)
        atom_types_s1 = tf.repeat(tf.expand_dims(atom_types_vec, axis=0), len(crds1), axis=0)
        Xs1 = descriptor(crds1, atom_types_s1)
        Xs1 = tf.reshape(Xs1, [Xs1.shape[0], Xs1.shape[1]*Xs1.shape[2]])
        ks = model.kernel.K(Xs1,X)
        preds_ = tf.matmul(ks, alpha)
    preds_f_ = tape.gradient(preds_, crds1)
    if i == 0:
        preds = preds_
        preds_f = preds_f_
    else:
        preds = tf.concat((preds,preds_),0)
        preds_f = tf.concat((preds_f,preds_f_),0)

print(tf.math.reduce_any(tf.math.is_nan(preds)))
print(tf.math.reduce_any(tf.math.is_nan(preds_f)))

# Read Forces 
force = gradient
energy = energy[::mfin]

print('Force RMSE', tf.math.sqrt(tf.reduce_mean(((force)-preds_f)**2)))
print('AI-SE Force RMSE', tf.math.sqrt(tf.reduce_mean((force)**2)))

print('Heavy Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force)-preds_f)**2, indices=[0,4,8], axis=1))))
print('AI-SE Heavy Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force)**2, indices=[0,4,8], axis=1))))

print('Solute Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force)-preds_f)**2, indices=[i for i in range(nsolu)], axis=1))))
print('AI-SE Solute Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force)**2, indices=[i for i in range(nsolu)], axis=1))))

print('Solvent Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force)-preds_f)**2, indices=[i for i in range(nsolu,natm)], axis=1))))
print('AI-SE Solvent Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force)**2, indices=[i for i in range(nsolu,natm)], axis=1))))

print('Training Data Force RMSE', tf.math.sqrt(tf.reduce_mean(((force[::nskip,:9])-preds_f[::nskip,:9,:])**2)))
print('Training Data AI-SE Force RMSE', tf.math.sqrt(tf.reduce_mean((force[::nskip,:9])**2)))

print('Energy RMSE', tf.math.sqrt(tf.reduce_mean((tf.squeeze(preds)-tf.squeeze(energy))**2)))
print('AI-SE Energy RMSE', tf.math.sqrt(tf.reduce_mean(energy**2)))

print('Training Data Energy RMSE', tf.math.sqrt(tf.reduce_mean((tf.squeeze(preds)[::nskip]-tf.squeeze(energy)[::nskip])**2)))
print('Training Data AI-SE Energy RMSE', tf.math.sqrt(tf.reduce_mean(tf.squeeze(energy)[::nskip]**2)))

params = gpflow.utilities.parameter_dict(model)
print(params)

print(model.trainable_variables)

# Use basinhopping algorithm to optimize hyper parameters
import scipy
from scipy.optimize import basinhopping

x0 = []
# Loop through all kernels in the list
for kk in model.kernel.kernels:
    x0.append(kk.lengthscales.numpy())
    x0.append(kk.variance.numpy())
# Add the likelihood parameter
x0.append(model.likelihood.variance.numpy())

class Global_Bounds:
    def __init__(self, xmax=[1000.0 for i in range(2*nsolu+1)], xmin=[0.01 for i in range(2*nsolu+1)]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

global_bounds = Global_Bounds()
local_bounds=scipy.optimize.Bounds(lb=0.01,ub=1000.0)

def optimize_with_basinhopping(model, initial_params, bounds, accept_test, n_iterations=100):
    @tf.function
    def objective_func_wrapper(params):

        # Loop through all kernels in the list
        j = 0
        for i in range(nsolu):
            model.kernel.kernels[i].lengthscales.assign(params[j])
            j += 1
            model.kernel.kernels[i].variance.assign(params[j])
            j += 1
        model.likelihood.variance.assign(params[j])

        with tf.GradientTape() as tape_jac:
            tape_jac.watch(model.trainable_variables)
            loss = model.training_loss()
        gradients = tape_jac.gradient(loss, model.trainable_variables)
        loss, gradients = tf.cast(loss, dtype=tf.float64), tf.cast(gradients, dtype=tf.float64)
        return loss, gradients #, hess

    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'jac': True, 'hess': False}

    def minimizer(params):
        loss, gradients = objective_func_wrapper(params)
        return loss.numpy().astype(np.float64), gradients.numpy().astype(np.float64) #, hess.numpy().astype(np.float64)

    result = basinhopping(objective_func_wrapper, initial_params, accept_test=accept_test, minimizer_kwargs=minimizer_kwargs, disp=True, niter=n_iterations)
    optimized_params = tf.convert_to_tensor(result.x)
    optimized_loss = tf.convert_to_tensor(result.fun)

    return optimized_params, optimized_loss

x_opt, loss = optimize_with_basinhopping(model, x0, local_bounds, Global_Bounds())
print(x_opt, loss)

# Loop through all kernels in the list
j = 0
for i in range(nsolu):
    model.kernel.kernels[i].lengthscales.assign(x_opt[j])
    j += 1
    model.kernel.kernels[i].variance.assign(x_opt[j])
    j += 1
model.likelihood.variance.assign(x_opt[j])

print("Trained1: ", gpflow.utilities.parameter_dict(model))
params = gpflow.utilities.parameter_dict(model)

import pickle 

with open('params_k_rbf_atom_f64.pkl', 'wb') as f:
    pickle.dump(params, f)
       
# Make Predictions
kinv = tf.linalg.inv(model.kernel.K(X) + model.likelihood.variance * tf.eye(tf.shape(model.kernel.K(X))[0], dtype=ddtype))
alpha = tf.matmul(kinv,Y)

for i in range(nconf):
    crds1 = crds[i*1:1*(i+1),:,:]
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(crds1)
        Xs1 = descriptor(crds1, atom_types_s1)
        Xs1 = tf.reshape(Xs1, [Xs1.shape[0], Xs1.shape[1]*Xs1.shape[2]])
        ks = model.kernel.K(Xs1,X)
        preds_ = tf.matmul(ks, alpha)
    preds_f_ = tape.gradient(preds_, crds1)
    if i == 0:
        preds = preds_
        preds_f = preds_f_
    else:
        preds = tf.concat((preds,preds_),0)
        preds_f = tf.concat((preds_f,preds_f_),0)

print(tf.math.reduce_any(tf.math.is_nan(preds)))
print(tf.math.reduce_any(tf.math.is_nan(preds_f)))

for i in range(nconf):
    if tf.math.reduce_any(tf.math.is_nan(preds_f[i,:,:])):
        for j in range(natm):
            for k in range(3):
                if tf.math.is_nan(preds_f[i,j,k]):
                    print(i,j,k)

np.savez_compressed('const_k_rbf', alpha=alpha.numpy(), X=X.numpy(), Y=Y.numpy())

print('Energy RMSE', tf.math.sqrt(tf.reduce_mean((tf.squeeze(preds)-tf.squeeze(energy))**2)))
print('AI-SE Energy RMSE', tf.math.sqrt(tf.reduce_mean(energy**2)))

print('Training Data Energy RMSE', tf.math.sqrt(tf.reduce_mean((tf.squeeze(preds)[::nskip]-tf.squeeze(energy)[::nskip])**2)))
print('Training Data AI-SE Energy RMSE', tf.math.sqrt(tf.reduce_mean(tf.squeeze(energy)[::nskip]**2)))

print('Solute Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force)-preds_f)**2, indices=[i for i in range(nsolu)], axis=1))))
print('AI-SE Solute Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force)**2, indices=[i for i in range(nsolu)], axis=1))))

print('Training Solute Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force[::nskip])-preds_f[::nskip])**2, indices=[i for i in range(nsolu)], axis=1))))
print('Training AI-SE Solute Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force[::nskip])**2, indices=[i for i in range(nsolu)], axis=1))))

print('Heavy Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force)-preds_f)**2, indices=[0,4,8], axis=1))))
print('AI-SE Heavy Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force)**2, indices=[0,4,8], axis=1))))

print('Training Heavy Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force[::nskip])-preds_f[::nskip])**2, indices=[0,4,8], axis=1))))
print('Training AI-SE Heavy Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force[::nskip])**2, indices=[0,4,8], axis=1))))

print('Solvent Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force)-preds_f)**2, indices=[i for i in range(nsolu,natm)], axis=1))))
print('AI-SE Solvent Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force)**2, indices=[i for i in range(nsolu,natm)], axis=1))))

print('Training Solvent Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather(((force[::nskip])-preds_f[::nskip])**2, indices=[i for i in range(nsolu,natm)], axis=1))))
print('Training AI-SE Solvent Force RMSE', tf.math.sqrt(tf.reduce_mean(tf.gather((force[::nskip])**2, indices=[i for i in range(nsolu,natm)], axis=1))))

force = force.numpy()
pred_f = preds_f.numpy()

coords = crds

# analyze solvent force
rij = pairwise_vector(coords)
dij = tf.norm(rij, ord='euclidean', axis=3)
dij = dij[:,:,nsolu-1:]
dij = tf.math.reduce_min(dij, axis=1, keepdims=False, name=None)
dij_np = dij.numpy()
x1, x2 = 0.0, 0.5
for i in range(24):
    temp = np.where((dij_np > x1) & (dij_np <= x2), 1.0, 0.0)
    args = np.argwhere((temp == 1.0))
    rmse = np.sqrt(np.average((force[args[:,0], args[:,1]+nsolu, :] - pred_f[args[:,0], args[:,1]+nsolu, :])**2))
    rmse2 = np.sqrt(np.average((force[args[:,0], args[:,1]+nsolu, :])**2))
    print(x2, rmse, rmse2)
    x1 += 0.5
    x2 += 0.5

temp = np.where(dij_np <= 5.2, 1.0, 0.0)
args = np.argwhere((temp == 1.0))
rmse = np.sqrt(np.average((force[args[:,0], args[:,1]+nsolu, :] - pred_f[args[:,0], args[:,1]+nsolu, :])**2))
rmse2 = np.sqrt(np.average((force[args[:,0], args[:,1]+nsolu, :])**2))
print('Solvent within cutoff: ', rmse, rmse2) 

end = time.time()
print("Elapsed time: ", end-start);

exit()


