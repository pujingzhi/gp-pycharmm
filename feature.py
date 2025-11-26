import math
import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as Tensor
from dims import nsolu, n_atm_types

ddtype = tf.float64

@tf.function
def pairwise_vector(coords: Tensor) -> Tensor:
    num_batches, num_channels, _ = coords.shape #tf.shape(coords)
    rij = coords[:, :nsolu, None] - coords[:, None]
    mask = ~tf.eye(nsolu, num_channels, dtype=tf.bool) # remove self-interaction
    mask = tf.repeat(tf.expand_dims(mask,0), rij.shape[0], 0)
    rij = tf.reshape(tf.boolean_mask(rij, mask),(num_batches, nsolu, num_channels - 1, 3))
    return rij

@tf.function
def symmetry_function_g2(rij: Tensor, Rcr: float, EtaR: Tensor, ShfR: Tensor) -> Tensor:
    dij = tf.norm(rij, ord='euclidean', axis=3)
    fij = tf.where(dij < Rcr, (tf.cos(dij / Rcr * math.pi) + 1) * 0.5, tf.zeros_like(dij))
    g2 = tf.reduce_sum(tf.math.exp(tf.expand_dims(-EtaR,axis=2) * (tf.expand_dims(dij,axis=-1) - tf.expand_dims(ShfR,axis=2))**2) * tf.expand_dims(fij,axis=-1), axis=2)
    return g2 

@tf.function
def symmetry_function_g3(rij: Tensor, Rca: float, Zeta: Tensor, EtaA: Tensor) -> Tensor:
    x = tf.range(tf.shape(rij)[1])
    c = tf.squeeze(tf_combos(x))
    r12 = tf.gather(rij, indices=c[:,0], axis=2)
    r13 = tf.gather(rij, indices=c[:,1], axis=2)

    r23 = r12 - r13
    d12 = tf.norm(r12,ord='euclidean',axis=3)
    d13 = tf.norm(r13,ord='euclidean',axis=3)
    d23 = tf.norm(r23,ord='euclidean',axis=3)
    f12 = (tf.math.cos(d12 / Rca * math.pi) + 1) * 0.5
    f13 = (tf.math.cos(d13 / Rca * math.pi) + 1) * 0.5
    f23 = (tf.math.cos(d23 / Rca * math.pi) + 1) * 0.5
    f12 = tf.where(d12 < Rca, f12, tf.zeros_like(f12))
    f13 = tf.where(d13 < Rca, f13, tf.zeros_like(f13))
    f23 = tf.where(d23 < Rca, f23, tf.zeros_like(f23))
    cosine = tf.einsum('ijkl,ijkl->ijk', r12, r13) / (d12 * d13)
    g3 = tf.reduce_sum(2**(1 - tf.expand_dims(Zeta,axis=2)) * (1 + tf.expand_dims(cosine,axis=-1))**tf.expand_dims(Zeta,axis=2) * tf.math.exp(tf.expand_dims(-EtaA,axis=2) * tf.expand_dims(d12**2 + d13**2 + d23**2,axis=-1)) * tf.expand_dims(f12 * f13 * f23,axis=-1), axis=2)
    return g3

@tf.function
def symmetry_function_g3ani(rij: Tensor, Rca: float, Zeta: Tensor, ShfZ: Tensor, EtaA: Tensor, ShfA: Tensor) -> Tensor:
    x = tf.range(nsolu-1) #tf.shape(rij)[2])
    #val = 0.5*(-1*nsolu**2 + 2*(natm-1)*nsolu - nsolu)
    c = tf_combos(x)#[:int(val)]
    #print(rij.shape)
    rij = tf.gather(rij, c, axis=2)
    #print(rij.shape)
    r12 = rij[:,:,:,0]
    r13 = rij[:,:,:,1]
    #print(tf.math.reduce_any(rij == 0.0))
    #print(tf.math.reduce_any(r12 == 0.0))
    #print(tf.math.reduce_any(r13 == 0.0))
    #print(r12.shape, r13.shape)
    #r12 = tf.gather(rij, indices=c[:,0], axis=2)
    #r13 = tf.gather(rij, indices=c[:,1], axis=2)
    r23 = r12 - r13
    d12 = tf.norm(r12,ord='euclidean',axis=3)
    d13 = tf.norm(r13,ord='euclidean',axis=3)
    #print(np.argwhere(d12.numpy() == 0))
    #print(np.argwhere(d13.numpy() == 0))

    f12 = (tf.math.cos(d12 / Rca * math.pi) + 1) * 0.5
    f13 = (tf.math.cos(d13 / Rca * math.pi) + 1) * 0.5
    f12 = tf.where(d12 < Rca, f12, tf.zeros_like(f12))
    f13 = tf.where(d13 < Rca, f13, tf.zeros_like(f13))
    cosine = tf.einsum('ijkl,ijkl->ijk', r12, r13) / (d12 * d13)
    cosine = tf.math.cos(tf.expand_dims(tf.math.acos(cosine),axis=-1) - tf.expand_dims(ShfA,axis=2))
    g3 = tf.reduce_sum(2**(1 - tf.expand_dims(Zeta,axis=2)) * (1 + cosine)**tf.expand_dims(Zeta,axis=2) * tf.math.exp(tf.expand_dims(-EtaA,axis=2) * (0.5 * tf.expand_dims(d12 + d13, axis=-1) - tf.expand_dims(ShfZ,axis=2))**2) * tf.expand_dims(f12 * f13, axis=-1), axis=2)
    return g3

def numpy_pairwise_combinations(x):
    x = np.array(x)
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]

def tf_combos(inputs):
  combos = tf.py_function(numpy_pairwise_combinations, [inputs], Tout=tf.int64)
  return combos

class Feature(tf.keras.layers.Layer):
    def __init__(self, Rcr: float, EtaR: Tensor, ShfR: Tensor, Rca: float, Zeta: Tensor, EtaA: Tensor) -> None:
        super().__init__(trainable=False, dtype=ddtype)
        assert len(EtaR) == len(ShfR)
        assert len(Zeta) == len(EtaA)
        self.Rcr = Rcr
        self.Rca = Rca
        self.EtaR = tf.convert_to_tensor(EtaR, dtype=ddtype)
        self.ShfR = tf.convert_to_tensor(ShfR, dtype=ddtype)
        self.Zeta = tf.convert_to_tensor(Zeta, dtype=ddtype)
        self.EtaA = tf.convert_to_tensor(EtaA, dtype=ddtype)

    def call(self, coords: Tensor, atom_types: Tensor, norm=True) -> Tensor:
        num_batches, num_channels, _ = tf.shape(coords)
        rij = pairwise_vector(coords)
        EtaR = tf.gather(self.EtaR,indices=atom_types)
        ShfR = tf.gather(self.ShfR,indices=atom_types)
        Zeta = tf.gather(self.Zeta,indices=atom_types)
        EtaA = tf.gather(self.EtaA,indices=atom_types)
        g2 = symmetry_function_g2(rij, self.Rcr, EtaR, ShfR)
        g3 = symmetry_function_g3(rij, self.Rca, Zeta, EtaA)
        print(g2.shape)
        if norm:
            g2 = g2 / tf.norm(g2, axis=-1, keepdims=True)
            g3n = tf.norm(g3, axis=-1, keepdims=True)
            g3one = tf.ones_like(g3n)
            g3n = tf.where(g3n==0.0, g3one, g3n)
            g3 = g3 / g3n
        return tf.concat((g2, g3), axis=2)

    @property
    def output_length(self) -> int:
        return len(self.EtaR[0]) + len(self.EtaA[0])

class FeatureANI(tf.keras.layers.Layer):
    def __init__(self, Rcr: float, EtaR: Tensor, ShfR: Tensor, Rca: float, Zeta: Tensor, ShfZ: Tensor, EtaA: Tensor, ShfA: Tensor) -> None:
        super().__init__(trainable=False, dtype=ddtype)
        assert len(EtaR) == len(ShfR)
        assert len(Zeta) == len(ShfZ) == len(EtaA) == len(ShfA)
        self.Rcr = Rcr
        self.Rca = Rca
        self.EtaR = tf.convert_to_tensor(EtaR, dtype=ddtype)
        self.ShfR = tf.convert_to_tensor(ShfR, dtype=ddtype)
        self.Zeta = tf.convert_to_tensor(Zeta, dtype=ddtype)
        self.ShfZ = tf.convert_to_tensor(ShfZ, dtype=ddtype)
        self.EtaA = tf.convert_to_tensor(EtaA, dtype=ddtype)
        self.ShfA = tf.convert_to_tensor(ShfA, dtype=ddtype)

    ##@tf.function
    def call(self, coords: Tensor, atom_types, norm=True)  -> Tensor:
        num_batches, num_channels, _ = coords.shape[0],  coords.shape[1],  coords.shape[2] #tf.shape(coords)
        rij = pairwise_vector(coords)
        EtaR = tf.gather(self.EtaR, indices=atom_types)
        ShfR = tf.gather(self.ShfR, indices=atom_types[:,:nsolu])
        Zeta = tf.gather(self.Zeta, indices=atom_types[:,:nsolu])
        ShfZ = tf.gather(self.ShfZ, indices=atom_types[:,:nsolu])
        EtaA = tf.gather(self.EtaA, indices=atom_types[:,:nsolu])
        ShfA = tf.gather(self.ShfA, indices=atom_types[:,:nsolu])
        g2 = symmetry_function_g2(rij, self.Rcr, EtaR, ShfR)
        g3 = symmetry_function_g3ani(rij, self.Rca, Zeta, ShfZ, EtaA, ShfA)
        if norm: 
            g2 = g2 / tf.norm(g2, axis=-1, keepdims=True)
            g3n = tf.norm(g3, axis=-1, keepdims=True)
            g3one = tf.ones_like(g3n)
            g3n = tf.where(g3n==0.0, g3one, g3n)
            g3 = g3 / g3n
        return tf.concat((g2, g3), axis=2)

    @property
    def output_length(self) -> int:
        return len(self.EtaR[0]) + len(self.EtaA[0])

def get_descriptor(ani=True, n_atm_types=n_atm_types):
    ani = True
    if ani:
        # From ANI-2x_8x
        Rcr = 5.1000e+00
        Rca = 3.5000e+00
        EtaR = [1.9700000e+01]
        ShfR = [8.0000000e-01,1.0687500e+00,1.3375000e+00,1.6062500e+00,1.8750000e+00,2.1437500e+00,2.4125000e+00,2.6812500e+00,2.9500000e+00,3.2187500e+00,3.4875000e+00,3.7562500e+00,4.0250000e+00,4.2937500e+00,4.5625000e+00,4.8312500e+00]
        Zeta = [1.4100000e+01]
        ShfZ = [3.9269908e-01,1.1780972e+00,1.9634954e+00,2.7488936e+00]
        EtaA = [1.2500000e+01]
        ShfA = [8.0000000e-01,1.1375000e+00,1.4750000e+00,1.8125000e+00,2.1500000e+00,2.4875000e+00,2.8250000e+00,3.1625000e+00]
        EtaR, ShfR = np.array(np.meshgrid(EtaR, ShfR)).reshape(2, -1)
        Zeta, ShfZ, EtaA, ShfA = np.array(np.meshgrid(Zeta, ShfZ, EtaA, ShfA)).reshape(4, -1)
        EtaR = np.repeat([EtaR], n_atm_types, axis=0)
        ShfR = np.repeat([ShfR], n_atm_types, axis=0)
        Zeta = np.repeat([Zeta], n_atm_types, axis=0)
        ShfZ = np.repeat([ShfZ], n_atm_types, axis=0)
        EtaA = np.repeat([EtaA], n_atm_types, axis=0)
        ShfA = np.repeat([ShfA], n_atm_types, axis=0)
        descriptor = FeatureANI(Rcr, EtaR, ShfR, Rca, Zeta, ShfZ, EtaA, ShfA)
    else:
        Rcr = 6.0
        Rca = 6.0
        ShfR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]*n_atm_types
        EtaR = [0.0, 0.04, 0.14, 0.32, 0.71, 1.79]*n_atm_types
        Zeta = [1, 2, 4, 8, 16, 32]*n_atm_types
        EtaA = [0.0, 0.04, 0.14, 0.32, 0.71, 1.79]*n_atm_types
        descriptor = Feature(Rcr, EtaR, ShfR, Rca, Zeta, EtaA)
    return descriptor

