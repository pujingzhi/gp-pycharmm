import gpflow
import tensorflow as tf

from gpflow import Parameter
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import positive
from gpflow.utilities.ops import difference_matrix, square_distance
from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData

from scipy.spatial.distance import cdist

ddtype = tf.float64
gpflow.config.set_default_float(ddtype)
f64 = gpflow.utilities.to_default_float

class GPRwDO(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: RegressionData,
        crd_transform,
        atom_types, 
        kernel,
        grads, 
        Xin, 
        mean_function = None,
        noise_variance = None,
        likelihood = None,
    ):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)
        self.crd_transform = crd_transform
        self.atom_types = atom_types
        self.grads = grads
        self.Xin = Xin 
        self.likelihood.variance = gpflow.Parameter(1.0, transform=positive(lower=1e-6), dtype=ddtype, name='likelihood_variance')
        self.likelihood.der_variance = gpflow.Parameter(1.0, transform=positive(lower=1e-6), dtype=ddtype, name='der_likelihood_variance')

    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.log_marginal_likelihood()

    def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
        and I is the corresponding identity matrix.
        """
        k_diag = tf.linalg.diag_part(K)
        s_diag_obs = tf.fill(tf.shape(tf.shape(X)[0]), self.likelihood.variance)
        s_diag_der = tf.fill((tf.shape(k_diag)-tf.shape(X)[0]), self.likelihood.der_variance)
        s_diag = tf.concat((s_diag_obs, s_diag_der), 0)
        return tf.linalg.set_diag(K, k_diag + s_diag)

    def log_marginal_likelihood(self) -> tf.Tensor:
        crd, Y = self.data
        atom_types = tf.repeat(tf.expand_dims(self.atom_types, axis=0), tf.shape(crd)[0], axis=0)
        K = self.kernel.K(crd_transform=self.crd_transform, atom_types=atom_types, crd1=crd, grads=self.grads, Xin=self.Xin)
        num_data = tf.shape(Y)[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag_obs = tf.fill(tf.expand_dims(tf.shape(crd)[0],-1), self.likelihood.variance)
        s_diag_der = tf.fill((tf.shape(k_diag)-tf.shape(crd)[0]), self.likelihood.der_variance)
        s_diag = tf.concat((s_diag_obs, s_diag_der), 0)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(Y) 
        log_prob = gpflow.logdensities.multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)
    def predict_f(
        self, Xnew: gpflow.models.model.InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> gpflow.models.model.MeanAndVariance:
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(Y_data)
        kmm = self.kernel(crd_transform, X_data)
        knn = self.kernel(crd_transform, Xnew, full_cov=full_cov)
        kmn = tf.transpose(self.kernel.Ks(crd_transform, Xnew, X_data)) #X_data, Xnew)
        kmm_plus_s = self._add_noise_cov(kmm)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var

class K_ext_combination(gpflow.kernels.Kernel):
    def __init__(self, ka):
        self.ka = ka
        super().__init__()

    def K(self, crd_transform, atom_types, crd1, grads, Xin, crd2=None):
        k_sum = 0.0
        for kk in self.ka:
            k_sum += kk.K(crd_transform, atom_types, crd1, grads, Xin)
        return k_sum

    def Ks(self, crd_transform, crd1, crd2, grads, Xs, Xin):
        k_sum = 0.0
        for kk in self.ka:
            k_sum += kk.Ks(crd_transform, crd1, crd2, grads, Xs, Xin)
        return k_sum

    def Ks1(self, crd_transform, crd1, X2, grads, Xs):
        k_sum = 0.0
        for kk in self.ka:
            k_sum += kk.Ks1(crd_transform, crd1, X2, grads, Xs)
        return k_sum

    def K_diag(self, crd_transform, atom_types, crd1, grads, crd2=None):
        tf.linalg.diag_part(self.K(rd_transform, atom_types, crd1, grads, crd2=None))


@tf.function
def gather(x, ind, axis):
    return tf.gather(x + 0, ind, axis=axis)

class K_ext(gpflow.kernels.RBF):
    def __init__(self, nsolu, atom_types, active_dims):
        self.nsolu = nsolu
        self.atom_types =  atom_types
        self.active_dims = active_dims 
        super().__init__(active_dims=self.active_dims)
    
    ##@tf.function
    def K(self, crd_transform, atom_types, crd1, grads, Xin, crd2=None):
        # There must be an X 2 located in a different memory reference
        if crd2 is None:
            crd2 = tf.identity(crd1) #.copy()
        nconf1, natm1, ncc1 = crd1.shape
        nconf2, natm2, ncc2 = crd2.shape
        atom_types1 = tf.repeat(tf.expand_dims(self.atom_types, axis=0), tf.shape(crd1)[0], axis=0)
        atom_types2 = tf.repeat(tf.expand_dims(self.atom_types, axis=0), tf.shape(crd2)[0], axis=0)
        crd1_solu = tf.reshape(crd1[:,:self.nsolu], (nconf1,-1))
        crd2_solu = tf.reshape(crd2[:,:self.nsolu], (nconf2,-1))
        crd1_solv = tf.reshape(crd1[:,self.nsolu:], (nconf1,-1))
        crd2_solv = tf.reshape(crd2[:,self.nsolu:], (nconf2,-1))
        assert natm1 == natm2
        X = tf.gather(tf.reshape(Xin, (nconf1, -1)), self.active_dims, axis=-1)
        X2 = tf.gather(tf.reshape(Xin, (nconf2, -1)), self.active_dims, axis=-1)
        grads = tf.gather(grads, self.active_dims, axis=1)
        dxdc1 = tf.identity(grads)
        dxdc2 = tf.identity(grads)

        K00 = super().K(X,X2)
        d = tf.shape(X)[1] #X.shape[-1]

        X_ = X/self.lengthscales
        X2_ =X2/self.lengthscales

        outer = tf.expand_dims(X_, axis=1) - tf.expand_dims(X2_, axis=0)
        outer = tf.transpose(outer/self.lengthscales, perm=[0,2,1])

        # First grad block
        outer1 = tf.reshape(tf.transpose(tf.einsum("ijk,kjl->ikl", outer, dxdc2), [0,2,1]), (nconf1, nconf2*self.nsolu*ncc1))
        K01 = outer1 * tf.tile(K00,[1,self.nsolu*3])

        # Second grad block
        outer2 = tf.reshape(tf.transpose(tf.einsum("ijk,ijl->ikl", outer, dxdc1), [2,0,1]), (nconf1*self.nsolu*ncc1, nconf2))
        K10 = -outer2 * tf.tile(K00,[self.nsolu*3, 1])

        # Hessian block
        outer1_x = tf.reshape(outer, (nconf1, -1))
        outer2_x = tf.transpose(outer, perm=[2, 1, 0])
        outer2_x = tf.reshape(outer2_x, (nconf2, -1))
        outer2_x = tf.transpose(outer2_x, perm=[1,0])
        mat = tf.eye(int(d), dtype=ddtype) / tf.pow(self.lengthscales, 2)
        kp_eye = tf.linalg.LinearOperatorFullMatrix(mat)
        kp = tf.linalg.LinearOperatorKronecker([kp_eye, tf.linalg.LinearOperatorFullMatrix(tf.ones((nconf1, nconf2), dtype=ddtype))])

        outer3_x = tf.reshape(tf.repeat(tf.expand_dims(outer1_x, axis=0), d, axis=0),(nconf1*d, nconf2*d)) *  tf.reshape(tf.repeat(tf.expand_dims(outer2_x, axis=0), d, axis=1),(nconf1*d, nconf2*d))
        chain_rule = tf.expand_dims(kp.to_dense() - outer3_x, 0)

        d2Kdx1dx2 = tf.squeeze(chain_rule) * tf.tile(K00, [d,d]) #tf.repeat(tf.expand_dims(K00, axis=1), d, axis=1)

        d2Kdx1dx2 = tf.reshape(d2Kdx1dx2, (d, nconf1, d, nconf2))
        dxdc1_in = tf.transpose(dxdc1, (1,0,2)) #tf.reshape(dxdc1, (dxdc1.shape[0], dxdc1.shape[1], dxdc1.shape[2] * dxdc1.shape[3]))
        dxdc2_in = tf.transpose(dxdc2, (1,0,2)) #tf.reshape(dxdc2, (dxdc2.shape[0], dxdc2.shape[1], dxdc2.shape[2] * dxdc2.shape[3]))

        # use tensordot to perform tensor contractions
        d2Kdcc1dx2 = tf.einsum('ijkl,ijm->jklm', d2Kdx1dx2, dxdc1_in)
        d2Kdcc1dcc2 = tf.einsum('ijkl,jkm->iklm', d2Kdcc1dx2, dxdc2_in)
        hess = tf.transpose(d2Kdcc1dcc2, perm=[2, 0, 3, 1])
        K11 = tf.reshape(hess, (nconf1 * self.nsolu * ncc1, nconf2 * self.nsolu * ncc2))

        K0 = tf.concat((K00, K01), 1)
        K1 = tf.concat((K10, K11), 1)
        K = tf.concat((K0, K1), 0)

        return K

    def Ks(self, crd_transform, crd1, crd2, grads, Xs, Xin):
        nconf1, natm1, ncc1 = crd1.shape
        nconf2, natm2, ncc2 = crd2.shape
        crd1_solu = tf.reshape(crd1[:,:self.nsolu], (nconf1,-1)) # Reshape is for convenience in getting derivatives shapes
        crd2_solu = tf.reshape(crd2[:,:self.nsolu], (nconf2,-1))
        crd1_solv = tf.reshape(crd1[:,self.nsolu:], (nconf1,-1))
        crd2_solv = tf.reshape(crd2[:,self.nsolu:], (nconf2,-1))
        atom_types1 = tf.repeat(tf.expand_dims(self.atom_types, axis=0), len(crd1), axis=0)
        atom_types2 = tf.repeat(tf.expand_dims(self.atom_types, axis=0), len(crd2), axis=0)
        X = tf.gather(tf.reshape(Xs, (nconf1, -1)), self.active_dims, axis=-1)
        X2 = tf.gather(tf.reshape(Xin, (nconf2, -1)), self.active_dims, axis=-1)
        grads = tf.gather(grads, self.active_dims, axis=1)
        dxdc2 = grads

        K00 = super().K(X,X2)
        d = tf.shape(X)[1] #X.shape[-1]

        # Scale the inputs by the lengthscale (for stability)
        X_ = X/self.lengthscales
        X2_ = X2/self.lengthscales

        outer = tf.expand_dims(X_, axis=1) - tf.expand_dims(X2_, axis=0)
        outer = tf.transpose(outer, perm=[0,2,1])/self.lengthscales

        # First grad block
        outer1 = tf.reshape(tf.transpose(tf.einsum("ijk,kjl->ikl", outer, dxdc2), [0,2,1]), (nconf1, nconf2*self.nsolu*ncc1))
        K01 = outer1 * tf.tile(K00,[1,self.nsolu*3])
        K = tf.concat((K00, K01), 1)

        return K

    def Ks1(self, crd_transform, crd1, X2in, grads, Xs):
        nconf1, natm1, ncc1 = crd1.shape
        nconf2, natm2, d = X2in.shape
        crd1_solu = tf.reshape(crd1[:,:self.nsolu], (nconf1,-1)) # Reshape is for convenience in getting derivatives shapes
        crd1_solv = tf.reshape(crd1[:,self.nsolu:], (nconf1,-1))
        atom_types1 = tf.repeat(tf.expand_dims(self.atom_types, axis=0), len(crd1), axis=0)
        X = tf.gather(tf.reshape(Xs,(nconf1, -1)), self.active_dims, axis=-1)
        X2 = tf.gather(tf.reshape(X2in, (nconf2, -1)), self.active_dims, axis=-1)
        grads = tf.gather(grads, self.active_dims, axis=1)
        dxdc2 = grads

        K00 = super().K(X,X2)
        d = tf.shape(X)[1] #X.shape[-1]

        # Scale the inputs by the lengthscale (for stability)
        X_ = X/self.lengthscales
        X2_ = X2/self.lengthscales

        outer = tf.expand_dims(X_, axis=1) - tf.expand_dims(X2_, axis=0)
        outer = tf.transpose(outer, perm=[0,2,1])/self.lengthscales

        # First grad block
        outer1 = tf.reshape(tf.transpose(tf.einsum("ijk,kjl->ikl", outer, dxdc2), [0,2,1]), (nconf1, nconf2*self.nsolu*ncc1))
        K01 = outer1 * tf.tile(K00,[1,self.nsolu*3])
        K = tf.concat((K00, K01), 1)

        return K
                                                    
    def dKs(self, crd_transform, crd1, crd2):
        '''
        Not updated yet
        '''
        nconf1, natm1, ncc1 = crd1.shape
        nconf2, natm2, ncc2 = crd2.shape
        crd1 = tf.reshape(crd1, [nconf1, ncc1*natm1])
        crd2 = tf.reshape(crd2, [nconf2, ncc2*natm2])
        atom_types1 = tf.repeat(tf.expand_dims(self.atom_types, axis=0), len(crd1), axis=0)
        atom_types2 = tf.repeat(tf.expand_dims(self.atom_types, axis=0), len(crd2), axis=0)
        assert natm1 == natm2
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(crd2_solu)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(crd1_solu)
                crd1 = tf.concat((crd1_solu, crd1_solv), axis=1)
                crd2 = tf.concat((crd2_solu, crd2_solv), axis=1)
                X = crd_transform(tf.reshape(crd1, [nconf1, natm1, ncc1]), atom_types1)
                X2 = crd_transform(tf.reshape(crd2, [nconf2, natm2, ncc2]), atom_types2)
                K00 =  (1/self.nsolu**2)*tf.reduce_sum(super().K(X,X2), [1,3])
            dkdx = -1 * tape2.batch_jacobian(K00,crd1_solu) # dkdx methods equivalent
            dkdx_ = tape2.gradient(K00, crd1_solu)
        d2kdx2 = tf.transpose(tape.jacobian(dkdx_, crd2_solu), [1,0,3,2])
        K = tf.concat((tf.reshape(tf.transpose(dkdx, [2,0,1]), [nconf1*ncc1*self.nsolu,nconf2]), tf.reshape(d2kdx2, [nconf1*ncc1*self.nsolu, nconf2*ncc2*self.nsolu])), axis=1)
        return K

