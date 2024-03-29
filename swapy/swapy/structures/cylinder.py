"""A cylindrical dielectric-lined wakefield accelerator structure

The equations used to solve wakefields in this structure are based on
the treatment presented in:

K-Y. Ng, "Wake fields in a dielectric-lined waveguide," Phys. Rev. D 42 (1990)
https://doi.org/10.1103/PhysRevD.42.1819
"""

import numpy as np
from scipy import special as spec, optimize as opt

class Cylinder:
    """A cylindrical dielectric-lined wakefield accelerator structure"""

    def __init__(self, a, b, mu, epsilon):

        self.a = a
        self.b = b
        self.mu = mu
        self.epsilon = epsilon

    def _pm(self, xa, xb, m):
        """Computes the p_m value given in Ng, Eqn. 2.17"""

        pm = spec.jn(m, xa) * spec.yn(m, xb) - spec.yn(m, xa) * spec.jn(m, xb)

        return pm

    def _rm(self, xa, xb, m):
        """Computes the r_m value given in Ng, Eqn. 2.17"""

        rm = spec.jvp(m, xa)*spec.yn(m, xb) - spec.yvp(m, xa)*spec.jn(m, xb)

        return rm

    def _ppm(self, xa, xb, m):
        """Computes the p'_m value given in Ng, Eqn. 2.20"""

        ppm = spec.jn(m, xa) * spec.yvp(m, xb) - spec.yn(m, xa) * spec.jvp(m, xb)

        return ppm

    def _rpm(self, xa, xb, m):
        """Computes the r'_m value given in Ng, Eqn. 2.20"""

        rpm = spec.jvp(m, xa)*spec.yvp(m, xb) - spec.yvp(m, xa)*spec.jvp(m, xb)

        return rpm

    def dispersion(self, s, m):
        """Computes the analytic disperion relation D_m (Ng, Eqn. 4.12)
        """

        # Define coordinate variables
        xa = s * self.a
        xb = s * self.b

        # Compute p_m and p'_m
        pm = self._pm(xa, xb, m)
        ppm = self._ppm(xa, xb, m)

        # Compute dispersion relation for m==0 (Ng, Eq. 3.10)
        if m==0:
            Dm = xa * ppm + xa * xb * pm / (2. * self.epsilon)

        else:
            
            # Compute r_m and r'_m
            rm = self._rm(xa, xb, m)
            rpm = self._rpm(xa, xb, m)

            # Compute dispersion relation for m>0 (Ng, Eq. 4.12)
            A = xb**2 / (m + 1.0) - m * (self.mu * self.epsilon + 1.0)
            B = xb * (self.epsilon * ppm * rm + self.mu * rpm * pm)
            Dm = pm * rm * A + B

        return Dm
    
    def get_modes(self, m, n_roots, k_step=1.e-1, tol=1.e-6, delta=1.e-4):
        """Computes eigenfrequencies, eigenvectors, and modal Green's function amplitudes

        Args:
          - m: Primary mode order
          - n_roots: Number of mode roots to compute
          - k_step: Wavevector step used to traverse dispersion relation (default 1.e-1)
          - tol: Tolerance used for solving dispersion relation roots (default 1.e-3)
          - delta: Wavevector variation used for dispersion relation derivative (default 1.e-6)
        """

        # Initialize mode & wavevector
        root = 0
        k = k_step

        # Compute dispersion relation roots for each mode
        # Note: these are the reduced eigenfrequencies x_(m,lambda) in Ng
        Xml = np.zeros(n_roots)
        while root < n_roots:
            
            # Get dispersion values for this & next wavevector
            Dm = self.dispersion(k, m)
            Dm_next = self.dispersion(k + k_step, m)

            # Store mode if crossing occurs over wavevector step 
            if np.sign(Dm) == -np.sign(Dm_next):
                Xml[root] = opt.fsolve(self.dispersion, k, args=(m), xtol=tol)
                root += 1

            k += k_step

        # Compute d/dx D_m(x)|x=X_ml for each mode 
        dDdx = (
            self.dispersion(Xml + delta, m) - self.dispersion(Xml - delta, m)
        ) / (2. * delta * self.a)

        # Compute modal Green's function amplitudes & wavevectors
        Xml *= self.a
        xi = self.b / self.a
        Gml = Xml * self._pm(Xml, Xml*xi, m) / dDdx
        if m>0:
            Gml *= self._rm(Xml, Xml*xi, m)
        Kml = Xml / np.sqrt(self.epsilon*self.mu - 1.)

        return Gml, Kml, Xml

    def greens(self, m, r0, R, Z, n_roots, Gml=None, Kml=None):
        """Computes transverse and longitudinal Green's functions for the cylinder

        Args:
          - m: Radial mode index
          - r0: Array of source radii
          - R: Array of sampling radii
          - Z: Array of sampling longitudes
          - n_roots: Number of mode roots used in calculation
          - dk: Wavevector step-size used in calculation (default 0.1)
          
        Returns:
          - Gt: Transverse Green's function, G_t(r0, R, Z)
          - Gl: Longitudinal Green's function, G_l(r0, R, Z)
        """

        nS = len(r0) # Number of source radii
        nR = len(R) # Number of sampled radii
        nZ = len(Z) # Number of sampled longitudes

        #
        if (Gml is None) or (Kml is None):
            Gml, Kml, _ = self.get_modes(m, n_roots)

        # Compute Green's functions in the absence of radial dependence (m=0)
        # Panofsky-Wenzel: dEz/dr==0
        if m==0:

            # Compute longitudinal Green's function
            AL = -4. / (self.epsilon * self.a * self.b)
            RL = 1. + 0. * np.einsum('i,j->ij', r0, R)
            SL = np.cos(np.einsum('i,j->ij', Z, Kml)) * Gml
            GL =  AL * np.einsum('ij,kl->iljk', RL, SL)

            # Return trivial transverse Green's function
            GT = np.zeros((nS, n_roots, nR, nZ))
        
        # Compute Green's functions with radial dependence (m>0)
        else:

            # Compute longitudinal Green's function
            AL = 8./self.a**2
            RL = (np.einsum('i,j->ij', r0, R) / self.b**2)**m
            SL = np.cos(np.einsum('i,j->ij', Z, Kml)) * Gml
            GL = AL * np.einsum('ij,kl->iljk', RL, SL)

            # Compute transverse Green's function
            AT = 8./self.a**2 * m * np.sqrt(self.epsilon-1.)
            RT = np.einsum('i,j->ij', (r0/self.b)**m, (R/self.a)**(m-1))     
            ST = np.sin(np.einsum('i,j->ij', Z, Kml)) * Gml
            GT = AT * np.einsum('ij,kl->iljk', RT, ST)
        
        return GL, GT
