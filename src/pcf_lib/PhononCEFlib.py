import numpy as np
import yaml
import PyCrystalField.PyCrystalField as cef
from copy import deepcopy
from scipy import optimize

################## My own prefactor starting from S.H.O:
THzTomeV = 4.1357
kB = 8.617333262e-2 #meV/K
hbar = 6.582119569e-13 # meV*s
JouleTomeV = 6.241509e21 # meV/J
amuToKg = 1.660539e-27 # kg/amu
aaperm = 1e10 # \AA /m 

prefactor = hbar**2 / (2* amuToKg * JouleTomeV) *(aaperm**2)
# print('prefactor:', np.sqrt(prefactor))


#####################################################

a_z = np.array([[1,0],[0,-1]])/2
a_mu = np.array([[0,1],[0,0]])
a_mu_dag = np.array([[0,0],[1,0]])
Hphon_small = np.dot(a_mu_dag, a_mu) 

III = np.identity(2)

##################################



class PhononCEF_Operators:
    def __init__(self, CEF_J):

        ####### Define CEF operators
        self.CEFJx = np.kron(cef.Operator.Jx(CEF_J).O.real, III)
        self.CEFJy = np.kron(cef.Operator.Jy(CEF_J).O.real, III)
        self.CEFJz = np.kron(cef.Operator.Jz(CEF_J).O.real, III)
        self.CEFJplus  = self.CEFJx + 1j* self.CEFJy
        self.CEFJminus = self.CEFJx - 1j* self.CEFJy

        matrixsize = int(2*CEF_J + 1)
        ## Define phonon operators
        self.PHOJp1 = np.kron(np.identity(matrixsize), a_mu)
        self.PHOJm1 = np.kron(np.identity(matrixsize), a_mu_dag)
        self.PHOJz =  np.kron(np.identity(matrixsize), a_z)
        
    def CEFtransition(self, ket1, ket2):
        ax = np.dot(ket1, np.dot(self.CEFJx, ket2))**2
        ay = np.dot(ket1, np.dot(self.CEFJy, ket2))**2
        az = np.dot(ket1, np.dot(self.CEFJz, ket2))**2
        return np.real(ax + ay + az)

    def PHOtransition(self, ket1, ket2):
        ap1 = np.dot(ket1, np.dot(self.PHOJp1, ket2))**2
        am1 = np.dot(ket1, np.dot(self.PHOJm1, ket2))**2
        return np.real(ap1 + am1)


########################################################    

class CEF_Phon_obj:
    def __init__(self, cefobj, phOO):
        self.cefobj = cefobj
        self.H_cef = np.kron(self.cefobj.H, III)
        self.H_phons = np.kron(np.identity(len(cefobj.H)),  Hphon_small)
        self.H_int = np.kron(phOO,  a_mu_dag + a_mu)

        self.PC_op = PhononCEF_Operators(cefobj.J)


    def diagonalize(self, phononE, coupling):
        Ev1, Eve1 = np.linalg.eigh(self.H_cef + self.H_phons*phononE + self.H_int*coupling)

        self.eigenvalues = Ev1 - np.min(Ev1)
        self.eigenvectors = Eve1.T

    def newCEFCoeff(self, coeff):
        self.cefobj.newCoeff(coeff)
        self.H_cef = np.kron(self.cefobj.H, III)

    def normalizedCEFNeutronSpectrum(self, Earray, ResFunc, gamma = 0, Temp = 1e-4):
        '''neutron spectrum without the ki/Kf correction'''

        eigenkets = self.eigenvectors.real
        intensity = np.zeros(len(Earray))

        # make angular momentum ket object
        eigenkets =  self.eigenvectors

        # for population factor weights
        beta = 1/(8.61733e-2*Temp)  # Boltzmann constant is in meV/K
        Z = sum([np.exp(-beta*en) for en in self.eigenvalues])

        for i, ket_i in enumerate(eigenkets):
            # compute population factor
            pn = np.exp(-beta *self.eigenvalues[i])/Z
            if pn > 1e-3:  #only compute for transitions with enough weight
                for j, ket_j in enumerate(eigenkets):
                    # compute amplitude
                    mJn = self.PC_op.CEFtransition(ket_i,ket_j)
                    deltaE = self.eigenvalues[j] - self.eigenvalues[i]
                    GausWidth = ResFunc(deltaE)  #peak width due to instrument resolution
                    intensity += ((pn * mJn * self.cefobj._voigt(x=Earray, x0=deltaE, alpha=GausWidth, 
                                                        gamma=gamma)).real).astype('float64')
        return intensity

    def PhononNeutronSpectrum(self, Earray, ResFunc, gamma = 0):
        '''Assumes zero temperature for now'''

        eigenkets = self.eigenvectors.real
        intensity = np.zeros(len(Earray))

        # make angular momentum ket object
        eigenkets =  self.eigenvectors

        ## For now we assume T=0 
        ev0 = eigenkets[0]
        ev0b = eigenkets[1]

        for j, ev in enumerate(self.eigenvalues):
            # compute population factor
            ev1 = eigenkets[j]
            intensPH = self.PC_op.PHOtransition(ev0, ev1) + self.PC_op.PHOtransition(ev0b, ev1)
        
            GausWidth = ResFunc(ev)  #peak width due to instrument resolution
            intensity += intensPH*(self.cefobj._voigt(x=Earray, x0=ev, alpha=GausWidth, 
                                                gamma=gamma).real).astype('float64')
        return intensity



    def fitdata(self, chisqfunc, fitargs, method='Powell', **kwargs):
        '''fits data to CEF parameters'''

        initialChisq = chisqfunc(self, **kwargs)
        print('Initial err=', initialChisq, '\n')

        # define parameters
        # if len(self.cefobj.B) != len(kwargs['coeff']):
        #     raise ValueError('coeff needs to have the same length as self.B')

        # Define function to be fit
        fun, p0, resfunc = cef.makeFitFunction(chisqfunc, fitargs, **dict(kwargs, CFPH_Object=self) )


        ############## Fit, using error function  #####################
        p_best = optimize.minimize(fun, p0, method=method)
        ###############################################################

        print(fun(p_best.x))
        print(chisqfunc(self, **kwargs))
        finalChisq = fun(p_best.x)
        print('\rInitial err =', initialChisq, '\tFinal err =', finalChisq)
        
        result = resfunc(p_best.x)
        #print '\nFinal values: ', result
        result['Chisq'] = finalChisq
        return result



################################################# 
### Functions for computing the full neutron spectrum from operators instead of a DFT model

def normalizedCEFNeutronSpectrum(eigenvalues, eigenkets, Earray, Temp, alpha, gamma = 0):
    intensity = np.zeros(len(Earray))
    # for population factor weights
    beta = 1/(8.61733e-2*Temp)  # Boltzmann constant is in meV/K
    Z = sum([np.exp(-beta*en) for en in eigenvalues])
    for i, ket_i in enumerate(eigenkets):
        # compute population factor
        pn = np.exp(-beta *eigenvalues[i])/Z
        if pn > 1e-3:  #only compute for transitions with enough weight
            for j, ket_j in enumerate(eigenkets):
                # compute amplitude
                mJn = PC_op.CEFtransition(ket_i,ket_j)
                deltaE = eigenvalues[j] - eigenvalues[i]
                intensity += ((pn * mJn * voigt(x=Earray, x0=deltaE, alpha=alpha, 
                                                    gamma=gamma)).real).astype('float64')
    return intensity




######################################################


shifts = np.array([[i,j,k] for i in range(-1,2) for j in range(-1,2) for k in range(-1,2)])


class DFT_phonon_CEF:
    def __init__(self, dftoutputfile, cefLigand):

        try: # check if it's already imported
            p = dftoutputfile['points']
            dftoutput = dftoutputfile
        except TypeError: # Otherwise import
            with open(dftoutputfile, 'r') as ofile:
                dftoutput = yaml.safe_load(ofile)
        self.dftoutput = dftoutput 

        ## Number of phonons
        self.nphons = len(dftoutput['phonon'][0]['band'])
        print('There are', self.nphons, 'phonon modes')
        ## Number of wavevectors
        self.nQ = len(dftoutput['phonon'])
        print(' and', self.nQ, 'wavevectors')
        self.QQ = np.array([dftoutput['phonon'][n]['q-position'] for n in range(self.nQ)])

        self.Lig = cefLigand
        self.originalbonds = deepcopy(self.Lig.bonds)
        self.natom = dftoutput['natom']

        ## Step 1: match the bonds to the atoms
        latticmatrix = np.array(dftoutput['lattice']).T  ## Doesn't work if the unit cells don't match

        # latticmatrix = latM2
        self.latM = np.dot(self.Lig.LatticeTransformM, latticmatrix) ## transform from ABC space to bond cartesian space
        # self.CartRotMat = np.dot(self.latM, np.linalg.inv(np.array(dftoutput['lattice']).T))  ## Rotate from one cartesian
        #                                                                           ## axis to another
        self.CartRotMat = self.Lig.LatticeTransformM 
        self.latMinv = np.linalg.inv(self.latM) ## transform from ABC space to cartesian space


        self.pointsABC = np.array([dftoutput['points'][ii]['coordinates'] 
                                    for ii in range(dftoutput['natom'])])
        # print([pp + shifts for pp in self.pointsABC])
        self.pointsABC = np.vstack([pp + shifts for pp in self.pointsABC])
        self.pointsCART = np.array([np.dot(self.latM, pp) 
                                    for pp in self.pointsABC])

        pointslist = np.repeat(np.arange(dftoutput['natom']),len(shifts))
        
        ## Match the central ion to a point vector
        centralIon_index = pointslist[np.all(np.around((self.pointsABC - self.Lig.CentralIonPos),5) == 0, axis=1)]
        self.centralIon_index = centralIon_index[0]
        print('central ion index:', self.centralIon_index, dftoutput['points'][self.centralIon_index])

        # print(self.pointsCART)
        # ## Loop through bonds and match bonds to pointsCART
        self.bond_indices = []
        self.bond_masses = []
        for i,b1 in enumerate(cefLigand.bonds):
            b_abc = np.dot(self.latMinv, b1)
            bbb = self.pointsABC - self.Lig.CentralIonPos
            dd = np.around((bbb - b_abc),6)

            index = pointslist[np.all((dd == 0) + (dd == 1) + (dd == -1), axis=1)]
            try:
                self.bond_indices.append(index[0])
            except IndexError as exc:
                exc.args += ('Probably the ligand positions are inconsistent between .cif and DFT files.',)
                raise IndexError("\033[91mLigand positions from .cif file and from phonon mesh.yaml files do not match!"+
                                 '\n\t Please adjust the .cif file so all atomic positions match the phonon output '+
                                 '\n\t to at least seven decimal places. \033[0m') from exc

            self.bond_masses.append(dftoutput['points'][index[0]]['mass'])

        ### Define the CEF operators used to compute things. 
        ionJ = cef.Jion[cefLigand.ion][-1]
        self.ops = PhononCEF_Operators(ionJ)

        ## Collate all energies to a single array
        self.mode_energies = np.array([[self.dftoutput['phonon'][wavevector]['band'][band]['frequency'] 
                              for wavevector in range(self.nQ)] for band in range(self.nphons)])* THzTomeV



    def computedistortion(self, wavevector, band, displacement=1.0):
        '''band comes from the DFT file: eg dft_eigenvectors['phonon'][0]['band'][3]'''
        newbonds = deepcopy(self.Lig.bonds)
        ph_band = self.dftoutput['phonon'][wavevector]['band'][band]
        bandevs = np.array(ph_band['eigenvector'])[:,:,0]   ## Discard the imaginary part

        ci_eev = bandevs[self.centralIon_index] # eigenvector of central ion displacement

        energy = ph_band['frequency']* THzTomeV # leave in frequency

        for i,b1 in enumerate(self.Lig.bonds):
            if energy > 1e-3:
                # amplitude = np.sqrt(prefactor/(self.bond_masses[i]* energy))
                amplitude = np.sqrt(prefactor/(self.bond_masses[i]* energy)) * (1+2)  # Multiply by 3 for a single phonon
                # amplitude = np.sqrt(self.natom*prefactor/(self.bond_masses[i]* energy))
                # print('amplitude:', amplitude)
            else: amplitude = 0
            eev = bandevs[self.bond_indices[i]]
            # newbonds[i] = b1 + displacement*(np.dot(self.latM, eev) - np.dot(self.latM, ci_eev) )  # Assumes ABC vectors
            newbonds[i] = b1 + amplitude*displacement*(np.dot(self.CartRotMat, eev) -\
                                              np.dot(self.CartRotMat, ci_eev))  # Assumes cartesian vectors
        return newbonds
    

    def PhononOperator(self, wavevector, band, displacement = 1.0,  
                       symequiv=None, LigandCharge=None, printB=False):
        ## See if a point charge model has already been built
        if (symequiv == None) and (LigandCharge == None):
            symequiv = self.Lig.symequiv
            LigandCharge = self.Lig.LigandCharge

        ## Compute the undistorted CEF Hamiltonian
        self.Lig.bonds = self.originalbonds
        CE_original = self.Lig.PointChargeModel(symequiv = symequiv, LigandCharge=LigandCharge, printB=False)
        self.CEFobj = CE_original
        ## Compute the distorted CEF Hamiltonian
        newbonds = self.computedistortion(wavevector, band, displacement)
        self.Lig.bonds = newbonds 
        CeEE = self.Lig.PointChargeModel(symequiv = symequiv, LigandCharge=LigandCharge, printB=printB)

        ## Find the mode energy
        energy = self.dftoutput['phonon'][wavevector]['band'][band]['frequency']* THzTomeV 

        return CeEE.H - CE_original.H, energy
        


    def DefineCEFPhon_Hamiltonian(self, phononOp, phononE, CEFobj = None):
        '''CEFobj can be something arbitrary, not necessarily the point
        charge value.'''
        if CEFobj == None:
            CEFobj = self.CEFobj

        self.H_cef = np.kron(CEFobj.H, III)
        H_phon1 = np.kron(np.identity(len(CEFobj.H)),  Hphon_small)

        ########### Define Hamiltonian
        H_int = np.kron(phononOp,  a_mu_dag + a_mu)
        H_phons = H_phon1*phononE

        self.Hfull = self.H_cef + H_phons + H_int

        ## Diagonalize
        Ev1, Eve1 = np.linalg.eigh(self.Hfull)
        self.Eval1 = Ev1 - np.min(Ev1)
        self.Evec1 = Eve1.T 



    #####################################################
    ###### Functions to compute neutron spectra  #########
        
    def Compute_CEF_NeutronSpectrum(self, hbaromega, Gwidth, Lwidth, Temp=0):
        if Temp < 1e-5:
            CEF_neutronIntensity_1 = np.zeros_like(hbaromega)
            ev0 = self.Evec1[0]
            ev0b = self.Evec1[1]
            for j,ev in enumerate(self.Eval1):
                ev1 = self.Evec1[j]
                intensCF = self.ops.CEFtransition(ev0, ev1) + self.ops.CEFtransition(ev0b, ev1)
                #intensPH = PHOtransition(ev0, ev1) + PHOtransition(ev0b, ev1)

                CEF_neutronIntensity_1 += self.CEFobj._voigt(hbaromega, ev, Gwidth, Lwidth)*intensCF 
            return CEF_neutronIntensity_1

        else:
            intensity = np.zeros_like(hbaromega)
            eigenvalues = self.Eval1
            eigenkets = self.Evec1

            # for population factor weights
            beta = 1/(8.61733e-2*Temp)  # Boltzmann constant is in meV/K
            Z = sum([np.exp(-beta*en) for en in eigenvalues])

            for i, ket_i in enumerate(eigenkets):
                # compute population factor
                pn = np.exp(-beta *eigenvalues[i])/Z
                if pn > 1e-3:  #only compute for transitions with enough weight
                    for j, ket_j in enumerate(eigenkets):
                        # compute amplitude
                        mJn = self.ops.CEFtransition(ket_i,ket_j)
                        deltaE = eigenvalues[j] - eigenvalues[i]
                        intensity += ((pn * mJn * self.CEFobj._voigt(x=Earray, x0=deltaE, alpha=Gwidth, 
                                                            gamma=Lwidth)).real).astype('float64')
            return intensity


    def Compute_PHO_NeutronSpectrum(self, hbaromega, Gwidth, Lwidth, Temp=0):
        PHO_neutronIntensity_1 = np.zeros_like(hbaromega)
        ## For now we assume T=0 
        ev0 = self.Evec1[0]
        ev0b = self.Evec1[1]
        for j,ev in enumerate(self.Eval1):
            ev1 = self.Evec1[j]
            # intensCF = CEFtransition(ev0, ev1) + CEFtransition(ev0b, ev1)
            intensPH = self.ops.PHOtransition(ev0, ev1) + self.ops.PHOtransition(ev0b, ev1)

            PHO_neutronIntensity_1 += self.CEFobj._voigt(hbaromega, ev, Gwidth, Lwidth)*intensPH
        return PHO_neutronIntensity_1
    

    def Compute_NeutronSpectrum(self, wavevector, band, displacement, hbaromega, Gwidth, Lwidth, Temp=0):
        '''Create Hamiltonian, diagonalize, and compute T=0 neutron spectrum.'''
        phoOO, phoEE = self.PhononOperator(wavevector, band, displacement=displacement)
        self.DefineCEFPhon_Hamiltonian(phoOO, phoEE)
        CEF_nspec = self.Compute_CEF_NeutronSpectrum(hbaromega, Gwidth, Lwidth, Temp)
        PHO_nspec = self.Compute_PHO_NeutronSpectrum(hbaromega, Gwidth, Lwidth, Temp)
        return CEF_nspec, PHO_nspec
    













#############################################

# Define custom colormap
from matplotlib.colors import ListedColormap

cpal10 = np.array([0.89411765, 0.10196078, 0.10980392, 1.])
cpal11 = np.array([0.21568627, 0.49411765, 0.72156863, 1.])

totcol = 256
newcolors0 = np.zeros((256,4))
newcolors1 = np.zeros((256,4)) 
for i in range(totcol):
    r = i/totcol
    white = np.array([1,1,1,r])
    newcolors0[i, :] = cpal10*white
    newcolors1[i, :] = cpal11*white
MyCmp0 = ListedColormap(newcolors0)
MyCmp1 = ListedColormap(newcolors1)



####################################################