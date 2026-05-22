import numpy as np
import copy
from sympy import var, solve
from scipy.optimize import fsolve
import CoolProp           # for AbstractState input-pair constants (CoolProp.DmassUmass_INPUTS, ...)
import CoolProp.CoolProp as CP

class BaseThermodynamicModel:

  ###  Attributes
  R_universal      = 8.31446261815324      # Universal gas constant [j/(mol k)]
  R_specific       = -1.0                  # Specific gas constant [J/(kg K)]
  molecular_weight = -1.0                  # Molecular weight [kg/mol]
  gamma            = -1.0                  # Ratio of heat capacities [-]

  #####

  ### Constructor
  def __init__(self):
    pass

  ### Methods
  def calculatePressureFromTemperatureDensity(self, T, rho):
    
    return 0.0


  def calculateTemperatureFromPressureDensity(self, P, rho):

    return 0.0
  

  def calculateTemperatureFromPressureDensityWithInitialGuess(self, T, P, rho):

    return 0.0

  
  def calculateInternalEnergyFromPressureTemperatureDensity(self, P, T, rho):

    return 0.0


  def calculateEntropyFromPressureTemperatureDensity(self, P, T, rho):

    return 0.0


  def calculatePressureTemperatureFromDensityInternalEnergy(self, P, T, rho, e):

    return 0.0


  def calculatePressureTemperatureEntropyFromDensityInternalEnergy(self, rho, e):
    # (P, T, s) from (rho, e). Default = the two existing calls; models with a single
    # flash (e.g. CoolProp AbstractState) override this to avoid the extra evaluation.
    P, T = self.calculatePressureTemperatureFromDensityInternalEnergy(-1.0, -1.0, rho, e)
    s = self.calculateEntropyFromPressureTemperatureDensity(P, T, rho)
    return P, T, s


  def calculateDensityInternalEnergyFromPressureTemperature(self, rho, e, P, T):

    return 0.0


  def calculateSpecificHeatCapacities(self, c_v, c_p, P, T, rho):

    return 0.0 


  def calculateHeatCapacitiesRatio(self, P, T, rho):

    return 0.0 


  def calculateSoundSpeed(self, P, T, rho):

    return 0.0


  def calculateVolumeExpansivity(self, T, bar_v):

    return 0.0


  def calculateIsothermalCompressibility(self, T, bar_v):

    return 0.0
 

  def calculateIsentropicCompressibility(self, P, T, bar_v):
  
    return 0.0


###################################################################
##################### IDEAL GAS ###################################
###################################################################

class IdealGasModel(BaseThermodynamicModel):

  ###  Attributes


  ### Constructor
  def __init__(self, R_specific, gamma):

    super(BaseThermodynamicModel,self).__init__()  
    self.R_specific       = R_specific
    self.gamma            = gamma    
    self.molecular_weight = self.R_universal/self.R_specific


  ### Methods
  def calculatePressureFromTemperatureDensity(self, T, rho):

    # Equation of state
    P = rho*self.R_specific*T

    return P


  def calculateTemperatureFromPressureDensity(self, P, rho):
    
    # Equation of state
    T = P/(self.R_specific*rho)

    return T
  

  def calculateTemperatureFromPressureDensityWithInitialGuess(self, T, P, rho):

    # Equation of state
    T = P/(self.R_specific*rho)

    return T


  def calculateInternalEnergyFromPressureTemperatureDensity(self, P, T, rho):

    # Specific heat at constant volume
    c_v = self.R_specific/(self.gamma - 1.0)

    # Specific internal energy 
    e = c_v*T

    return e


  def calculateEntropyFromPressureTemperatureDensity(self, P, T, rho):

    # Specific heat at constant volume
    c_v = self.R_specific/(self.gamma - 1.0)

    # Specific entropy 
    s = c_v*np.log( P/( rho**self.gamma ) )

    return s


  def calculatePressureTemperatureFromDensityInternalEnergy(self, P, T, rho, e):

    c_v = self.R_specific/(self.gamma - 1.0)
    P = e*rho*(self.gamma - 1.0 )
    T = e/c_v

    return P, T


  def calculateDensityInternalEnergyFromPressureTemperature( self, rho, e, P, T):

    c_v = self.R_specific/(self.gamma - 1.0)

    e   = c_v*T
    rho = P/(e*(self.gamma - 1.0))

    return rho, e 


  def calculateSpecificHeatCapacities(self, c_v, c_p, P, T, rho):
  
    c_v = self.R_specific/(self.gamma - 1.0)
    c_p = c_v*self.gamma

    return c_v, c_p


  def calculateHeatCapacitiesRatio(self, P, rho):

    return self.gamma


  def calculateSoundSpeed(self, P, T, rho):

    sos = np.sqrt(self.gamma*P/(rho))

    return sos


  def calculateVolumeExpansivity(self, T, bar_v):

    expansivity = 1.0/T

    return expansivity


  def calculateIsothermalCompressibility(self, T, bar_v):

    isothermal_compressibility = bar_v/(self.molecular_weight*self.R_specific*T)

    return isothermal_compressibility


  def calculateIsentropicCompressibility(self, P, T, bar_v):

    dP_dT_const_v = self.molecular_weight*self.R_specific/bar_v
    dP_dv_const_T = (-1.0)*self.molecular_weight*self.R_specific*T/(bar_v*bar_v)

    isothermal_compressibility = ( -1.0 )/(bar_v*dP_dv_const_T)
    expansivity                = ( -1.0 )*(dP_dT_const_v/(bar_v*dP_dv_const_T))

    c_v = self.R_specific/(self.gamma - 1.0 )
    c_p = c_v*self.gamma

    bar_c_p = self.molecular_weight*c_p

    isentropic_compressibility = (isothermal_compressibility - ((bar_v*T*expansivity**2.0)/bar_c_p))

    return isentropic_compressibility
   

###################################################################
##################### REAL GAS ####################################
###################################################################

class PengRobinsonModel(BaseThermodynamicModel):

  ### Atributes (Variables that don't change with temperature)
  acentric_factor               = -1.0   
  critical_temperature          = -1.0
  critical_pressure             = -1.0  
  critical_molar_volume         = -1.0
  NASA_coefficients             = (-1.0)*np.ones(15)
  eos_b                         = -1.0
  eos_ac                        = -1.0
  eos_kappa                     = -1.0
  P_ref_std                     = 1.0e5   # NASA-7 standard-state pressure [Pa] (1 bar)
  max_aitken_iter               = 1000
  aitken_relative_tolerance     = 1.0e-5
  ### Nonlinear solver parameters
  xtol   = 1.0e-10		    # Relative error between two consecutive iterates
  epsfcn = 1.0e-5			# Step length for the forward-difference approximation of the Jacobian
  factor = 1.0e-1			# Parameter determining the initial step bound. Should be in the interval [0.1:100]
  max_newton_iter = 100		# Max iterations for the 1-D temperature inversion (rho,e)->T
  newton_rtol     = 1.0e-10	# Relative tolerance for the 1-D temperature inversion


  ### Constructor
  def __init__(self, molecular_weight, acentric_factor, critical_temperature, critical_pressure, critical_molar_volume, NASA_coefficients):

    super(BaseThermodynamicModel,self).__init__()  
    self.molecular_weight           = molecular_weight
    self.R_specific                 = self.R_universal/molecular_weight     
    self.acentric_factor            = acentric_factor
    self.critical_temperature       = critical_temperature
    self.critical_pressure          = critical_pressure
    self.critical_molar_volume      = critical_molar_volume
    self.NASA_coefficients          = NASA_coefficients

    self.eos_b  = 0.077796*( self.R_universal*self.critical_temperature/self.critical_pressure )
    #print('eos_b: {}'.format(eos_b))

    self.eos_ac = 0.457236*(( self.R_universal*self.critical_temperature)**2.0/(self.critical_pressure))
    #print('eos_ac: {}'.format(eos_ac))

    if acentric_factor > 0.49:
      self.eos_kappa = 0.379642 + 1.48503*self.acentric_factor - 0.164423*self.acentric_factor**2.0 + 0.016666*self.acentric_factor**3.0 
    else:
      self.eos_kappa = 0.37464 + 1.54226*self.acentric_factor - 0.26992*self.acentric_factor**2.0


  ### Methods

  def n_solve( self, functions, variables, norm_factors ):

    func = lambda x : [ f(*x) for f in functions ]
    variables = fsolve( func, variables, xtol = self.xtol, epsfcn = self.epsfcn, factor = self.factor, diag = norm_factors )

    return variables


  def calculateTemperatureFromPressureDensity( self, P, rho ):

    # Calculate molar volume 
    bar_v = self.molecular_weight/rho
    
    # Calculate temperature guess using ideal-gas model
    T = P*bar_v/self.R_universal

    x_0 = T

    for iter in range(self.max_aitken_iter):
      x_1 = ((bar_v - self.eos_b) / self.R_universal) * (P + (self.calculate_eos_a(x_0) /((bar_v**2.0) + 2.0 * self.eos_b * bar_v - (self.eos_b**2.0))))
      x_2 = ((bar_v - self.eos_b) / self.R_universal) * (P + (self.calculate_eos_a(x_1) /((bar_v**2.0) + 2.0 * self.eos_b * bar_v - (self.eos_b**2.0))))
    
      denominator = x_2 - 2.0 * x_1 + x_0

      #T = x_2 - (pow(x_2 - x_1, 2.0)) / denominator
      T = x_2 - (x_2 - x_1)**2.0 / (denominator + 1.0e-10)
    
      if abs((T - x_2) / T) < self.aitken_relative_tolerance:
          break  # If the result is within tolerance, leave the loop!
          
      x_0 = T  # Otherwise, update x_0 to iterate again...
      
    return T


  def calculateTemperatureFromPressureDensityWithInitialGuess( self, T, P, rho ):

    ### Define functions
    functions = []
    functions.append( lambda variable_T : ( ( self.calculatePressureFromTemperatureDensity(variable_T,rho) ) - P )/P )

    ### Initialize variables: T
    variables = np.zeros( 1 )
    variables[0] = T    # Use input T value as initial guess

    ### Set normalization factors of Jacobian's diagonal: T
    norm_factors = np.zeros( 1 )
    norm_factors[0] = copy.deepcopy( abs( variables[0] ) )

    ### Solve nonlinear system
    variables = self.n_solve( functions, variables, norm_factors )
    #print( variables )

    ### Assign solution to T
    T = variables[0]
    #print( T)

    return T


  def calculateInternalEnergyFromPressureTemperatureDensity(self, P, T, rho):

    bar_v = self.molecular_weight/rho
    e = (1.0/self.molecular_weight)*self.calculateMolarInternalEnergyFromPressureTemperatureMolarVolume(P, T, bar_v)

    return e


  def calculateEntropyFromPressureTemperatureDensity(self, P, T, rho):

    bar_v = self.molecular_weight/rho
    s = (1.0/self.molecular_weight)*self.calculateMolarEntropyFromPressureTemperatureMolarVolume(P, T, bar_v)

    return s


  def calculatePressureTemperatureFromDensityInternalEnergy(self, P, T, rho, e):

    # At fixed density the internal energy depends on T alone, with de/dT|_v = c_v > 0,
    # so this is a 1-D root find in T (not the 2-D (P,T) system it looks like): P is
    # fixed by the EoS once (T, rho) are known. Newton on T with the exact derivative c_v,
    # warm-started from the incoming T. Same result as the old fsolve(P,T) but with one
    # unknown and no MINPACK / finite-difference-Jacobian overhead.
    # Robustness clamp: the iterate T_k is kept inside a range where every model term
    # stays evaluable -- T_FLOOR keeps sqrt(T) (eos_a, its derivatives) real, and T_CEIL
    # keeps the NASA polynomials in range (they call exit() above 6000 K). The Newton step
    # is also damped to at most a factor ~1.5/0.5 per iteration so a bad e (e.g. ke>E in a
    # diverging cell) cannot fling T_k to a negative/NaN value and crash the whole run; the
    # cell is left at a clamped value for the convergence watchdog/tool to flag instead.
    T_FLOOR = 1.0       # K
    T_CEIL  = 5999.0    # K (NASA polynomials are valid only below 6000 K)
    bar_v = self.molecular_weight/rho
    T_k = min(max(T, T_FLOOR), T_CEIL)
    for _ in range(self.max_newton_iter):
      P_k   = self.calculatePressureFromTemperatureDensity(T_k, rho)
      e_k   = self.calculateMolarInternalEnergyFromPressureTemperatureMolarVolume(P_k, T_k, bar_v)/self.molecular_weight
      # Newton derivative c_v = de/dT|_v. Compute c_v directly (the expensive c_p departure
      # function is not needed here).
      std_bar_c_v = self.calculateMolarStdCpFromNASApolynomials(T_k) - self.R_universal
      c_v = (std_bar_c_v + self.calculateDepartureFunctionMolarCv(P_k, T_k, bar_v))/self.molecular_weight
      if (not np.isfinite(c_v)) or c_v <= 0.0:
        break  # no reliable Newton step (e.g. c_v ill-behaved near critical); keep current T_k
      dT    = (e - e_k)/c_v
      dT    = max(min(dT, 0.5*T_k), -0.5*T_k)  # damp so the iterate cannot leave the valid range
      T_k  += dT
      T_k   = min(max(T_k, T_FLOOR), T_CEIL)
      if abs(dT) <= self.newton_rtol*abs(T_k):
        break

    P = self.calculatePressureFromTemperatureDensity(T_k, rho)
    T = T_k

    return P, T


  def calculateDensityInternalEnergyFromPressureTemperature(self, rho, e, P, T):

    # Auxiliar parameters
    eos_a  = self.calculate_eos_a(T)
    eos_en = P*self.eos_b + self.R_universal*T
    a      = P
    b      = 2.0*P*self.eos_b - eos_en
    c      = (-1.0)*P*self.eos_b*self.eos_b - 2.0*eos_en*self.eos_b + eos_a
    d      = (-1.0)*self.eos_b*(eos_a + (-1.0)*eos_en*self.eos_b)

    # Cubic solve in molar volume, then select the physical root.
    v_1, v_2, v_3 = self.calculateRootsCubicPolynomial(a, b, c, d)
    bar_v = self.selectPhysicalMolarVolume(P, T, (v_1, v_2, v_3))
    rho = self.molecular_weight/bar_v

    # Calculate e
    e = (1.0/self.molecular_weight)*self.calculateMolarInternalEnergyFromPressureTemperatureMolarVolume(P, T, bar_v)

    return rho, e


  def selectPhysicalMolarVolume(self, P, T, roots):
    # Choose the physical molar volume among the cubic roots.
    #  - one real root  -> single phase, take it (the supercritical / common case)
    #  - three real roots (two-phase dome) -> the middle root is mechanically unstable;
    #    pick the stable phase = lower fugacity (lower Gibbs energy) between the liquid
    #    (smallest v) and vapor (largest v) roots.
    # A physical molar volume is real and exceeds the covolume eos_b.
    real_vs = []
    for rt in roots:
      if abs(rt.imag) < 1.0e-10*max(1.0, abs(rt.real)) and rt.real > self.eos_b:
        real_vs.append(rt.real)

    if len(real_vs) == 0:
      # No physical root (should not happen for valid P,T); fall back to the largest real part.
      return max(rt.real for rt in roots)
    if len(real_vs) == 1:
      return real_vs[0]

    v_liq = min(real_vs)
    v_vap = max(real_vs)
    A = self.calculate_A(P, T)
    B = self.calculate_B(P, T)
    phi_liq = self.calculate_fugacity(self.calculate_Z(P, T, v_liq), A, B)
    phi_vap = self.calculate_fugacity(self.calculate_Z(P, T, v_vap), A, B)
    return v_liq if phi_liq < phi_vap else v_vap


  def calculateSpecificHeatCapacities(self, c_v, c_p, P, T, rho):

    bar_v = self.molecular_weight/rho
    std_bar_c_p = self.calculateMolarStdCpFromNASApolynomials(T)
    std_bar_c_v = std_bar_c_p - self.R_universal
    
    c_v = (1.0/self.molecular_weight)*(std_bar_c_v + self.calculateDepartureFunctionMolarCv(P, T, bar_v))
    c_p = (1.0/self.molecular_weight)*(std_bar_c_p + self.calculateDepartureFunctionMolarCp(P, T, bar_v))

    return c_v, c_p


  def calculateHeatCapacitiesRatio(self, P, rho):
    
    bar_v = self.molecular_weight/rho
    T = self.calculateTemperatureFromPressureDensity(P, rho)
    std_bar_c_p = self.calculateMolarStdCpFromNASApolynomials(T)
    std_bar_c_v = std_bar_c_p - self.R_universal

    c_v = (1.0/self.molecular_weight) * (std_bar_c_v + self.calculateDepartureFunctionMolarCv(P, T, bar_v))
    c_p = (1.0/self.molecular_weight) * (std_bar_c_p + self.calculateDepartureFunctionMolarCp(P, T, bar_v))

    gamma = c_p/c_v
    
    return gamma


  def calculateSoundSpeed(self, P, T, rho):
    
    bar_v = self.molecular_weight/rho

    sos = np.sqrt( 1.0/(rho*self.calculateIsentropicCompressibility(P, T, bar_v)) )

    return sos
  

  def calculateVolumeExpansivity(self, T,bar_v ):

    dP_dT_const_v = self.calculateDPDTConstantMolarVolume(T,bar_v)
    dP_dv_const_T  = self.calculateDPDvConstantTemperature(T,bar_v)

    Expansivity = (-1.0)*(dP_dT_const_v/(bar_v*dP_dv_const_T))

    return Expansivity
  

  def calculateIsothermalCompressibility(self, T, bar_v):

    dP_dv_const_T = self.calculateDPDvConstantTemperature(T, bar_v)
    isothermal_compressibility = (-1.0)/(bar_v*dP_dv_const_T)

    return isothermal_compressibility
  

  def calculateIsentropicCompressibility(self, P, T, bar_v):

    isothermal_compressibility = self.calculateIsothermalCompressibility(T, bar_v)
    expansivity                = self.calculateVolumeExpansivity(T, bar_v)
    bar_c_p                    = self.calculateMolarStdCpFromNASApolynomials(T) + self.calculateDepartureFunctionMolarCp(P, T, bar_v)
      
    isentropic_compressibility = (isothermal_compressibility - ((bar_v * T * (expansivity ** 2.0)) / bar_c_p))
    
    return isentropic_compressibility
  

  def calculatePressureFromTemperatureDensity(self, T, rho):

    bar_v = self.molecular_weight/rho
    P = (self.R_universal*T/(bar_v - self.eos_b)) - (self.calculate_eos_a( T )/(bar_v*bar_v + 2.0*self.eos_b*bar_v - self.eos_b*self.eos_b))
    
    return P
  

  def calculateMolarInternalEnergyFromPressureTemperatureMolarVolume(self,P, T, bar_v):

    bar_e = self.calculateMolarStdEnthalpyFromNASApolynomials(T) + self.calculateDepartureFunctionMolarEnthalpy(P, T, bar_v) - P * bar_v
    
    return bar_e
   

  def calculateMolarEntropyFromPressureTemperatureMolarVolume(self,P, T, bar_v):

    # s(T,P) = s_std(T)            [NASA std-state entropy at P_ref_std]
    #        - R*ln(P/P_ref_std)   [ideal-gas pressure dependence; was missing]
    #        + Delta_s_departure   [real-gas residual relative to ideal at (T,P)]
    bar_s = ( self.calculateMolarStdEntropyFromNASApolynomials(T)
              - self.R_universal*np.log(P/self.P_ref_std)
              + self.calculateDepartureFunctionMolarEntropy(P, T, bar_v) )

    return bar_s


  def calculate_eos_a( self, T ):
     
     eos_a = (0.457*((self.R_universal*self.critical_temperature)**2)/(self.critical_pressure))*(1+self.eos_kappa*(1-np.sqrt(T/self.critical_temperature)))**2
     
     return eos_a


  def calculate_eos_a_first_derivative(self, T):

    eos_a_first_derivative = self.eos_kappa*self.eos_ac*( ( self.eos_kappa/self.critical_temperature ) - ( ( 1.0 + self.eos_kappa )/np.sqrt(T*self.critical_temperature ) ) )
    # eos_a_first_derivative = eos_kappa*eos_ac*( ( eos_kappa/critical_temperature ) - ( ( 1.0 + eos_kappa )/sqrt( T*critical_temperature ) ) );
   
    return eos_a_first_derivative


  def calculate_eos_a_second_derivative(self, T):
 
    eos_a_second_derivative = (self.eos_kappa*self.eos_ac*(1.0 + self.eos_kappa))/(2.0*np.sqrt(T**3.0)*self.critical_temperature)

    return eos_a_second_derivative


  def calculate_Z(self, P, T,bar_v):

    Z = (P*bar_v)/(self.R_universal*T)

    return Z
  

  def calculate_A(self, P, T ):

    eos_a = self.calculate_eos_a( T )

    A = (eos_a*P)/((self.R_universal*T)**2.0)

    return A
  

  def calculate_B(self, P, T):
    
    B = (self.eos_b*P)/(self.R_universal*T)

    return B
  

  def calculate_M(self, Z, B):

    M = (Z**2+2.0*B*Z-B**2)/(Z - B)
    
    return M
  

  def calculate_N(self, eos_a_first_derivative, B):

    N = eos_a_first_derivative * (B/(self.eos_b*self.R_universal))

    return N


  def calculate_fugacity(self, Z, A, B):

    term1 = Z - 1.0 - np.log( Z - B )
    term2 = A/( 2.0*np.sqrt( 2.0 )*B )
    term3 = np.log( ( Z + ( 1.0 + np.sqrt( 2.0 ) )*B)/( Z + ( 1.0 - np.sqrt( 2.0 ) )*B ) )

    return np.exp( term1 - term2*term3 )


  def calculateMolarStdCpFromNASApolynomials(self, T):

    std_bar_c_p = 0.0
    if 200.0 <= T < 1000.0:
        std_bar_c_p = self.R_universal*( self.NASA_coefficients[7] + self.NASA_coefficients[8]*T + self.NASA_coefficients[9]*(T**2.0) + self.NASA_coefficients[10]*(T**3.0) + self.NASA_coefficients[11]*(T**4.0) )
    elif 1000.0 <= T < 6000.0:
        std_bar_c_p = self.R_universal*( self.NASA_coefficients[0] + self.NASA_coefficients[1]*T + self.NASA_coefficients[2]*(T**2.0) + self.NASA_coefficients[3]*(T**3.0) + self.NASA_coefficients[4]*(T**4.0) )
    elif T < 200:
        # Assume constant temperature below T = 200 K	    
        T_min = 200.0	    
        std_bar_c_p = self.R_universal*( self.NASA_coefficients[7] + self.NASA_coefficients[8]*T_min + self.NASA_coefficients[9]*(T_min**2.0) + self.NASA_coefficients[10]*(T_min**3.0) + self.NASA_coefficients[11]*(T_min**4.0) )
    else:
        print(f"\nNASA 7-coefficient polynomials for std bar c_p. T = {T} is above 6000 K.\n\n")
        exit()

    return std_bar_c_p
     

  def calculateMolarStdEnthalpyFromNASApolynomials(self, T):

    std_bar_h = 0.0
    if T >= 200.0 and T < 1000.0:
      std_bar_h = self.R_universal*T*( self.NASA_coefficients[7] + self.NASA_coefficients[8]*T/2.0 + self.NASA_coefficients[9]*(T**2.0)/3.0 + self.NASA_coefficients[10]*(T**3.0)/4.0 + self.NASA_coefficients[11]*(T**4.0)/5.0 + self.NASA_coefficients[12]/T )
    elif T >= 1000.0 and T < 6000.0:
      std_bar_h = self.R_universal*T*( self.NASA_coefficients[0] + self.NASA_coefficients[1]*T/2.0 + self.NASA_coefficients[2]*(T**2.0)/3.0 + self.NASA_coefficients[3]*(T**3.0)/4.0 + self.NASA_coefficients[4]*(T**4.0)/5.0 + self.NASA_coefficients[5]/T )
    elif T < 200.0:
      T_min = 200.0
      std_bar_h_min   = self.R_universal*T_min*( self.NASA_coefficients[7] + self.NASA_coefficients[8]*T_min/2.0 + self.NASA_coefficients[9]*(T_min**2.0)/3.0 + self.NASA_coefficients[10]*(T_min**3.0)/4.0 + self.NASA_coefficients[11]*(T_min**4.0)/5.0 + self.NASA_coefficients[12]/T_min )
      std_bar_h_slope = self.R_universal*( self.NASA_coefficients[7] + self.NASA_coefficients[8]*T_min + self.NASA_coefficients[9]*(T_min**2.0) + self.NASA_coefficients[10]*(T_min**3.0) + self.NASA_coefficients[11]*(T_min**4.0) )
      std_bar_h = std_bar_h_min + std_bar_h_slope*(T-T_min)
    else:
       print(f"\nNASA 7-coefficient polynomials for std bar h. T = {T} is above 6000 K.\n\n")
       exit()

    return std_bar_h


  def calculateMolarStdEntropyFromNASApolynomials(self, T):

    std_bar_s = 0.0
    if T >= 200.0 and T < 1000.0:
      std_bar_s = self.R_universal*( self.NASA_coefficients[7]*np.log(T) + self.NASA_coefficients[8]*T + self.NASA_coefficients[9]*(T**2.0)/2.0 + self.NASA_coefficients[10]*(T**3.0)/3.0 + self.NASA_coefficients[11]*(T**4.0)/4.0 + self.NASA_coefficients[13] )
    elif T >= 1000.0 and T < 6000.0:
      std_bar_s = self.R_universal*( self.NASA_coefficients[0]*np.log(T) + self.NASA_coefficients[1]*T + self.NASA_coefficients[2]*(T**2.0)/2.0 + self.NASA_coefficients[3]*(T**3.0)/3.0 + self.NASA_coefficients[4]*(T**4.0)/4.0 + self.NASA_coefficients[6] )
    elif T < 200.0:
      T_min = 200.0
      std_bar_s_min   = self.R_universal*( self.NASA_coefficients[7]*np.log(T_min) + self.NASA_coefficients[8]*T_min + self.NASA_coefficients[9]*(T_min**2.0)/2.0 + self.NASA_coefficients[10]*(T_min**3.0)/3.0 + self.NASA_coefficients[11]*(T_min**4.0)/4.0 + self.NASA_coefficients[13] )
      std_bar_s_slope = self.R_universal*( self.NASA_coefficients[7]/T_min + self.NASA_coefficients[8] + self.NASA_coefficients[9]*T_min + self.NASA_coefficients[10]*(T_min**2.0) + self.NASA_coefficients[11]*(T_min**3.0) )
      std_bar_s = std_bar_s_min + std_bar_s_slope*(T-T_min)
    else:
       print(f"\nNASA 7-coefficient polynomials for std bar s. T = {T} is above 6000 K.\n\n")
       exit()

    return std_bar_s


  def calculateDepartureFunctionMolarCp(self, P, T, bar_v):

    # Peng-Robinson model:
    # D.Y. Peng, D. B. Robinson 
    # A new two-constants equation of state
    # Industrial and Engineering Chemistry: Fundamental , 15 , 59-64 , 1976.

    eos_a_first_derivative  = self.calculate_eos_a_first_derivative( T )
    eos_a_second_derivative = self.calculate_eos_a_second_derivative ( T )
    Z                       = self.calculate_Z(P, T, bar_v)
    A                       = self.calculate_A(P, T)
    B                       = self.calculate_B(P, T)
    M                       = self.calculate_M(Z, B)
    N                       = self.calculate_N(eos_a_first_derivative, B)

    Delta_bar_c_p = ((self.R_universal*(M - N)**2))/((M**2.0) - 2.0*A* (Z + B)) - ((T*eos_a_second_derivative)/(2.0*np.sqrt(2.0)*self.eos_b))*np.log((Z + (1.0 - np.sqrt(2.0))*B)/(Z + (1.0 + np.sqrt(2.0))*B)) - self.R_universal
    
    return Delta_bar_c_p


  def calculateDepartureFunctionMolarCv(self, P, T, bar_v):

    # Peng-Robinson model 
    # D. Y. Peng, D. B. Robinson 
    # A new two-constant equations of State
    # Industrial and engineering Chemistry: Fundamental, 15, 59-64 , 1976.

    eos_a_second_derivative = self.calculate_eos_a_second_derivative( T )
    Z                       = self.calculate_Z(P, T, bar_v)
    B                       = self.calculate_B(P, T)

    Delta_bar_c_v = (-1.0)*((T*eos_a_second_derivative)/(2.0*np.sqrt( 2.0)*self.eos_b))*np.log((Z+(1.0 - np.sqrt( 2.0 ))*B)/(Z+(1.0 + np.sqrt(2.0))*B))
    
    return Delta_bar_c_v


  def calculateDepartureFunctionMolarEnthalpy(self, P, T, bar_v):

    # Peng-Robinson model 
    # D. Y. Peng, D. B. Robinson 
    # A new two-constant equations of State
    # Industrial and engineering Chemistry: Fundamental, 15, 59-64 , 1976.

    eos_a                  = self.calculate_eos_a( T )
    eos_a_first_derivative = self.calculate_eos_a_first_derivative( T )
    Z                      = self.calculate_Z(P, T, bar_v)
    B                      = self.calculate_B(P, T)

    Delta_bar_h = self.R_universal*T*(Z - 1.0) + (( eos_a - eos_a_first_derivative*T)/(2.0*np.sqrt(2.0)*self.eos_b))*np.log((Z + (1.0 - np.sqrt(2.0))*B)/(Z + (1.0 + np.sqrt(2.0))*B))
    
    return Delta_bar_h
 

  def calculateDepartureFunctionMolarEntropy(self, P, T, bar_v):

    # Peng-Robinson model 
    # D. Y. Peng, D. B. Robinson 
    # A new two-constant equations of State
    # Industrial and engineering Chemistry: Fundamental, 15, 59-64 , 1976.

    Z = self.calculate_Z(P, T, bar_v)
    A = self.calculate_A(P, T)
    B = self.calculate_B(P, T)

    Delta_bar_s = self.R_universal*( np.log(Z - B) + ( A/( 2.0*np.sqrt( 2.0 )*B ) )*( self.eos_kappa*np.sqrt( T/self.critical_temperature )/( 1.0 + self.eos_kappa*( 1.0 - np.sqrt( T/self.critical_temperature ) ) ) )*np.log((Z + (1.0 - np.sqrt(2.0))*B)/(Z + (1.0 + np.sqrt(2.0))*B)) )
    
    return Delta_bar_s


  def calculateTemperatureFromPressureMolarVolume(self, P, bar_v):

    # Initial temperature guess using ideal-gas model
    T = P*bar_v/self.R_universal
        
    # Aitken’s delta-squared process:
    x_0 = T

    for iter in range(self.max_aitken_iter):
      x_1 = ((bar_v - self.eos_b) / self.R_universal) * (P + (self.calculate_eos_a(x_0) / (bar_v**2 + 2*self.eos_b*bar_v - self.eos_b**2)))
      x_2 = ((bar_v - self.eos_b) / self.R_universal) * (P + (self.calculate_eos_a(x_1) / (bar_v**2 + 2*self.eos_b*bar_v - self.eos_b**2)))

      denominator = x_2 - 2*x_1 + x_0
      T = x_2 - ((x_2 - x_1)**2) / denominator

      if abs((T - x_2) / T) < self.aitken_relative_tolerance:
        break	# If the result is within tolerance, leave the loop!
            
        x_0 = T	# Otherwise, update x_0 to iterate again ...

    return T

  
  def calculateDPDTConstantMolarVolume(self, T, bar_v):

    eos_a_first_derivative  = self.calculate_eos_a_first_derivative( T )

    dP_dT_const_v = (self.R_universal/(bar_v - self.eos_b))-(eos_a_first_derivative/(bar_v*bar_v + 2.0*bar_v*self.eos_b - self.eos_b*self.eos_b))

    return dP_dT_const_v


  def calculateDPDvConstantTemperature(self, T, bar_v):

    eos_a = self.calculate_eos_a( T )

    dP_dv_const_T = (-1.0)*((self.R_universal*T)/(bar_v-self.eos_b)**2) + (eos_a*(2.0*bar_v + 2.0*self.eos_b))/((bar_v**2.0) + 2.0*bar_v*self.eos_b - self.eos_b**2.0)**2.0
    
    return dP_dv_const_T


  def calculateRootsCubicPolynomial( self, a, b, c, d):
    
    if a == 0:
      print("The coefficient of the cube of x is 0. Please use the utility for a SECOND degree quadratic. No further action taken.")
      return None, None, None                                                   # To define a null number 

    if d == 0:
      print("One root is 0. Now divide through by x and use the utility for a SECOND degree quadratic to solve the resulting equation for the other two roots. No further action taken.")
      return None, None, None                                                   # To define a null number

    b /= a                                                                      # normalize all coeficients value of the cubic equation                                                                            
    c /= a                                                                      # normalize all coeficients value of the cubic equation 
    d /= a                                                                      # normalize all coeficients value of the cubic equation 

    disc, q, r, dum1, s, t, term1, r13 = 0, 0, 0, 0, 0, 0, 0, 0                 #intermediate parameters of the roots
    q = (3.0*c - (b*b))/9.0
    r = -(27.0*d) + b*(9.0*c - 2.0*(b*b))
    r /= 54.0
    disc = q*q*q + r*r

    term1 = b/3.0

    if disc > 0:
      s = r + np.sqrt(disc)
      if s < 0:
        s = (-1.0)*( (-s)**(1.0/3.0) )
      else:
        s = s**(1.0/3.0)
      t = r - np.sqrt(disc)
      if t < 0:
        t = (-1.0)*( (-t)**(1.0/3.0) ) 
      else:
        t = t**(1.0/3.0) 
      root_1 = complex( (-1.0)*term1 + s + t, 0.0)
      term1 += (s + t)/2.0
      term1  = np.sqrt(3.0)*(-t+s)/2
      root_2 = complex( (-1.0)*( b/3.0 + (s + t)/2.0 ), term1 )
      root_3 = complex( (-1.0)*( b/3.0 + (s + t)/2.0 ), (-1.0)*term1 )
      #print(root_1.real)
      #print(root_2.real)
      #print(root_3.real)
      return root_1, root_2, root_3

    # End if (disc > 0)

    # disc <= 0: three real roots (two-phase region). Trigonometric solution
    # (casus irreducibilis). Caller is responsible for selecting the physical root.
    if q < 0.0:
      arg = r/np.sqrt((-q)**3.0)
      arg = max(-1.0, min(1.0, arg))                    # clamp for floating-point safety
      theta_c = np.arccos(arg)
      sqrt_mq = np.sqrt(-q)
      root_1 = complex( 2.0*sqrt_mq*np.cos( theta_c/3.0 )                 - b/3.0, 0.0 )
      root_2 = complex( 2.0*sqrt_mq*np.cos( (theta_c + 2.0*np.pi)/3.0 )   - b/3.0, 0.0 )
      root_3 = complex( 2.0*sqrt_mq*np.cos( (theta_c + 4.0*np.pi)/3.0 )   - b/3.0, 0.0 )
      return root_1, root_2, root_3

    # q >= 0 with disc <= 0: degenerate triple real root
    root = complex( (-1.0)*b/3.0, 0.0 )
    return root, root, root


##################################################################
####################### COOLPROP MODEL ###########################
##################################################################

class CoolPropModel(BaseThermodynamicModel):

  ###  Attributes


  ### Constructor
  def __init__(self, fluid_name):       # Install CoolProp: pip3 install coolprop

    super(BaseThermodynamicModel,self).__init__()
    self.fluid = fluid_name
    # Persistent low-level state: one update() per call, then cheap property queries --
    # no PropsSI string parsing and no redundant re-flash of the same state. Backend
    # 'HEOS' is the full Span-Wagner reference EoS; switch to 'BICUBIC&HEOS' to tabulate.
    self._AS = CP.AbstractState('HEOS', fluid_name)
    self._IN_DU = CoolProp.DmassUmass_INPUTS   # (rho, e)
    self._IN_DT = CoolProp.DmassT_INPUTS       # (rho, T)
    self._IN_PT = CoolProp.PT_INPUTS           # (P, T)
    self._IN_DP = CoolProp.DmassP_INPUTS       # (rho, P)
    self._IN_QT = CoolProp.QT_INPUTS           # (Q, T)
    self._TWOPHASE = CoolProp.iphase_twophase
    self.molecular_weight = CP.PropsSI("M", self.fluid)
    self.R_specific = self.R_universal/self.molecular_weight


  ### Methods

  def calculatePressureFromTemperatureDensity(self, T, rho):
    self._AS.update(self._IN_DT, rho, T)
    return self._AS.p()


  def calculateTemperatureFromPressureDensity(self, P, rho):
    self._AS.update(self._IN_DP, rho, P)
    return self._AS.T()


  def calculateTemperatureFromPressureDensityWithInitialGuess(self, T, P, rho):
    self._AS.update(self._IN_DP, rho, P)
    return self._AS.T()


  def calculateInternalEnergyFromPressureTemperatureDensity(self, P, T, rho):
    self._AS.update(self._IN_DT, rho, T)
    return self._AS.umass()


  def calculateEntropyFromPressureTemperatureDensity(self, P, T, rho):
    self._AS.update(self._IN_DT, rho, T)
    return self._AS.smass()


  def calculatePressureTemperatureFromDensityInternalEnergy(self, P, T, rho, e):
    self._AS.update(self._IN_DU, rho, e)
    return self._AS.p(), self._AS.T()


  def calculatePressureTemperatureEntropyFromDensityInternalEnergy(self, rho, e):
    # Single flash -> P, T and s together (the hot path in thermodynamic_state).
    self._AS.update(self._IN_DU, rho, e)
    return self._AS.p(), self._AS.T(), self._AS.smass()


  def calculateDensityInternalEnergyFromPressureTemperature(self, rho, e, P, T):
    self._AS.update(self._IN_PT, P, T)
    return self._AS.rhomass(), self._AS.umass()


  def calculateSpecificHeatCapacities(self, c_v, c_p, P, T, rho):
    self._AS.update(self._IN_DT, rho, T)
    if self._AS.phase() == self._TWOPHASE:
      # CoolProp's c_v,c_p are not meaningful inside the dome; quality-weight the
      # saturated single-phase values (adequate for the explicit time-step / CFL).
      rho_v, rho_l, _, _, cv_v, cv_l, cp_v, cp_l = self._saturated_properties(T)
      x = self._vapor_quality(rho, rho_v, rho_l)
      return x*cv_v + (1.0-x)*cv_l, x*cp_v + (1.0-x)*cp_l
    return self._AS.cvmass(), self._AS.cpmass()


  def calculateHeatCapacitiesRatio(self, P, T, rho):
    c_v, c_p = self.calculateSpecificHeatCapacities(-1.0, -1.0, P, T, rho)
    return c_p/c_v


  def calculateSoundSpeed(self, P, T, rho):
    self._AS.update(self._IN_DT, rho, T)
    if self._AS.phase() == self._TWOPHASE:
      # CoolProp gives no 'A' inside the dome. Wood's homogeneous-equilibrium speed of
      # sound: 1/(rho a^2) = alpha/(rho_v a_v^2) + (1-alpha)/(rho_l a_l^2), alpha = void
      # fraction. Collapses far below either phase (the physical two-phase drop).
      rho_v, rho_l, a_v, a_l, _, _, _, _ = self._saturated_properties(T)
      x = self._vapor_quality(rho, rho_v, rho_l)
      alpha = min(max(x * rho / rho_v, 0.0), 1.0)
      inv = alpha/(rho_v*a_v*a_v) + (1.0-alpha)/(rho_l*a_l*a_l)
      return np.sqrt(1.0/(rho*inv))
    return self._AS.speed_sound()


  ### Two-phase helpers (CoolProp supplies neither sound speed nor c_v,c_p in the dome)
  def _saturated_properties(self, T):
    self._AS.update(self._IN_QT, 1.0, T)
    rho_v, a_v, cv_v, cp_v = self._AS.rhomass(), self._AS.speed_sound(), self._AS.cvmass(), self._AS.cpmass()
    self._AS.update(self._IN_QT, 0.0, T)
    rho_l, a_l, cv_l, cp_l = self._AS.rhomass(), self._AS.speed_sound(), self._AS.cvmass(), self._AS.cpmass()
    return rho_v, rho_l, a_v, a_l, cv_v, cv_l, cp_v, cp_l

  def _vapor_quality(self, rho, rho_v, rho_l):
    x = (1.0/rho - 1.0/rho_l) / (1.0/rho_v - 1.0/rho_l)
    return min(max(x, 0.0), 1.0)


  def calculateVolumeExpansivity(self, T, rho):
        
    expansivity = CP.PropsSI('ISOBARIC_EXPANSION_COEFFICIENT', 'T', T, 'D', rho, self.fluid)
        
    return expansivity


  def calculateIsothermalCompressibility(self, T, rho):
        
    isothermal_compressibility = CP.PropsSI('ISOTHERMAL_COMPRESSIBILITY', 'T', T, 'D', rho, self.fluid)
        
    return isothermal_compressibility


  def calculateIsentropicCompressibility(self, P, T, rho):
        
    isentropic_compressibility = CP.PropsSI('ISENTROPIC_COMPRESSIBILITY', 'T', T, 'D', rho, self.fluid)
        
    return isentropic_compressibility
   

################################################################################
######################### BASE TRANSPORT COEFFICIENTS ##########################
################################################################################


class BaseTransportCoefficients:				#### Base transport coefficients

  ### Atributes (Variables that don't change with temperature)
  R_universal      = 8.31446261815324      # Universal gas constant [j/(mol k)]
  #R_specific       = -1.0                  # Specific gas constant [J/(kg K)]
  #molecular_weight = -1.0                  # Molecular weight [kg/mol]
  mu_value         = -1.0                  # Dynamic viscosity [Pa·s]
  kappa_value      = -1.0                  # Thermal Conductivity [ W/(m·K)]


  ### Constructor 
  def __init__(self, R_universal):

    self.R_universal = R_universal 


  ### Methods 

  def calculateDynamicViscosity(self, P, T, rho):

    return 0.0


  def calculateThermalConductivity(self, P, T, rho): 

    return 0.0 


################################################################################
####################### CONSTANT TRANSPORT COEFFICIENTS ########################
################################################################################


class ConstantTransportCoefficients(BaseTransportCoefficients):                 ### Constant transport coefficients

  ### Constructor
  def __init__(self, mu, kappa):

    super(BaseTransportCoefficients,self).__init__()
    self.mu       = mu 
    self.kappa    = kappa 

  
  ### Methods

  def calculateDynamicViscosity(self, P, T, rho):
    
    return( self.mu )


  def calculateThermalConductivity(self, P, T, rho):
    
    return( self.kappa )
 

################################################################################
################### LOW-PRESSURE GAS TRANSPORT COEFFICIENTS ####################
################################################################################


class LowPressureGasTransportCoefficients(BaseTransportCoefficients):           ### Low-pressure gas variable transport coefficients

  ### Atributes (Variables that don't change with temperature)
  mu_0        = -1.0
  kappa_0     = -1.0 
  T_0         = -1.0 
  S_mu        = -1.0
  S_kappa     = -1.0 


  ### Constructor 
  def __init__(self, mu_0, kappa_0, T_0, S_mu, S_kappa):

    super(BaseTransportCoefficients,self).__init__()  
    self.mu_0    =  mu_0
    self.kappa_0 = kappa_0
    self.T_0     = T_0
    self.S_mu    = S_mu 
    self.S_kappa = S_kappa


  def calculateDynamicViscosity(self, P, T, rho):

    return ((self.mu_0*(T/self.T_0)**1.5)*((self.T_0 + self.S_mu)/(T + self.S_mu)))

  
  def calculateThermalConductivity(self, P, T, rho):

    return (self.kappa_0*((T/self.T_0)**1.5))*((self.T_0 + self.S_kappa)/(T + self.S_kappa))


################################################################################
#################### HIGH-PRESSURE TRANSPORT COEFFICIENTS ######################
################################################################################


class HighPressureTransportCoeficients(BaseTransportCoefficients):              ### High-pressure transport coefficients

  ### Atributes (Variables that don't change with temperature)
  molecular_weight                = -1.0 
  critical_temperature            = -1.0
  critical_molar_volume           = -1.0
  acentric_factor                 = -1.0
  dipole_moment                   = -1.0
  association_factor              = -1.0
  NASA_coefficients               = (-1.0)*np.ones(15)


  ### Constructor
  def __init__(self, molecular_weight, acentric_factor, critical_temperature, critical_molar_volume, NASA_coefficients, dipole_moment,association_factor):

    super(BaseTransportCoefficients,self).__init__()  
    self.molecular_weight           = molecular_weight
    self.critical_temperature       = critical_temperature
    self.critical_molar_volume      = critical_molar_volume   
    self.acentric_factor            = acentric_factor
    self.dipole_moment              = dipole_moment
    self.association_factor         = association_factor  
    self.NASA_coefficients          = NASA_coefficients
    
    ### Adimensional dipole moment -- Poling et al. The properties of gases and liquids. McGraw-Hill, 2001.
    self.adimensional_dipole_moment = 131.3*( self.dipole_moment/np.sqrt( ( 1.0e6*self.critical_molar_volume )*self.critical_temperature ) )
      
    ### Viscosity mu -- Poling et al. The properties of gases and liquids. McGraw-Hill, 2001. (9.40, Table 9-6)
    a1_mu  = 6.324;   a2_mu  = 1.210e-3;    a3_mu  = 5.283;   a4_mu  = 6.623;   a5_mu  = 19.745    
    a6_mu  = -1.900;  a7_mu  = 24.275;      a8_mu  = 0.7972;  a9_mu  = -0.2382; a10_mu = 0.06863   
    b1_mu  = 50.412;  b2_mu  = -1.154e-3 ; b3_mu  = 254.209;  b4_mu  = 38.096 ;  b5_mu  = 7.630;   b6_mu  = -12.537; b7_mu  = 3.450;  b8_mu  = 1.117;   b9_mu  = 0.06770; b10_mu = 0.3479; c1_mu  = -51.680; c2_mu  = -6.257e-3;  c3_mu  = -168.48; c4_mu  = -8.464; d10_mu = -0.727;
    c5_mu  = -14.354; c6_mu  = 4.985     ; c7_mu  = -11.291;  c8_mu  = 0.01235;  c9_mu  = -0.8163; c10_mu = 0.5926 ; d1_mu  = 1189.0; d2_mu  = 0.03728; d3_mu  = 3898.0;  d4_mu = 31.42;   d5_mu  = 31.53;   d6_mu  = -18.15;     d7_mu  = 69.35;   d8_mu  = -4.117; d9_mu  = 4.025;

    self.E1_mu  = a1_mu  + b1_mu*self.acentric_factor  + c1_mu*self.adimensional_dipole_moment**4.0  + d1_mu*self.association_factor
    self.E2_mu  = a2_mu  + b2_mu*self.acentric_factor  + c2_mu*self.adimensional_dipole_moment**4.0  + d2_mu*self.association_factor
    self.E3_mu  = a3_mu  + b3_mu*self.acentric_factor  + c3_mu*self.adimensional_dipole_moment**4.0  + d3_mu*self.association_factor
    self.E4_mu  = a4_mu  + b4_mu*self.acentric_factor  + c4_mu*self.adimensional_dipole_moment**4.0  + d4_mu*self.association_factor
    self.E5_mu  = a5_mu  + b5_mu*self.acentric_factor  + c5_mu*self.adimensional_dipole_moment**4.0  + d5_mu*self.association_factor
    self.E6_mu  = a6_mu  + b6_mu*self.acentric_factor  + c6_mu*self.adimensional_dipole_moment**4.0  + d6_mu*self.association_factor
    self.E7_mu  = a7_mu  + b7_mu*self.acentric_factor  + c7_mu*self.adimensional_dipole_moment**4.0  + d7_mu*self.association_factor
    self.E8_mu  = a8_mu  + b8_mu*self.acentric_factor  + c8_mu*self.adimensional_dipole_moment**4.0  + d8_mu*self.association_factor
    self.E9_mu  = a9_mu  + b9_mu*self.acentric_factor  + c9_mu*self.adimensional_dipole_moment**4.0  + d9_mu*self.association_factor
    self.E10_mu = a10_mu + b10_mu*self.acentric_factor + c10_mu*self.adimensional_dipole_moment**4.0 + d10_mu*self.association_factor

    ### Thermal conductivity k -- Poling et al. The properties of gases and liquids. McGraw-Hill, 2001. (10.23, Table 10-3)
    a1_k = 2.4166*1.00;  a3_k = 6.6107*1.00; a5_k = 7.9274*0.10; a7_k = 9.1089*10.0;  b2_k = -1.5094*1.00; b4_k = -8.9139*1.00; b6_k = 1.2801*10.00; c1_k = -9.1858*0.10; c3_k = 6.4760*10.00; c5_k = -6.9369*0.10; c7_k = -5.4217*10.0; d2_k = 6.9983*10.00; d4_k = 7.4344*10.00; d6_k = 6.5529*10.00
    a2_k = -5.0924*0.1;  a4_k = 1.4543*10.0; a6_k = -5.8634*1.0; b1_k = 7.4824*0.100; b3_k = 5.6207*1.000; b5_k = 8.2019*0.100; b7_k = 1.2811*100.0; c2_k = -4.9991*10.0; c4_k = -5.6379*1.00; c6_k = 9.5893*1.000; d1_k = 1.2172*100.0; d3_k = 2.7039*10.00; d5_k = 6.3173*1.000; d7_k = 5.2381*100.0

    self.B1_k = a1_k + b1_k*self.acentric_factor + c1_k*self.adimensional_dipole_moment**4.0 + d1_k*self.association_factor
    self.B2_k = a2_k + b2_k*self.acentric_factor + c2_k*self.adimensional_dipole_moment**4.0 + d2_k*self.association_factor
    self.B3_k = a3_k + b3_k*self.acentric_factor + c3_k*self.adimensional_dipole_moment**4.0 + d3_k*self.association_factor
    self.B4_k = a4_k + b4_k*self.acentric_factor + c4_k*self.adimensional_dipole_moment**4.0 + d4_k*self.association_factor
    self.B5_k = a5_k + b5_k*self.acentric_factor + c5_k*self.adimensional_dipole_moment**4.0 + d5_k*self.association_factor
    self.B6_k = a6_k + b6_k*self.acentric_factor + c6_k*self.adimensional_dipole_moment**4.0 + d6_k*self.association_factor
    self.B7_k = a7_k + b7_k*self.acentric_factor + c7_k*self.adimensional_dipole_moment**4.0 + d7_k*self.association_factor

  ### Methods 

  def calculateDynamicViscosity(self, P, T, rho):

    # T. H. Chung, L. L. Lee, K. E. Starling.
    # Applications of kinetic gas theories and multiparameter correlation for prediction of dilute gas viscosity and thermal conductivity.
    # Industrial & Engineering Chemistry Fundamentals, 23, 8-13, 1984.

    # T. H. Chung, M. Ajlan, L. L. Lee, K. E. Starling.
    # Generalized multiparameter correlation for nonpolar and polar fluid transport properties.
    # Industrial & Engineering Chemistry Fundamentals, 27, 671-679, 1988.

    # Auxiliar coefficients
    v     = self.molecular_weight/rho
    Y     = self.critical_molar_volume/( 6.0*v )
    T_ast = 1.2593*( T/self.critical_temperature )
    Omega = 1.16145*T_ast**(-0.14874)  + 0.52487*np.exp( -0.77320*T_ast ) + 2.16178*np.exp( -2.43787*T_ast )
    G1    = ( 1.0 - 0.5*Y )/((1.0 - Y)**3.0)
    Fc    = 1.0 - 0.2756*self.acentric_factor + 0.059035*self.adimensional_dipole_moment**4.0 + self.association_factor
    
    # Additional auxiliar coefficients
    G2_mu      = ( self.E1_mu*( 1.0 - np.exp( -self.E4_mu*Y ) )/Y + self.E2_mu*G1*np.exp( self.E5_mu*Y ) + self.E3_mu*G1 )/( self.E1_mu*self.E4_mu + self.E2_mu + self.E3_mu )
    mu_ast_ast = (self.E7_mu*Y**2.0)*G2_mu*np.exp( (self.E8_mu + self.E9_mu/T_ast) + self.E10_mu*T_ast**(-2.0))
    mu_ast     = ( np.sqrt( T_ast )*Fc/Omega )*( 1.0/G2_mu + self.E6_mu*Y ) + mu_ast_ast

    # Calculate viscosity
    mu = (1.0e-7)*mu_ast*( ( 36.344*np.sqrt( ( 1.0e3*self.molecular_weight )*self.critical_temperature ) )/ (1.0e6*self.critical_molar_volume)**(2.0/3.0)) 

    return( mu )
  

  def calculateThermalConductivity(self, P, T, rho): 

    # T. H. Chung, L. L. Lee, K. E. Starling.
    # Applications of kinetic gas theories and multiparameter correlation for prediction of dilute gas viscosity and thermal conductivity.
    # Industrial & Engineering Chemistry Fundamentals, 23, 8-13, 1984.

    # T. H. Chung, M. Ajlan, L. L. Lee, K. E. Starling.
    # Generalized multiparameter correlation for nonpolar and polar fluid transport properties.
    # Industrial & Engineering Chemistry Fundamentals, 27, 671-679, 1988.

    # Auxiliar coefficients
    std_bar_c_p =  self.calculateMolarStdCpFromNASApolynomials(T)

    v           = self.molecular_weight/rho
    Y           = self.critical_molar_volume/( 6.0*v )
    T_ast       = 1.2593*( T/self.critical_temperature )
    Omega       = 1.16145*pow( T_ast, -0.14874 ) + 0.52487*np.exp( -0.77320*T_ast ) + 2.16178*np.exp( -2.43787*T_ast )
    G1          = ( 1.0 - 0.5*Y )/pow( 1.0 - Y, 3.0 )
    Fc          = 1.0 - 0.2756*self.acentric_factor + 0.059035*pow( self.adimensional_dipole_moment, 4.0 ) + self.association_factor
    std_bar_c_p = self.calculateMolarStdCpFromNASApolynomials( T )

    # Additional auxiliar coefficients
    mu_0_k  = 40.785e-7*Fc*np.sqrt( 1.0e3*self.molecular_weight*T )/( pow( 1.0e6*self.critical_molar_volume, 2.0/3.0 )*Omega )
    alpha_k = ( std_bar_c_p/self.R_universal - 1.0 ) - 1.5                                                                                   
    beta_k  = 0.7862 - 0.7109*self.acentric_factor + 1.3168*pow( self.acentric_factor, 2.0 )
    gamma_k = 2.0 + 10.5*pow( T/self.critical_temperature, 2.0 )
    Psi_k   = 1.0 + alpha_k*( ( 0.215 + 0.28288*alpha_k - 1.061*beta_k + 0.26665*gamma_k )/( 0.6366 + beta_k*gamma_k + 1.061*alpha_k*beta_k) )
    q_k     = 0.003586*( np.sqrt( self.critical_temperature/self.molecular_weight )/pow( ( 1.0e6*self.critical_molar_volume ), 2.0/3.0 ) )
    G3_k    = ( ( ( self.B1_k/Y )*( 1.0 - np.exp( (-1.0)*self.B4_k*Y ) ) ) + ( self.B2_k*G1*np.exp( self.B5_k*Y ) ) + ( self.B3_k*G1 ) )/( self.B1_k*self.B4_k + self.B2_k + self.B3_k )

    # Calculate thermal conductivity
    kappa = ( 31.2*mu_0_k*Psi_k/self.molecular_weight )*( 1.0/G3_k + self.B6_k*Y ) + q_k*self.B7_k*pow( Y, 2.0 )*np.sqrt( T/self.critical_temperature)*G3_k

    return( kappa )
  

  def calculateMolarStdCpFromNASApolynomials(self, T): 

    std_bar_c_p = 0.0
    if (T>=200.0) and (T<1000.0) :
        std_bar_c_p = self.R_universal*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*T + self.NASA_coefficients[9]*T**2.0 + self.NASA_coefficients[10]*T**3.0 + self.NASA_coefficients[11]* T**4.0)
    elif (T>=1000.0) and (T<6000.0):
        std_bar_c_p = self.R_universal*(self.NASA_coefficients[0] + self.NASA_coefficients[1]*T + self.NASA_coefficients[2]*T**2.0 + self.NASA_coefficients[3]*T**3.0 + self.NASA_coefficients[4]*T**4.0)
    elif (T < 200):
        # Assume constant temperature below T = 200 K	    
        T_min = 200.0	    
        std_bar_c_p = self.R_universal*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*T_min + self.NASA_coefficients[9]*T_min**2.0 + self.NASA_coefficients[10]*T_min**3.0 + self.NASA_coefficients[11]*T_min**4.0)
    else:
        print(f"\nNASA 7-coefficient polynomials for std bar c_p. T = {T} is above 6000 K.\n\n")
        exit()

    return std_bar_c_p


################################################################################
####################### COOLPROP TRANSPORT COEFFICIENTS ########################
################################################################################


class CoolPropTransportCoefficients(BaseTransportCoefficients):                 ### Constant transport coefficients

  ### Constructor
  def __init__(self, fluid_name):

    super(BaseTransportCoefficients,self).__init__()
    self.fluid = fluid_name

  
  ### Methods

  def calculateDynamicViscosity(self, P, T, rho):
    
    mu = CP.PropsSI('V', 'P', P, 'T', T, self.fluid)
    
    return( mu )


  def calculateThermalConductivity(self, P, T, rho):
    
    #kappa = CP.PropsSI('CONDUCTIVITY', 'P', P, 'T', T, self.fluid)
    kappa = CP.PropsSI('L', 'P', P, 'T', T, self.fluid)
    
    return( kappa )
