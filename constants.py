import numpy as np
import scipy.constants as cst

TAU = 4.6e-9 # 423 nm transition lifetime [s]
GAMMA = 1/TAU
CA40_MOLAR_MASS = 39.9625908 # g/mol https://pubchem.ncbi.nlm.nih.gov/compound/Calcium-40
CA40_MASS = CA40_MOLAR_MASS * 1e-3 / cst.Avogadro
PEAK_FREQ = 709.07855e12 # Hz
TRANSITION_FREQ = 709078373.01e6 # Hz
I_SAT = 45 # W/m²
I_LASER = 10.73e-3 / np.pi*(5e-3)**2 # W/m² (power/detector area)

# correction offsets for frequency range
FREQ_CORRECTION_LOW = 0.18e9
FREQ_CORRECTION_HIGH = -0.11e9
FREQ_CORRECTION = [FREQ_CORRECTION_LOW, FREQ_CORRECTION_HIGH]
