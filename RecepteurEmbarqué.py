# enable plots in the notebook
import matplotlib.pyplot as plt
# makes our plots prettier
import seaborn
seaborn.set(style='ticks')

#import the audio playback widget
from IPython.display import Audio

# useful librairies
import mir_eval
import numpy as np
import scipy
from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
from pydub import AudioSegment 
from ModulationPy import PSKModem, QAMModem
import wave
import contextlib

################################################################################################
##
##                          Canal à virer 
##
################################################################################################
zu, sigma = 0,0
z = np.random.normal(zu,sigma,np.size(son_emetteur))

son_emetteur_noise = son_emetteur + z

random_desynch = np.random.randint(1,100)
vzeros = np.zeros(random_desynch,dtype = int)
son_recepteur = np.concatenate([vzeros,son_emetteur_noise])

################################################################################################
################################################################################################


lh_samples_filtered = fir_high_pass(son_, sampleRate, 12500, 461, np.int16)             # First pass
lh_samples_filtered = fir_high_pass(lh_samples_filtered, sampleRate, 12500, 461, np.int16) # Second pass

Fp = 16000 #Fréquence porteuse
t = np.linspace(0,np.size(son)/Fsamp, np.size(son)) 

I_reel = 2*son*np.cos(2*np.pi*Fp*t) #composante réel en phase 
Q_quadrature = -2*son*np.sin(2*np.pi*Fp*t) #composante réel en quadrature


# Vérification son et son2 : normal de ne pas avoir les mêmes valeurs. 
# --> supprimer facteurs 2 de I_reel et Q_quadrature
son2 = I_reel*np.cos(2*np.pi*Fp*t)-Q_quadrature*np.sin(2*np.pi*Fp*t)

I = np.convolve (I_reel, h_rc,'same')
Q = np.convolve (Q_quadrature,h_rc,'same')

x_t = I+1j*Q #signal complexe en bande de base

ech = x_t[::int(Tsymbol*Fsamp)]

convo2 = np.convolve(h_rc,h_rc,'same')

Mconvo2 =np.max(convo2)
div =ech/(len(son)/len(modulation)) #
#div = ech/(Tsymbol*Fsamp)


