############################################################
## Fichier pour l'émeteur du Rasberry Pi
############################################################
#Importation de nos bibliothèques
import numpy as np
from numpy.fft import fft
import scipy
import scipy.signal as signal
from scipy.signal import butter, lfilter, freqz, sosfilt, sosfreqz, filtfilt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.io import wavfile
import wave
#%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
from commpy.filters import rcosfilter
from ModulationPy import PSKModem, QAMModem
from pydub import AudioSegment



##  Traitement du signal
y ,sr = wavfile.read("interstellar.wav")
durée = sr.shape[0] / y #durée du fichier audio en seconde = nb d'échantillon / fréquence 
temps = np.linspace(0, durée, sr.shape[0]) # <- Pour savoir quand est ce que chaque point est prélevé 
# la transformée de Fourier à court terme (STFT).
# Les STFT peuvent être utilisés comme moyen de quantifier le changement de la fréquence 
# et du contenu de phase d'un signal non stationnaire au fil du temps.
D = librosa.stft(y)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
# Fréquence d'échantillonnage
fe = 1 # Hz
# Fréquence de nyquist
f_nyq = fe / 2.  # Hz
# Fréquence de coupure
fc = 0.4999  # Hz
# Préparation du filtre de Butterworth en passe-bas
b, a = signal.butter(4, fc/f_nyq, 'low', analog=False)

# Application du filtre
s_but = signal.filtfilt(b, a, y)
D = librosa.stft(s_but)
log_power = librosa.amplitude_to_db(D**2, ref = np.max)
## Émetteur
modem = QAMModem(16, bin_input=True, soft_decision=False, bin_output=True)
def binaire(s):
    ords = (ord(c) for c in s)
    shifts = (7, 6, 5, 4, 3, 2, 1, 0)
    return [(o >> shift) & 1 for o in ords for shift in shifts]

def char(bits):
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

texte = "L'objectif de ce projet est de créer un système de communication permettant de transmettre de l'information, en l'occurrence une image via la diffusion d'une musique. Afin de parvenir à cet objectif, nous allons utiliser la technique du tatouage du son pour dissimuler les données de notre image dans les hautes fréquences, qui sont inaudibles à l'oreille humaine. Ensuite, à l'aide d’un micro les capter, et après analyse retrouver l'image transmise. Pour ce faire nous disposons de 2 cartes Raspberry pi, un micro et des enceintes."
texte_bin = binaire(texte) #Texte convertit en binaire
texte_binaire = np.array(texte_bin)#Permet de convertir list en array

image = np.array([1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0]) # message d'entrée

msg = np.concatenate((image,texte_binaire)) #msg = image + texte que l'on souhaite transmettre 
texte_char = char(texte_binaire) #Convertit du binaire au texte -> Vérification de la conversion en binaire
modulation = modem.modulate(msg) # modulation -> Moduler un tableau de bits en symboles de constellation
## Convolution
Tsymbol = 0.000187
Fsamp = 48000
alpha = 1/2
N = int(5*Tsymbol*Fsamp)
delta_symbols = np.zeros(len(modulation)*int(Tsymbol*Fsamp),dtype =complex)
delta_symbols[::int(Tsymbol*Fsamp)]= modulation
temps, h_rc = rcosfilter(N, alpha, Tsymbol, Fsamp)
convo = np.convolve(delta_symbols , h_rc, mode="same") 
cnv = librosa.stft(np.real(convo))
cnv2=librosa.amplitude_to_db(abs(cnv))
D = librosa.stft(np.real(convo))
log_power = librosa.amplitude_to_db(D**2, ref = np.max)
Fcoupure = 16000 #Fréquence de coupure = milieu de la bande passante
t=np.linspace(0 , np.size(convo)/Fsamp , np.size(convo))
son = np.real(convo*np.exp(2*np.pi*1j*Fcoupure*t))
sonf = librosa.stft(son)
sonff=librosa.amplitude_to_db(abs(sonf))
D = librosa.stft(son)
log_power = librosa.amplitude_to_db(D**2, ref = np.max)
song = AudioSegment.from_wav("son.wav")
# reduce volume by 10 dB
#song = song - 10

# but let's make him very quiet
song = song - 35

# save the output
song.export("quieter1.wav", "wav")
sound1 = AudioSegment.from_file("son_filtre.wav") # son filtré 
sound2 = AudioSegment.from_file("quieter1.wav") 
#mix sound2 with sound1, starting at 47,5% into sound1)
tmpsound = sound1.overlay(sound2, position=0.475 * len(sound1))
tmpsound.export('tmpsound.wav',format='wav')


print("message envoyé")
