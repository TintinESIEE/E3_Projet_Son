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
from commpy.filters import rcosfilter, rrcosfilter


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

def get_start_end_frames(nFrames, sampleRate, tStart=None, tEnd=None):

    if tStart and tStart*sampleRate<nFrames:
        start = tStart*sampleRate
    else:
        start = 0

    if tEnd and tEnd*sampleRate<nFrames and tEnd*sampleRate>start:
        end = tEnd*sampleRate
    else:
        end = nFrames

    return (start,end,end-start)

def extract_audio(fname, tStart=None, tEnd=None):
    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        startFrame, endFrame, segFrames = get_start_end_frames(nFrames, sampleRate, tStart, tEnd)

        # Extract Raw Audio from multi-channel Wav File
        spf.setpos(startFrame)
        sig = spf.readframes(segFrames)
        spf.close()

        channels = interpret_wav(sig, segFrames, nChannels, ampWidth, True)

        return (channels, nChannels, sampleRate, ampWidth, nFrames)

def convert_to_mono(channels, nChannels, outputType):
    if nChannels == 2:
        samples = np.mean(np.array([channels[0], channels[1]]), axis=0)  # Convert to mono
    else:
        samples = channels[0]

    return samples.astype(outputType)

tStart=0
tEnd=20

channels, nChannels, sampleRate, ampWidth, nFrames = extract_audio('tmpsound.wav', tStart, tEnd)
samples = convert_to_mono(channels, nChannels, np.int16)

############################################################################################################################
############################################################################################################################
##
##                  à déterminer 
##
############################################################################################################################
############################################################################################################################

#Canal audio
zu, sigma = 0,0
z = np.random.normal(zu,sigma,np.size(samples2))

son_emetteur_noise = samples2 + z

random_desynch = np.random.randint(1,100)
vzeros = np.zeros(random_desynch,dtype = int)
son_recepteur = np.concatenate([vzeros,son_emetteur_noise])

##############################################################################################################################

lh_samples_filtered = fir_high_pass(son_, sampleRate, 12500, 461, np.int16)             # First pass
lh_samples_filtered = fir_high_pass(lh_samples_filtered, sampleRate, 12500, 461, np.int16) # Second pass

Fp = 16000 #Fréquence porteuse
t = np.linspace(0,np.size(lh_samples_filtered)/Fsamp, np.size(lh_samples_filtered)) 

I_reel = 2*lh_samples_filtered*np.cos(2*np.pi*Fp*t) #composante réel en phase 
Q_quadrature = -2*lh_samples_filtered*np.sin(2*np.pi*Fp*t) #composante réel en quadrature


# Vérification son et son2 : normal de ne pas avoir les mêmes valeurs. 
# --> supprimer facteurs 2 de I_reel et Q_quadrature
son2 = I_reel*np.cos(2*np.pi*Fp*t)-Q_quadrature*np.sin(2*np.pi*Fp*t)
#print(son)
#print(son2) 

I = np.convolve (I_reel, h_rc,'same')
Q = np.convolve (Q_quadrature, h_rc,'same')

If2 = librosa.stft(I)
IdB2 = librosa.amplitude_to_db(abs(If2))

x_t = I+1j*Q #signal complexe en bande de base

sans_synchro = x_t[::int(Tsymbol*Fsamp)]

convo_ss = np.convolve(h_rc,h_rc,'same')
Mconvo_ss =np.max(convo_ss)

div_ss =sans_synchro/Mconvo_ss

texte_corr="L'objectif de ce projet est de créer un système de communication permettant de transmettre de l'information, en l'occurrence une image via la diffusion d'une musique."
texte_bin_corr = binaire(texte_corr) #Texte convertit en binaire
texte_binaire_corr = np.array(texte_bin_corr)
msg_test = modem.modulate(texte_binaire_corr)


delta_symbols_corr = np.zeros(len(msg_test)*int(Tsymbol*Fsamp),dtype =complex)
delta_symbols_corr[::int(Tsymbol*Fsamp)]= msg_test

s_BB_oh = np.convolve(delta_symbols_corr, h_rc, mode="same") 

correl = scipy.signal.correlate(s_BB_oh,x_t[:np.size(s_BB_oh)])


########## peut être pas nécéssaire  ### 
##rshift = np.size(s_BB_oh)-np.argmax(np.abs(correl))

corr_alpha =[]
A = np.linspace(0,1,1000)
for alpha in A :
    r_BB_coh_oh = x_t[rshift:(np.size(s_BB_oh)+rshift)]*np.exp(2j*np.pi*alpha)
    R = np.corrcoef(np.real(r_BB_coh_oh),np.real(s_BB_oh))
    corr_alpha.append((R[0,1]))
    
ccc = np.argmax(corr_alpha)

test = x_t[rshift:]*np.exp(2j*np.pi*A[ccc])
test2 = test[::int(Fsamp*Tsymbol)]

convo2 = np.convolve(h_rc,h_rc,'same')
Mconvo2 =np.max(convo2)

div =test2/Mconvo2

demodulation2 = modem.demodulate(div) # modulation -> Moduler un tableau de bits en symboles de constellation

demodulation2 = np.round(demodulation2).astype(int)
texte_char = char(demodulation2) #Convertit du binaire au texte -> Vérification de la conversion en binaire
print(texte_char)



