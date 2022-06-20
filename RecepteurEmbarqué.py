###########################################################################################################################################
###########################################################################################################################################
##              ISYS 10 Communication via un canal audio : Comment dissimuler des données dans du son ?
##
## Auteurs : A.TANGAVELOU, A.AIMEUR, F.DANAN, A.RAOULT, Q.GUYOT
## Date : 20/06/2022
## Participants : A.AIMEUR, F.DANAN, A.RAOULT, Q.GUYOT, A.TANGAVELOU, C.SALZEDO
## Encadrant : V.BELMEGA 
##
## Code de de la carte Rasberry pi récepteur 
## realise le décodage du son envoyé par l'émetteur afin d'afficher les informations dissimuléses.
###########################################################################################################################################
###########################################################################################################################################

# enable plots in the notebook
import matplotlib.pyplot as plt## en com
# makes our plots prettier
import seaborn ## en com
seaborn.set(style='ticks')## en com

#import the audio playback widget
from IPython.display import Audio

# useful librairies
import mir_eval ## en com
import numpy as np
import scipy
from scipy.io import wavfile
from scipy import signal
import librosa ## en com
import librosa.display ## en com
from pydub import AudioSegment 
from ModulationPy import PSKModem, QAMModem
import wave
import contextlib
from commpy.filters import rrcosfilter
from turtle import *
import math
import time

# Différents types de filtres
def fir_high_pass(samples, fs, fH, N, outputType):
    # Referece: https://fiiir.com

    fH = fH / fs

    # Compute sinc filter.
    h = np.sinc(2 * fH * (np.arange(N) - (N - 1) / 2.)) 
    # Apply window.
    h *= np.hamming(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[int((N - 1) / 2)] += 1
    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(samples, h).astype(outputType)
    return s

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

def dessinerPixel(x , y):
    fillcolor("black")
    up()
    goto(x,y)
    down()
    begin_fill()
    for i in range(4):
        forward(10)
        left(90)
    end_fill()
    
def dessinerASCII(nomFichier, nbL):
    #Variables
    x =-300
    y = 100
    c = 0

    #parametres
    speed(15)
    delay(0)
    hideturtle()

    fichier = open(nomFichier, "r")
    lignes = fichier.readlines()
    
    
    for ligne in lignes:
        c += 1
        for bit in ligne:
            if(bit == "1" and c < nbL + 1):
                dessinerPixel(x , y)
            x+=10
        y-=10
        x=-300
        
    print("dessin terminé")

def convDemod(demodulation):
    """
    ENTREE : Prend en parametre le résultat de la démodulation du signal.
    SORTIE : Enregistre un fichier "resultat.txt" dans le dossier courant contenant l'image, et la phrase en binaire.
    
    """
    #variables :
    h = 0
    l = 0
    
    f = open('resultat.txt', 'w')    
    c=0
    for i in range(16):
        f.write(str(demodulation[i+752])[0])
        c += 1
        if(c==8):
            f.write("\n")
            c = 0
            
    for i in range(demodulation.size - (16+752)):
        f.write(str(demodulation[i+16+752])[0])
    f.close()
    
    f = open('resultat.txt', 'r') 
    lignes = f.readlines()
    
    c = 0
    for ligne in lignes:
        c += 1
        if(c == 1):
            l = ligne
        if(c == 2):
            h = ligne
    
    f.close()
    
    f = open('resultat.txt', 'w') 
    
    l = int(l, base = 2)
    h = int(h, base = 2)
    
    c = 0
    p = 0
    for i in range(demodulation.size - 16-752):
        c += 1
        f.write(str(demodulation[i+16+752])[0])
        if(c==32 and p<5):
            f.write("\n")
            p += 1
            c = 0
            
    f.close()
     
    dessinerASCII('resultat.txt', 5)
    
    fichier = open('resultat.txt', "r")
    lignes = fichier.readlines()
    texte = char(lignes[h])
    print("Phrase interprétée : " + texte)    
    f.close()
    
# Récupération du son + info
sampleRate, son_recepteur = wavfile.read("son_emmetteur.wav")

lh_samples_filtered = fir_high_pass(son_recepteur, sampleRate, 11500, 461, np.int16)             # First pass
lh_samples_filtered = fir_high_pass(lh_samples_filtered, sampleRate, 11500, 461, np.int16) # Second pass

# Demodulation
Fp = 16000 #Fréquence porteuse
t = np.linspace(0,np.size(lh_samples_filtered)/sampleRate, np.size(lh_samples_filtered)) 

I_reel = 2*lh_samples_filtered*np.cos(2*np.pi*Fp*t) #composante réel en phase 
Q_quadrature = -2*lh_samples_filtered*np.sin(2*np.pi*Fp*t) #composante réel en quadrature

alpha = 1/2
Tsymbol = 0.0001875*85
N = int(5 * Tsymbol * sampleRate)

temps, h_rc = rrcosfilter(N, alpha, Tsymbol, sampleRate)
I = np.convolve (I_reel, h_rc,'same')
Q = np.convolve (Q_quadrature,h_rc,'same')
x_t = I+1j*Q #signal complexe en bande de base

modem = QAMModem(16, 
                 bin_input=True,
                 soft_decision=False,
                 bin_output=True)

texte_corr="Interstellar prend comme point de départ un futur proche ressemblant à s’y méprendre au notre."
texte_bin_corr = binaire(texte_corr) #Texte convertit en binaire
texte_binaire_corr = np.array(texte_bin_corr)
msg_test = modem.modulate(texte_binaire_corr)

delta_symbols_corr = np.zeros(len(msg_test)*int(Tsymbol*sampleRate),dtype =complex)
delta_symbols_corr[::int(Tsymbol*sampleRate)]= msg_test
s_BB_oh = np.convolve(delta_symbols_corr, h_rc, mode="same") 

correl = scipy.signal.correlate(s_BB_oh,x_t[:np.size(s_BB_oh)])

rshift = np.size(s_BB_oh)-np.argmax(np.abs(correl))
corr_alpha =[]
A = np.linspace(0,1,1000)
for alpha in A :
    r_BB_coh_oh = x_t[rshift:(np.size(s_BB_oh)+rshift)]*np.exp(2j*np.pi*alpha)
    R = np.corrcoef(np.real(r_BB_coh_oh),np.real(s_BB_oh))
    corr_alpha.append((R[0,1]))

ccc = np.argmax(corr_alpha)

test = x_t[rshift:]*np.exp(2j*np.pi*A[ccc])
test2 = test[::int(sampleRate*Tsymbol)]

convo2 = np.convolve(h_rc,h_rc,'same')
Mconvo2 =np.max(convo2)
div =test2/Mconvo2

demodulation = modem.demodulate(div) # demodulation 
demodulation = np.round(demodulation).astype(int)
convDemod(demodulation)

time.sleep(20)
bye()
