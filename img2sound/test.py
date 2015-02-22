# http://stackoverflow.com/questions/10357992/how-to-generate-audio-from-a-numpy-array
import numpy as np
from scipy.io.wavfile import write
data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
scaled = np.int16(data/np.max(np.abs(data)) * 32767)
scaled = np.append(scaled, scaled)
write('test.wav', 44100, scaled)
