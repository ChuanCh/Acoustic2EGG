# From Voice_EGG.wav and Log.aiff generate input and output items.
# Voice_EGG.wav format:
#   Channel 1: Mic signal; Channel 2: EGG signal.
# Log.aiff format:

import wave
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tensorflow as tf

Directory = r'C:\Users\60163\Documents\SuperCollider\Recordings'

MicFile = ''
CycleFile = 'test_CycleDetection.wav'
FFTDFile = 'test_Log.csv'

# Log file including timing and FFT descriptors
FFTData = os.path.join(Directory, FFTDFile)
FFT = np.loadtxt(FFTData, delimiter=',')
sliced = []
for i in range(len(FFT)):
    dit = FFT[i, 0] * 44100
    sliced.append(round(dit))
print(len(sliced))

# Mic data sliced into sequences, --> Input
Samplerate, Mic = wavfile.read(os.path.join(Directory, CycleFile))
MicData = Mic[:, 0]
slicedData = []
for i in range(len(sliced)-1):
    temp = MicData[sliced[i]:sliced[i+1]]
    slicedData.append(temp)
slicedData = np.array(slicedData, dtype=object)
# SlicedMicData = []
# temp = []
# for i in range(len(Mic)):
#     if Mic[i, 1] == 0:
#         temp.append(Mic[i, 0])
#     else:
#         SlicedMicData.append(temp)
#         temp = []
# SlicedMicData = np.array(SlicedMicData, dtype=object)
# print(SlicedMicData.shape)

# Fourier Arrays --> output
FourierArray = np.array(FFT[:-1, 12:])

# Validation group, used to calculate the distance trained vs. collected
MetricData = FFT[:, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]]


# Put it in next py file
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(slicedData, FourierArray, epochs=100)

# Evaluate the model on test data
test_input_data = np.random.rand(100, 10)
test_output_data = np.random.rand(100, 1)
test_loss = model.evaluate(test_input_data, test_output_data, verbose=0)
print('Test loss:', test_loss)