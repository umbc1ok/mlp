import numpy as np
import matplotlib.pyplot as plt
import locale
import decimal

# Set the desired decimal separator
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
# Read data from file

#np.loadtxt(s, converters=float)
#data = np.loadtxt(delimiter =" " ,'data.txt')
epochs = []
errors = []
file = open('./Stats/globalError.data', 'r')
lines = file.readlines()
for line in lines:
    data = line.strip().split(' ')
   # error = locale(atof[data[1]])
    epochs.append(int(data[0]))
    errors.append(float(data[1].replace(',', '.')))
print(epochs)
print(errors)
# Extract epochs and errors

# Plotting the graph
plt.plot(epochs, errors, linestyle='-', color='b')
plt.xlim(0, 100)
#plt.ylim(0, 1)
# Labeling the graph
plt.title('Error vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')

# Displaying the graph
plt.show()