import numpy as np
import matplotlib.pyplot as plt

def age_young(x):
    if x <= 29:
        return 1
    elif 29 < x < 38:
        return (38 - x) / (38 - 29)
    else:
        return 0

def age_mid(x):
    if 33 < x <= 38:
        return (x - 33) / (38 - 33)
    elif 38 < x < 45:
        return (45 - x) / (45 - 38)
    else:
        return 0

def age_old(x):
    if 40 < x <= 48:
        return (x-40)/(48-40)
    elif 48 < x < 58:
        return (58-x)/(58-48)
    else:
        return 0

def age_veryold(x):
    if 52 < x < 60:
        return (x-52)/(60-52)
    elif 60 <= x:
        return 1
    else:
        return 0
    

# Generate x values from -1 to 3
# x = np.linspace(0, 80, 1000)

# # Calculate y values using the piecewise function
# y_young = np.array([age_young(xi) for xi in x])
# y_mid = np.array([age_mid(xi) for xi in x])
# y_old = np.array([age_old(xi) for xi in x])
# y_very_old = np.array([age_veryold(xi) for xi in x])
# # Plot the graph
# plt.plot(x, y_young, label="Young", color ='blue')
# plt.scatter(29, 1, color ='blue')
# # plt.annotate('(29, 1)', xy=(29,1), xytext=(27,1.04))
# plt.scatter(38, 0, color ='blue')
# # plt.annotate('(38, 0)', xy=(38,0), xytext=(36, -0.04))
# plt.plot(x, y_mid, label = "Middle", color= 'red')
# plt.scatter(33, 0, color= 'red')
# plt.scatter(38, 1, color= 'red')
# plt.scatter(45, 0, color= 'red')
# plt.plot(x, y_old, label = "Old", color ='orange')
# plt.scatter(40, 0, color ='orange')
# plt.scatter(48, 1, color ='orange')
# plt.scatter(58, 0, color ='orange')
# plt.plot(x, y_very_old, label = "Very Old", color ='green')
# plt.scatter(52, 0, color ='green')
# plt.scatter(60, 1, color ='green')
#
# plt.xlabel("Age")
# plt.legend()
# plt.grid(visible=True, which="both")
# plt.show()

def bloodPressure_low(x):
    if x <= 111:
        return 1
    elif 111 < x < 134:
        return (134-x)/(134-111)
    else:
        return 0

def bloodPressure_medium(x):
    if 127 < x <= 139:
        return (x-127)/(139-127)
    elif 139 < x < 153:
        return (153-x)/(153-139)
    else:
        return 0

def bloodPressure_high(x):
    if 142 < x <= 157:
        return (x-142)/(157-142)
    elif 157 < x < 172:
        return (172-x)/(172-157)
    else:
        return 0

def bloodPressure_veryhigh(x):
    if 154 < x < 171:
        return (x-154)/(171-154)
    elif 171 <= x:
        return 1
    else:
        return 0

# x = np.linspace(50, 250, 1000)
# y_low = np.array([bloodPressure_low(xi) for xi in x])
# y_medium = np.array([bloodPressure_medium(xi) for xi in x])
# y_high = np.array([bloodPressure_high(xi) for xi in x])
# y_very_high = np.array([bloodPressure_veryhigh(xi) for xi in x])
# # Plot the graph
# plt.plot(x, y_low, label="Low", color ='blue')
# plt.scatter(111, 1, color ='blue')
# plt.scatter(134, 0, color ='blue')
# plt.plot(x, y_medium, label = "Medium", color= 'red')
# plt.scatter(127, 0, color= 'red')
# plt.scatter(139, 1, color= 'red')
# plt.scatter(153, 0, color= 'red')
# plt.plot(x, y_high, label = "High", color ='orange')
# plt.scatter(142, 0, color ='orange')
# plt.scatter(157, 1, color ='orange')
# plt.scatter(172, 0, color ='orange')
# plt.plot(x, y_very_high, label = "Very High", color ='green')
# plt.scatter(154, 0, color ='green')
# plt.scatter(171, 1, color ='green')
#
# plt.xlabel("Tensiunea Arteriala")
# plt.legend()
# plt.grid(visible=True, which="both")
# plt.show()

def cholesterol_low(x):
    if x <= 151:
        return 1
    elif 151 < x < 197:
        return (197 - x) / (197 - 151)
    else:
        return 0


def cholesterol_medium(x):
    if 188 < x <= 215:
        return (x - 188) / (215 - 188)
    elif 215 < x < 250:
        return (250 - x) / (250 - 215)
    else:
        return 0


def cholesterol_high(x):
    if 217 < x <= 263:
        return (x - 217) / (263 - 217)
    elif 263 < x < 307:
        return (307 - x) / (307 - 263)
    else:
        return 0


def cholesterol_veryhigh(x):
    if 281 < x < 347:
        return (x - 281) / (347 - 281)
    elif 347 <= x:
        return 1
    else:
        return 0


# x = np.linspace(100, 400, 1000)
# y_low = np.array([cholesterol_low(xi) for xi in x])
# y_medium = np.array([cholesterol_medium(xi) for xi in x])
# y_high = np.array([cholesterol_high(xi) for xi in x])
# y_very_high = np.array([cholesterol_veryhigh(xi) for xi in x])
# # Plot the graph
# plt.plot(x, y_low, label="Low", color ='blue')
# plt.scatter(151, 1, color ='blue')
# plt.scatter(197, 0, color ='blue')
# plt.plot(x, y_medium, label = "Medium", color= 'red')
# plt.scatter(188, 0, color= 'red')
# plt.scatter(215, 1, color= 'red')
# plt.scatter(250, 0, color= 'red')
# plt.plot(x, y_high, label = "High", color ='orange')
# plt.scatter(217, 0, color ='orange')
# plt.scatter(263, 1, color ='orange')
# plt.scatter(307, 0, color ='orange')
# plt.plot(x, y_very_high, label = "Very High", color ='green')
# plt.scatter(281, 0, color ='green')
# plt.scatter(347, 1, color ='green')
#
# plt.xlabel("Colesterol[mg/dl]")
# plt.legend()
# plt.grid(visible=True, which="both")
# plt.show()


def normal(x):
    if x <= 0:
        return 1
    elif 0 < x < 0.4:
        return (0.4 - x) / 0.4
    else:
        return 0


def abnormal(x):
    if 0.2 < x <= 1:
        return (x - 0.2) / (1 - 0.2)
    elif 1 < x < 1.8:
        return (1.8 - x) / (1.8 - 1)
    else:
        return 0


def hypertrophy(x):
    if 1.4 < x < 1.9:
        return (x - 1.4) / (1.9 - 1.4)
    elif 1.9 <= x:
        return 1
    else:
        return 0

# x = np.linspace(-0.5, 2.5, 1000)
# y_low = np.array([normal(xi) for xi in x])
# y_medium = np.array([abnormal(xi) for xi in x])
# y_high = np.array([hypertrophy(xi) for xi in x])
# # Plot the graph
# plt.plot(x, y_low, label="Normal", color='blue')
# plt.scatter(0, 1, color='blue')
# plt.scatter(0.4, 0, color='blue')
# plt.plot(x, y_medium, label="Abnormal", color='red')
# plt.scatter(0.2, 0, color='red')
# plt.scatter(1, 1, color='red')
# plt.scatter(1.8, 0, color='red')
# plt.plot(x, y_high, label="Hypertrophy", color='orange')
# plt.scatter(1.4, 0, color='orange')
# plt.scatter(1.9, 1, color='orange')
#
# plt.xlabel("Rezultat EKG")
# plt.legend()
# plt.grid(visible=True, which="both")
# plt.show()

def heartRate_low(x):
    if x <= 100:
        return 1
    elif 100 < x < 141:
        return (141-x)/(141-100)
    else:
        return 0

def heartRate_medium(x):
    if 111 < x <= 152:
        return (x-111)/(152-111)
    elif 152 < x < 194:
        return (194-x)/(194-152)
    else:
        return 0

def heartRate_high(x):
    if 152 < x < 210:
        return (x-152)/(210-152)
    elif 210 <= x:
        return 1
    else:
        return 0

# x = np.linspace(-100, 400, 1000)
# y_low = np.array([heartRate_low(xi) for xi in x])
# y_medium = np.array([heartRate_medium(xi) for xi in x])
# y_high = np.array([heartRate_high(xi) for xi in x])
# # Plot the graph
# plt.plot(x, y_low, label="Low", color='blue')
# plt.scatter(100, 1, color='blue')
# plt.scatter(141, 0, color='blue')
# plt.plot(x, y_medium, label="Medium", color='red')
# plt.scatter(111, 0, color='red')
# plt.scatter(152, 1, color='red')
# plt.scatter(194, 0, color='red')
# plt.plot(x, y_high, label="High", color='orange')
# plt.scatter(152, 0, color='orange')
# plt.scatter(210, 1, color='orange')
#
# plt.xlabel("Tensiunea arteriala maxima")
# plt.legend()
# plt.grid(visible=True, which="both")
# plt.show()


def oldPeak_low(x):
    if x <= 1:
        return 1
    elif 1 < x < 2:
        return (2 - x) / (2 - 1)
    else:
        return 0


def oldPeak_risk(x):
    if 1.5 < x <= 2.8:
        return (x - 1.5) / (2.8 - 1.5)
    elif 2.8 < x < 4.2:
        return (4.2 - x) / (4.2 - 2.8)
    else:
        return 0


def oldPeak_terrible(x):
    if 2.5 < x < 4:
        return (x - 2.5) / (4 - 2.5)
    elif 4 <= x:
        return 1
    else:
        return 0

# x = np.linspace(0, 6, 1000)
# y_low = np.array([oldPeak_low(xi) for xi in x])
# y_medium = np.array([oldPeak_risk(xi) for xi in x])
# y_high = np.array([oldPeak_terrible(xi) for xi in x])
# # Plot the graph
# plt.plot(x, y_low, label="Low", color='blue')
# plt.scatter(1, 1, color='blue')
# plt.scatter(2, 0, color='blue')
# plt.plot(x, y_medium, label="Risk", color='red')
# plt.scatter(1.5, 0, color='red')
# plt.scatter(2.8, 1, color='red')
# plt.scatter(4.2, 0, color='red')
# plt.plot(x, y_high, label="Terrible", color='orange')
# plt.scatter(2.5, 0, color='orange')
# plt.scatter(4, 1, color='orange')
#
# plt.xlabel("Depresiunea ST")
# plt.legend()
# plt.grid(visible=True, which="both")
# plt.show()


def outPut_sick(x):
    if 0 < x <= 0.5:
        return x / 0.5
    elif 0.5 < x <= 1:
        return (1 - x) / (1 - 0.5)
    else:
        return 0


def healthy(x):
    if x <= 0:
        return 1
    elif 0 < x <= 1:
        return (1 - x) / (1 - 0)
    else:
        return 0
    
x = np.linspace(-1, 3, 1000)
y_low = np.array([outPut_sick(xi) for xi in x])
y_medium = np.array([healthy(xi) for xi in x])
# Plot the graph
plt.plot(x, y_low, label="Sick", color='blue')
plt.scatter(0, 0, color='blue')
plt.scatter(0.5, 1, color='blue')
plt.scatter(1, 0, color='blue')
plt.plot(x, y_medium, label="Healthy", color='red')
plt.scatter(0, 1, color='red')
plt.scatter(1, 0, color='red')

plt.xlabel("Rezultat sanatate")
plt.legend()
plt.grid(visible=True, which="both")
plt.show()