import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import matplotlib.colors
import numpy.random as rand 
from scipy import signal
# points to use for Gaussian window
NUM_PNT = 11

# window to show in plots
axis_vals = [55, 80,0,.15]

# read height data, separate by gender and remove rows without valid height/gender data
#df = pd.read_csv('national_longitu_study/default.csv')
#heights1 = df[(df.R4829000 == 0) & (df.R0779800 > 0) ].R0779800
#heights2 = df[(df.R4829000 == 1) & (df.R0779800 > 0) ].R0779800
df = pd.read_csv('sample_data.csv')
heights1 = df[(df.gender == 0) & (df.height > 0) ].height
heights2 = df[(df.gender == 1) & (df.height > 0) ].height


heights_1_mean = np.mean(heights1)
heights_2_mean = np.mean(heights2)
heights_1_std = np.std(heights1)
heights_2_std = np.std(heights2)

heights1_max_hist = .15 
heights2_max_hist = .15 
scale = .6

heights_comb = np.concatenate([heights1, heights2])
heights_comb_mean = np.mean(heights_comb)
heights_comb_std = np.std(heights_comb)
heights_comb_max_hist = .027
MU1_INIT = 50
MU2_INIT = 51
STD1_INIT = 10
STD2_INIT = 10

def simData():
    mu1 = MU1_INIT
    mu2 = MU2_INIT
    std1 = STD1_INIT
    std2 = STD2_INIT
    delta = 2
    t= 0.0
    dt = 1
    t_max = 25.0
    while t<t_max:

        # E-step
        print "mu1: " + str(mu1) + " mu2: " + str(mu2) + " std1: " + str(std1) + " std2: " + str(std2)
        Pn_1,Pn_2 = E_step(mu1,mu2,std1,std2)
        
        # M-step
        mu1,mu2,std1,std2,N1,N2 = M_step(Pn_1,Pn_2)

        t = t + dt
        yield mu1, mu2, std1, std2, N1, N2, t

def simPoints(simData):
    mu1,mu2,std1,std2,N1,N2,t = simData[0],simData[1],simData[2],simData[3],simData[4],simData[5],simData[6]

    line1.set_xdata( [mu1, mu1])
    level = .399 / std1 * N1/(N1+N2)
    line1.set_ydata( [0, level])
    line2.set_xdata( [mu1-std1, mu1+std1])
    line2.set_ydata( [level*scale, level*scale] )
    x_points,gaussian_wind = Gaussian_window(mu1,std1,1.0*N1/(N1+N2))
    g_curve1.set_xdata(x_points)
    g_curve1.set_ydata(gaussian_wind)
    

    line3.set_xdata( [mu2, mu2])
    level = .399 / std2 * N2/(N1+N2)
    line3.set_ydata( [0, level])
    line4.set_xdata( [mu2-std2, mu2+std2])
    line4.set_ydata( [level*scale, level*scale])
    x_points,gaussian_wind = Gaussian_window(mu2,std2,1.0*N2/(N1+N2))
    g_curve2.set_xdata(x_points)
    g_curve2.set_ydata(gaussian_wind)

    iter_text.set_text(iter_template%(t,mu1,std1,1.0*N1/(N1+N2),mu2,std2,1.0*N2/(N1+N2)))
    fig.canvas.draw()

def E_step(mu1,mu2,std1,std2):
    
    Pn_1 = np.array([1.0 / (1 + 1.0*std1/std2 * np.exp((x-mu1)**2 / 2 / std1 - (x-mu2)**2/2/std2))
                     for x in heights_comb] )
    Pn_2 = np.array([1-x for x in Pn_1])

    return Pn_1,Pn_2

def M_step(Pn_1, Pn_2):
    N1 = Pn_1.sum()
    N2 = Pn_2.sum()

    pi1 = N1/(N1+N2)
    pi2 = N2/(N1+N2)

    mu1 = 1.0 * sum(Pn_1 * heights_comb) /  N1
    mu2 = 1.0 * sum(Pn_2 * heights_comb) /  N2

    temp1 = [(x-mu1)**2 for x in heights_comb] / N1
    std1 = (sum(temp1 * Pn_1))**.5
    temp2 = [(x-mu2)**2 for x in heights_comb] / N2
    std2 = (sum(temp2 * Pn_2))**.5
    
    return mu1,mu2,std1,std2,N1,N2

def Gaussian_window(mu, std, weight=1):
    gaussian_wind = weight * signal.gaussian(NUM_PNT,std=std) / (2*np.pi)**.5 / std
    x_points = mu + np.linspace(-1.0*(NUM_PNT-1)/2,(NUM_PNT-1)/2,NUM_PNT)
    return x_points,gaussian_wind
        
def init():
    sns.distplot(heights_comb,label='Heights (combined)',bins=range(min(heights_comb),max(heights_comb)),color='#808080')
    plt.grid(True)
    plt.xlabel('Height (cm)',fontsize=16)

fig = plt.figure(1,figsize=(20,10))
ax0 = fig.add_subplot(2,2,1)
sns.distplot(heights1,label='Heights1',bins=range(min(heights1),max(heights1)),color='#0000F0')
plt.title('Height distribution (Female)')
ax0.axis(axis_vals)
ax0.grid(True)
level = .399 / heights1.std()
ax0.plot([heights1.mean(), heights1.mean()],[0, level],'b-o')
ax0.plot([heights1.mean()-heights1.std(), heights1.mean()+heights1.std()],[level*scale, level*scale],'b-o')
mu1, std1 = heights1.mean(),heights1.std()
x_points,gaussian_wind = Gaussian_window(mu1,std1)
ax0.plot(x_points,gaussian_wind,'o--')

ax0.text(70,.13,'mu = ' + str(np.round(heights1.mean(),decimals=1)) + ", std = "
         + str(np.round(heights1.std(),decimals=2))
         + ", w = " + str(np.round(1.0*len(heights1)/(len(heights1)+len(heights2)),decimals=2)),bbox=dict(facecolor='red',alpha=0.5))
plt.xlabel('Height (cm)')

ax1 = fig.add_subplot(2,2,2)
sns.distplot(heights2,label='Heights2',bins=range(min(heights2),max(heights2)),color='#008000')
plt.title('Height distribution (Male)')
mu2, std2 = heights2.mean(),heights2.std()
x_points,gaussian_wind = Gaussian_window(mu2,std2)
ax1.plot(x_points,gaussian_wind,'ro--')

ax1.text(55,.13,'mu = ' + str(np.round(heights2.mean(),decimals=1))
         + ", std = " + str(np.round(heights2.std(),decimals=2))
         + ", w = " + str(np.round(1.0*len(heights2)/(len(heights1)+len(heights2)),decimals=2)),
         bbox=dict(facecolor='red',alpha=0.5))
ax1.axis(axis_vals)
ax1.grid(True)
level = .399 / heights2.std()
ax1.plot([heights2.mean(), heights2.mean()],[0, level],'r-o')
ax1.plot([heights2.mean()-heights2.std(), heights2.mean()+heights2.std()],[level*scale, level*scale],'r-o')
plt.xlabel('Height (cm)')

ax = fig.add_subplot(2,1,2)
ax.axis(axis_vals)

line1, = ax.plot([MU1_INIT, MU1_INIT],[0, heights1_max_hist],'bo--')
line2, = ax.plot([MU1_INIT-STD1_INIT, MU1_INIT+STD1_INIT],[heights1_max_hist*scale, heights1_max_hist*scale],'bo--')
x_points,gaussian_wind = Gaussian_window(MU1_INIT,STD1_INIT,.5)
g_curve1, = ax.plot(x_points,gaussian_wind,'bo--')


line3, = ax.plot([MU2_INIT, MU2_INIT],[0, heights2_max_hist],'ro--')
line4, = ax.plot([MU2_INIT-STD2_INIT, MU2_INIT+STD2_INIT],[heights2_max_hist*scale, heights2_max_hist*scale],'ro--')
x_points,gaussian_wind = Gaussian_window(MU2_INIT,STD2_INIT,.5)
g_curve2, = ax.plot(x_points,gaussian_wind,'ro--')


iter_template = 'Iteration = %d\nmu1 = %.1f, std1 = %.2f, w1 = %.2f\nmu2 = %.1f, std2 = %.2f, w2 = %.2f'
iter_text = ax.text(0,.8,'',transform=ax.transAxes,fontsize=16,bbox=dict(facecolor='red',alpha=0.5))

raw_input('Press return to start animation')

ani = animation.FuncAnimation(fig,
                              simPoints,
                              simData,
                              init_func=init,
                              interval=2000,
                              repeat_delay=8000,
                              repeat=True)
    
plt.show()    
