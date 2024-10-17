import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

#initial conditions and parameters
y0_list = np.array([[1, 1], [1, 3], [1.5, 2], [0.5, 1], [0.5, 0.5]])
a = 1
b_list = [0.5, 1.5, 2, 3]

# initial anf final time used in numerical solutions
t0 = 0
tf = 30
#10000 might seem like is a bit much, but the plot looks better
t = np.linspace(t0, tf, 10000)

# axis limits used in plot_different_b() and plot_diferent_init()
xlim = [0, 4]
ylim = [0, 5]

# range used in plot_stream()
stream_range = [0, 5]


def differential(t, y0, b):
    x, y = y0
    dxdt = 1 - (1 + b)*x + a*x**2*y
    dydt = b*x - a*x**2*y
    return dxdt, dydt


def nullclines(b, x=np.linspace(-10, 10, 1000)):
    dx_null = ((1 + b)*x - 1) / (a*x**2)
    dy_null = b / (a*x)
    return x, dx_null, dy_null


# plots streamplot and nullclines for different values of b
def plot_streams(b_list=b_list, save=False):
    fig, axes = plt.subplots(2, 2, figsize=[10, 10], dpi=600) 
    axes = axes.flatten()
    x, y = np.meshgrid(np.linspace(stream_range[0], stream_range[1], 100), np.linspace(stream_range[0], stream_range[1], 100))
    for i in range(len(b_list)):
        b = b_list[i]
        f = lambda t, y0: differential(t, y0, b)
        x0, dx_null, dy_null = nullclines(b)
        dxdt, dydt = f(0, [x, y])
        axes[i].streamplot(x, y, dxdt, dydt, color='lightgrey', density=2)
        axes[i].plot(x0, dx_null, c='r', label='x nullcline')
        axes[i].plot(x0, dy_null, c='b', label='y nullcline')
        axes[i].plot(1, b/a, color='k', marker='o')
        axes[i].set_xlabel(r"dx/dt ($mol \cdot cm^{-3} \cdot s^{-1}$)")
        axes[i].set_ylabel(r"dy/dt ($mol \cdot cm^{-3} \cdot s^{-1}$)")
        axes[i].tick_params(axis='both', which='both', direction='in')  
        axes[i].minorticks_on() 
        axes[i].legend(loc='upper right')
        axes[i].set_xlim(stream_range)
        axes[i].set_ylim(stream_range)
        axes[i].title.set_text(f"a = 1, b = {b}")
    plt.tight_layout()
    if save:
        plt.savefig('Fig1.svg')
    plt.show()
        

# plots x(t) and y(t) for different values of b
def plot_different_b(b_list=b_list, save=False):
    fig, axes = plt.subplots(2, 2, figsize=[10, 10], dpi=600) 
    axes = axes.flatten()
    for i in range(len(b_list)):
        b = b_list[i]
        f = lambda t, y0: differential(t, y0, b)
        results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=y0_list[0], method="RK45", t_eval=t)
        x = results.y[0]
        y = results.y[1] 
        axes[i].plot(t, x, c='r', label='x(t)')
        axes[i].plot(t, y, c='b', label='y(t)')
        axes[i].set_xlabel("Time ($s$)")
        axes[i].set_ylabel(r"Concentration ($mol \cdot cm^{-3}$)")
        axes[i].tick_params(axis='both', which='both', direction='in')  
        axes[i].minorticks_on() 
        axes[i].legend(loc='upper right')
        axes[i].set_xlim([t0,tf])
        axes[i].title.set_text(f"a = 1, b = {b}")    
    plt.tight_layout()
    if save:
        plt.savefig('Fig2.svg')
    plt.show()


# plots phasespace of x and y for different values of b and initial conditions
def plot_different_init(b_list=b_list, y0_list=y0_list, save=False):
    fig, axes = plt.subplots(2, 2, figsize=[10, 10], dpi=600) 
    axes = axes.flatten()
    for i in range(len(b_list)):
        b = b_list[i]
        x_null, dx_null, dy_null = nullclines(b)
        axes[i].plot(x_null, dx_null, color='lightgrey', ls='--')
        axes[i].plot(x_null, dy_null, color='lightgrey', ls='--')
        for j in range(len(y0_list)):
            y0 = y0_list[j]
            f = lambda t, y0: differential(t, y0, b)
            results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
            x = results.y[0]
            y = results.y[1]
            axes[i].plot(x, y, label=f'$x_0$ = {y0[0]}, $y_0$ = {y0[1]}')
            axes[i].set_xlabel(r"x ($mol \cdot cm^{-3}$)")
            axes[i].set_ylabel(r"y ($mol \cdot cm^{-3}$)")
            axes[i].tick_params(axis='both', which='both', direction='in')  
            axes[i].minorticks_on() 
            axes[i].legend(loc='upper right')
            axes[i].title.set_text(f"a = 1, b = {b}")
            axes[i].set_xlim(xlim)
            axes[i].set_ylim(ylim)
    plt.tight_layout()
    if save:
        plt.savefig('Fig3.svg')
    plt.show()



def main():
    plot_streams(save=True)
    plot_different_b(save=True)
    plot_different_init(save=True)
    
    

if __name__ == "__main__":
    main()