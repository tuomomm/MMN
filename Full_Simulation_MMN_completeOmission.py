from brian2 import *
from pylab import *
import scipy.io


Nperpop = 100
gAMPA1 = 25.0
gAMPA2 = 25.0
gNMDA1 = 1.0
gNMDA2 = 1.0
gGABA = 30
tauD = 700
pv = 0.5
myseed = 1
# Parameters

if len(sys.argv) > 1:
    Nperpop = int(sys.argv[1])
if len(sys.argv) > 2:
    gAMPA1 = float(sys.argv[2])
if len(sys.argv) > 3:
    gAMPA2 = float(sys.argv[3])
if len(sys.argv) > 4:
    gNMDA1 = float(sys.argv[4])
if len(sys.argv) > 5:
    gNMDA2 = float(sys.argv[5])
if len(sys.argv) > 6:
    gGABA = float(sys.argv[6])
if len(sys.argv) > 7:
    tauD = int(sys.argv[7])


seed(myseed)
np.random.seed(myseed)
myseedAdd = '' if myseed == 1 else '_seed'+str(myseed)

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    f,axarr = subplots(1,2)
    axarr[0].plot(zeros(Ns), arange(Ns), 'ok', ms=0.8,mew=0.2)
    axarr[0].plot(ones(Nt), arange(Nt), 'ok', ms=0.8,mew=0.2)
    for i, j in zip(S.i, S.j):
        axarr[0].plot([0, 1], [i, j], '-b',lw=0.1)
    axarr[0].set_xticks([0, 1], ['Source', 'Target'])
    axarr[0].set_ylabel('Neuron index')
    axarr[0].set_xlim(-0.1, 1.1)
    axarr[0].set_ylim(-1, max(Ns, Nt))

    axarr[1].plot(S.i, S.j, 'ok',ms=0.8,mew=0.2)
    axarr[1].set_xlim(-1, Ns)
    axarr[1].set_ylim(-1, Nt)
    axarr[1].set_xlabel('Source neuron index')
    axarr[1].set_ylabel('Target neuron index')

    f.savefig("brian2_connectivity.eps")

start_scope()

#mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (MgCon / 3.57 (mM)))

eqs = '''
dv/dt = (stimulus2(t)*stimcoeff2+stimulus3(t)*stimcoeff3+stimulus(t)*stimcoeff+ g_leak*(e_leak-v) +s_nmdatot*(0-v)/(1+exp(-0.062*v)/3.57)  +s_ampatot*(0-v) +s_ampatot2*(0-v) + s_nmdatot2*(0-v)/(1+exp(-0.062*v)/3.57) +s_gabatot*(-90-v)+ s_gabatot2*(-90-v) + s_ampatot3*(0-v) + s_nmdatot3*(0-v)/(1+exp(-0.062*v)/3.57) + s_gabatot3*(-90-v))/tau : 1 (unless refractory)
I : 1
tau : second
s_ampatot : 1
s_nmdatot : 1
s_gabatot : 1
s_ampatot2 : 1
s_nmdatot2 : 1
s_gabatot2 : 1
s_ampatot3 : 1
s_nmdatot3 : 1
s_gabatot3: 1
stimcoeff: 1
stimcoeff2: 1
stimcoeff3: 1
e_leak : 1
g_leak : 1
'''


def create_synapse_equations(suffix='ampatot'):
    return f'''
    ds_ampa/dt = alpha_s*x_ampa*(1-s_ampa)-s_ampa/tau_ampa : 1
    dx_ampa/dt = -x_ampa/tau_x_ampa : 1
    ds_nmda/dt = alpha_s*x_nmda*(1-s_nmda)-s_nmda/tau_nmda : 1
    dx_nmda/dt = -x_nmda/tau_x_nmda : 1
    ds_gaba/dt = -s_gaba/tau_gaba : 1
    dD_glut/dt = (1-D_glut)/tau_D_glut : 1
    s_ampat{suffix}_post = g_ampa*s_ampa : 1 (summed)
    s_nmdat{suffix}_post = g_nmda*s_nmda : 1 (summed)
    s_gabat{suffix}_post = g_gaba*s_gaba : 1 (summed)
    tau_ampa : second
    tau_x_ampa : second
    tau_nmda : second
    tau_x_nmda : second
    tau_gaba : second
    tau_D_glut : second
    alpha_s : 1/second
    alpha_ampa : 1
    alpha_nmda : 1
    alpha_gaba : 1
    pv : 1
    g_ampa : 1
    g_nmda : 1
    g_gaba : 1
    '''
'''
this function exist because Brian does not allow for 
the connection of several neurons for the same summed variables
'''

eq_syn1 = create_synapse_equations('ot')
eq_syn2 = create_synapse_equations('ot2')
eq_syn3 = create_synapse_equations('ot3')

eq_syn_on_pre = '''
x_ampa += alpha_ampa*D_glut
x_nmda += alpha_nmda*D_glut
s_gaba += alpha_gaba
D_glut += -pv*D_glut
'''

'''
stimulus:responsible for the standard signal
stimulus:manage the MMN
stimulus2: responsible for the oddtone singal
'''
stimulus = TimedArray(array([0]*49+[120]+[0]*49+[120]+[0]*49+[120]+[0]*49+[0]+[0]*49+[120]+[0]*49+[120]+[0]*49+[120]+[0]*49+[120]+[0]*1000), dt=10*ms)
stimulus2 = TimedArray(array([0]*49+[100]+[0]*49+[100]+[0]*49+[100]+[0]*49+[100]+[0]*49+[100]+[0]*49+[100]+[0]*49+[100]+[0]*49+[100]+[0]*1000), dt=10*ms)
stimulus3 = TimedArray(array([0]*49+[0]+[0]*49+[0]+[0]*49+[0]+[0]*49+[0]+[0]*49+[0]+[0]*49+[0]+[0]*49+[0]+[0]*49+[0]+[0]*1000), dt=10*ms)

'''
standardSignalPopulation:    population responsible for generating the standard signal
outputPopulation:            population that displays the output
MMNManagerPopulation:        population responsible for managing the MMN
oddtoneSignalPopulation:     population responsible for generating the oddtone signal
'''
standardSignalPopulation = NeuronGroup(2*Nperpop, eqs, threshold='v>-40', reset='v = -80', refractory=5*ms, method='rk4')
outputPopulation = NeuronGroup(Nperpop, eqs, threshold='v>-40', reset='v = -80', refractory=5*ms, method='rk4')
MMNManagerPopulation = NeuronGroup(Nperpop, eqs, threshold='v>-40', reset='v = -80', refractory=5*ms, method='rk4') 
oddtoneSignalPopulation = NeuronGroup(Nperpop, eqs, threshold='v>-40', reset='v = -80', refractory=5*ms, method='rk4') 

# Configuration for standardSignalPopulation
standardSignalPopulation.v = -80
standardSignalPopulation.tau = [10]*(2*Nperpop)*ms
standardSignalPopulation.stimcoeff = [1]*Nperpop+[1]*Nperpop
standardSignalPopulation.g_leak = 2.0
standardSignalPopulation.e_leak = -80

# Configuration for outputPopulation
outputPopulation.v = -80
outputPopulation.tau = [10]*(Nperpop)*ms
outputPopulation.g_leak = 2.0
outputPopulation.e_leak = -80

# Configuration for MMNManagerPopulation
MMNManagerPopulation.v = -80
MMNManagerPopulation.tau = [10]*(Nperpop)*ms
MMNManagerPopulation.stimcoeff2 = [1]*Nperpop
MMNManagerPopulation.g_leak = 2.0
MMNManagerPopulation.e_leak = -80

# Configuration for oddtoneSignalPopulation
oddtoneSignalPopulation.v = -80
oddtoneSignalPopulation.tau = [10]*(Nperpop)*ms
oddtoneSignalPopulation.stimcoeff3 = [0]*Nperpop
oddtoneSignalPopulation.g_leak = 2.0
oddtoneSignalPopulation.e_leak = -80

# Synapses configurations
standardToOutputSynapses = Synapses(standardSignalPopulation, outputPopulation, model =eq_syn1, on_pre = eq_syn_on_pre)
MMNManagerToOutputSynapses = Synapses(MMNManagerPopulation, outputPopulation, model = eq_syn2, on_pre = eq_syn_on_pre)
standardToMMNManagerSynapses = Synapses(standardSignalPopulation, MMNManagerPopulation, model = eq_syn1, on_pre = eq_syn_on_pre)
oddtoneToOutputSynapses = Synapses(oddtoneSignalPopulation, outputPopulation, model =eq_syn3, on_pre = eq_syn_on_pre)


# Matrix A
A = np.c_[np.random.rand(Nperpop, Nperpop) < 0.5, np.random.rand(Nperpop, Nperpop) < 0.5]

# Matrix B
B = np.random.rand(Nperpop, Nperpop) < 0.5


taus = []
g_ampas = []
g_nmdas = []
g_gabas = []

g_ampas2 = []
g_nmdas2 = []
g_gabas2= []

g_ampas3 = []
g_nmdas3 = []
g_gabas3= []

g_ampas4 = []
g_nmdas4 = []
g_gabas4= []


'''
The following two functions show the circuit connections (all connections have a depressing effect when exciting):

standardToOutputSynapses:           standardSignalPopulation Excites outputPopulation
MMNManagerToOutputSynapses:        MMNManagerPopulation Excites outputPopulation
standardToMMNManagerSynapses:      standardSignalPopulation Inhibits MMNManagerPopulation
oddtoneToOutputSynapses:           oddtoneSignalPopulation Excites outputPopulation
'''


def connect_synapses_block0_to_B(synapse, g_ampas, g_nmdas, g_gabas, gAMPA1, gNMDA1, gAMPA2, gNMDA2, gGABA=None):
    for iy in range(0, Nperpop): # From block 0 of A
        for ix in range(0, Nperpop): # To B
            if A[iy, ix]:
                synapse.connect(i=iy, j=ix)
                g_ampas.append(gAMPA1 / Nperpop if gGABA is None else 0.0)
                g_nmdas.append(gNMDA1 / Nperpop if gGABA is None else 0.0)
                g_gabas.append(0.0 if gGABA is None else gGABA / Nperpop)

def connect_synapses_block1_to_B(synapse, g_ampas, g_nmdas, g_gabas, gAMPA1, gNMDA1, gAMPA2, gNMDA2, gGABA=None):
    for iy in range(0, Nperpop):  # Iterating over the rows of A (both blocks have the same number of rows)
        for ix in range(0, Nperpop):  # Iterating over the columns of B
            if A[iy, ix ]:  # Checking the connection from the second block of A
                synapse.connect(i=iy+Nperpop, j=ix)  # Adjusting the connection to map to B
                g_ampas.append(gAMPA1 / Nperpop if gGABA is None else 0.0)
                g_nmdas.append(gNMDA1 / Nperpop if gGABA is None else 0.0)
                g_gabas.append(0.0 if gGABA is None else gGABA / Nperpop)



def connect_synapses_B_to_B(synapse, g_ampas, g_nmdas, g_gabas, gAMPA1, gNMDA1, gAMPA2, gNMDA2, gGABA=None):
    for iy in range(0, Nperpop):  # From Matrix B
        for ix in range(0, Nperpop):  # To Matrix B
            if B[iy, ix]:
                synapse.connect(i=iy, j=ix)
                g_ampas.append(gAMPA1 / Nperpop if gGABA is None else 0.0)
                g_nmdas.append(gNMDA1 / Nperpop if gGABA is None else 0.0)
                g_gabas.append(0.0 if gGABA is None else gGABA / Nperpop)


                    
# Use the updated synapse names in the connection and initialization functions
connect_synapses_block0_to_B(standardToOutputSynapses, g_ampas, g_nmdas, g_gabas, gAMPA1, gNMDA1, gAMPA2, gNMDA2)
connect_synapses_B_to_B(MMNManagerToOutputSynapses, g_ampas2, g_nmdas2, g_gabas2, gAMPA1, gNMDA1, gAMPA2, gNMDA2)
connect_synapses_block1_to_B(standardToMMNManagerSynapses, g_ampas3, g_nmdas3, g_gabas3, 0, 0, 0, 0, gGABA)
connect_synapses_B_to_B(oddtoneToOutputSynapses, g_ampas4, g_nmdas4, g_gabas4, gAMPA1, gNMDA1, gAMPA2, gNMDA2)


tstop = 5000

def initialize_synapse(S, tauD, pv, g_ampas, g_nmdas, g_gabas):
    S.tau_x_ampa = 0.5*msecond   # AMPA rise time constant
    S.tau_ampa = 5*msecond       # AMPA decay time constant
    S.tau_x_nmda = 2*msecond     # NMDA rise time constant
    S.tau_nmda = 50*msecond      # NMDA decay time constant
    S.tau_gaba = 10*msecond      # GABA decay time constant
    S.tau_D_glut = tauD*msecond  # Presynaptic depression time constant
    S.alpha_ampa = 1.0
    S.alpha_nmda = 1.0
    S.alpha_gaba = 1.0
    S.alpha_s = 1.0/msecond # x to s coupling strength
    S.pv = pv
    S.g_ampa = g_ampas
    S.g_nmda = g_nmdas
    S.g_gaba = g_gabas
    S.s_ampa = 0
    S.x_ampa = 0
    S.s_nmda = 0
    S.x_nmda = 0
    S.s_gaba = 0
    S.D_glut = 1
    


initialize_synapse(standardToOutputSynapses, tauD, pv, g_ampas, g_nmdas, g_gabas)
initialize_synapse(MMNManagerToOutputSynapses, tauD, pv, g_ampas2, g_nmdas2, g_gabas2)
initialize_synapse(standardToMMNManagerSynapses, tauD, pv, g_ampas3, g_nmdas3, g_gabas3)
initialize_synapse(oddtoneToOutputSynapses, tauD, pv, g_ampas4, g_nmdas4, g_gabas4)

V_outputPopulation = StateMonitor(outputPopulation, 'v', record=True)
outputPopulationSpikeMonitor = SpikeMonitor(outputPopulation)
MMNManagerPopulationSpikeMonitor = SpikeMonitor(MMNManagerPopulation)
standardSignalPopulationSpikeMonitor = SpikeMonitor(standardSignalPopulation)
oddtonePopulationSpikeMonitor = SpikeMonitor(oddtoneSignalPopulation) 

run(tstop*ms)

# Visualizing synapse connectivity
visualise_connectivity(standardToOutputSynapses)

# Plotting the spikes for standardSignalPopulationSpikeMonitor
fig1, ax = subplots(3,2)
axarr = [ax[0,0],ax[0,1],ax[1,0],ax[1,1],ax[2,0],ax[2,1]]
for iax in range(0,len(axarr)):
  axarr[iax].tick_params(axis='both', which='major', labelsize=4)
  axarr[iax].set_xlim([0,5050])
  xiax = iax%2
  yiax = int(iax/2)
  axarr[iax].set_position([0.08+0.47*xiax, 0.72-0.32*yiax, 0.41, 0.26])
  for axis in ['top','bottom','left','right']:
    axarr[iax].spines[axis].set_linewidth(0.5)

stimvec = [stimulus,stimulus2,stimulus3]
for stimind in [0,1,2]:
    thisstim = stimvec[stimind]
    lastval = 0
    dt = thisstim.dt*1000
    vals = thisstim.values
    for itime in range(0,len(vals)):
        axarr[0].plot([itime*dt,itime*dt,(itime+1)*dt],[lastval-150*stimind,vals[itime]-150*stimind,vals[itime]-150*stimind],lw=0.5)
        lastval = vals[itime]
    
axarr[1].plot(standardSignalPopulationSpikeMonitor.t/msecond, array(standardSignalPopulationSpikeMonitor.i), 'b.', lw=0.5, ms=0.5, mew=0.5)
axarr[1].set_ylim([0, 2 * Nperpop])
axarr[1].set_title("standardSignalPopulationSpikeMonitor",fontsize=5)
axarr[1].set_xlabel("Time (ms)",fontsize=5)
axarr[1].set_ylabel("Standard Population Spike Monitor",fontsize=5)

# Plotting the spikes for MMNManagerPopulationSpikeMonitor
axarr[2].plot(MMNManagerPopulationSpikeMonitor.t/msecond, array(MMNManagerPopulationSpikeMonitor.i), 'b.', lw=0.5, ms=0.5, mew=0.5)
axarr[2].set_ylim([0, Nperpop])
axarr[2].set_title("MMNManagerPopulationSpikeMonitor",fontsize=5)
axarr[2].set_xlabel("Time (ms)",fontsize=5)
axarr[2].set_ylabel("MMN Manager Population Spike Monitor",fontsize=5)

# Plotting the spikes for outputPopulationSpikeMonitor
axarr[3].plot(outputPopulationSpikeMonitor.t/msecond, array(outputPopulationSpikeMonitor.i), 'b.', lw=0.5, ms=0.5, mew=0.5)
axarr[3].set_ylim([0, Nperpop])
axarr[3].set_title("outputPopulationSpikeMonitor",fontsize=5)
axarr[3].set_xlabel("Time (ms)",fontsize=5)
axarr[3].set_ylabel("Output Population Spike Monitor",fontsize=5)

# Plotting the spikes for oddtonePopulationSpikeMonitor
axarr[4].plot(oddtonePopulationSpikeMonitor.t/msecond, array(oddtonePopulationSpikeMonitor.i), 'b.', lw=0.5, ms=0.5, mew=0.5)
axarr[4].set_ylim([0, Nperpop])
axarr[4].set_title("oddtonePopulationSpikeMonitor",fontsize=5)
axarr[4].set_xlabel("Time (ms)",fontsize=5)
axarr[4].set_ylabel("Oddtone Population Spike Monitor",fontsize=5)

# Plotting the V_outputPopulation
axarr[5].plot(V_outputPopulation.t/msecond, V_outputPopulation.v[0], label="Neuron 0")
axarr[5].set_title("V_outputPopulation",fontsize=5)
axarr[5].set_xlabel("Time (ms)",fontsize=5)
axarr[5].set_ylabel("Voltage (V)",fontsize=5)
axarr[5].legend(fontsize=5)

fig1.savefig('brian2_threepops6_completeOmission_Nperpop'+str(Nperpop)+'_gAMPA'+str(gAMPA1)+'-'+str(gAMPA2)+'_gNMDA'+str(gNMDA1)+'-'+str(gNMDA2)+'_gGABA'+str(gGABA)+'_'+str(tauD)+'_figs.pdf')
