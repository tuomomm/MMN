# MMN
A network of integrate and fire model neurons that provides a plausible mechanism for the deviant detection in the frequency and omission mismatch negativity (MMN) protocols.

The python scripts presented here provide a Brian2-simulator-based model of deviance detection in the MMN protocols.
In the simulation scripts, a network model consisting of four populations (a "timer" population that is phase-locked to a delta oscillation assumed to be available in the brain, two populations sensitive to auditory stimuli at two different stimulus frequencies, and an output population that fires in response to a deviant) of integrate-and-fire model neurons is created. The neurons are connected to each other by conductance-based synaptic currents (as described in Wang 1999, PMC6782911). The population responding to the standard frequency sends excitatory inputs to the output population and inhibitory inputs to the phase-locked population (this is a simplification of a feed-forward inhibition). The output population receives excitatory inputs from the phase-locked population and the population responsive to the deviant frequency. We have tuned the AMPAR-, NMDAR- and GABABR-mediated synaptic conductances between these population so that the output population fires in reponse to the first tones of each sequence, deviant tones and omitted tones.

The following scripts simulate the deviant detection in different MMN protocols (all models equal with an equal phase-locked stimulus, only the auditory stimulus protocols are different):
Full_Simulation_MMN_differentPitch.py: A protocol consisting of 3 standards, 1 deviant, and 4 standards (500 ms interval)
Full_Simulation_MMN_completeOmission.py: A protocol consisting of 3 standards (500 ms interval), omitted stimulus (1 sec interval), and 4 standards (500 ms interval)
Full_Simulation_MMN_soonerAndDifferentPitch.py: A protocol where the 4th standard comes sooner than normally and there's a deviant at the expted time of the 4th stimulus, followed by 4 standards

In all three protocols, the output population fires in response to the first stimulus and the omitted or the deviant stimulus, while the response to the repeated standards is attenuated.

MIT License
(C) Tuomo Maki-Marttunen, Ahmed Eissa