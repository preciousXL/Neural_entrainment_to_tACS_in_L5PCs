import multiprocessing

import numpy as np
import neuron
from neuron import h
import time, os
import subprocess
import pickle, glob
import scipy.signal
from scipy.signal import find_peaks
import copy

def calc_uniformE_nonUnitEvec_direction(allSectionLists, Evec):
    Ex, Ey, Ez = Evec[0], Evec[1], Evec[2]
    for sec in allSectionLists:
        if h.ismembrane('xtra', sec=sec) and h.ismembrane('extracellular', sec=sec):
            for seg in sec:
                seg.es_xtra = -(Ex*seg.x_xtra + Ey*seg.y_xtra + Ez*seg.z_xtra) * 1e-3

def getRotation_RotVecDegree(vec0, vec1):
    if type(vec0) == 'list':
        vec0 = np.array(vec0)
    if type(vec1) == 'list':
        vec1 = np.array(vec1)
    vec0_norm = vec0 / np.linalg.norm(vec0)
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vecAx = np.cross(vec0, vec1)
    vecAx_norm = vecAx / np.linalg.norm(vecAx)
    angle = np.arccos(np.dot(vec0_norm, vec1_norm))
    return vecAx_norm, angle

def getRotationMatrix(vec0=None, vec1=None, vecAx=None, angle=None):
    # https://en.wikipedia.org/wiki/Rotation_matrix
    if vecAx==None and angle==None:
        vecAx, angle = getRotation_RotVecDegree(vec0, vec1)
    if type(vecAx) == 'list':
        vecAx = np.array(vecAx)
    s = np.sin(angle)
    c = np.cos(angle)
    t = 1 - c
    vecAx = vecAx / np.linalg.norm(vecAx)
    x, y, z = vecAx
    rotMatrix = np.array([[t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
                          [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
                          [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]])
    return rotMatrix

def calc_ROISurfEvec_to_OriginEvec(elmEvecs, elmNormals):
    num_elem    = elmEvecs.shape[0]
    originEvecs = np.zeros((num_elem, 3))
    for i in range(num_elem):
        Evec, normal = elmEvecs[i,:], elmNormals[i,:]
        rotMatrix_normal = getRotationMatrix(vec0=normal, vec1=[0, 0, 1])
        originEvecs[i, :] = np.dot(rotMatrix_normal, elmEvecs[i, :])
    return originEvecs

def calcSpikeTime_spikeIndex_FiringRate_spikeNumber_fromVth(vsoma, tvar, Vth=0.0):
    logicArray = vsoma >= 0.0
    logicArray_binary = logicArray.astype(np.int32)
    logicArray_diff = np.hstack((0, np.diff(logicArray_binary)))
    spikeIndex = logicArray_diff > 0.0
    spikeTime = tvar[spikeIndex]
    spikeNumber = np.sum(spikeIndex)
    firingRate = 1000 * spikeNumber / (tvar[-1] - tvar[0])
    spikeIndex = np.nonzero(spikeIndex)[0]
    return spikeTime, spikeIndex, firingRate, spikeNumber


def calc_field_spike_entrainment(vsoma=np.zeros(10), Evar=np.zeros(10), tvar=np.zeros(10), Vth=0.0):
    analyticSignal = scipy.signal.hilbert(Evar)
    tacsInstantaneousPhase = np.angle(analyticSignal)
    spikeTime, spikeIndex, firingRate, spikeNumber = calcSpikeTime_spikeIndex_FiringRate_spikeNumber_fromVth(vsoma, tvar, Vth=Vth)
    spikePhaseRadian = tacsInstantaneousPhase[spikeIndex]
    spikePhaseRadian[spikePhaseRadian < 0 ] += 2*np.pi
    pluralPLV = np.mean(np.exp(1j*spikePhaseRadian))
    spikePhaseDegree = np.rad2deg(spikePhaseRadian)
    fieldSpikePLV = np.abs(pluralPLV)

    return fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber


def calc_uniformE_theta_phi(allSectionLists, theta=0, phi=0):
    theta = theta * np.pi / 180
    phi = phi * np.pi / 180
    Ex = np.sin(theta) * np.cos(phi)
    Ey = np.sin(theta) * np.sin(phi)
    Ez = np.cos(theta)
    for sec in allSectionLists:
        if h.ismembrane('xtra', sec=sec) and h.ismembrane('extracellular', sec=sec):
            for seg in sec:
                seg.es_xtra = -(Ex * seg.x_xtra + Ey * seg.y_xtra + Ez * seg.z_xtra) * 1e-3


class Cell():
    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.list_cell_model_name = ['L1_NGC-DA_bNAC219_1', 'L1_NGC-DA_bNAC219_2', 'L1_NGC-DA_bNAC219_3',
                                     'L1_NGC-DA_bNAC219_4', 'L1_NGC-DA_bNAC219_5',
                                     'L23_PC_cADpyr229_1', 'L23_PC_cADpyr229_2', 'L23_PC_cADpyr229_3',
                                     'L23_PC_cADpyr229_4', 'L23_PC_cADpyr229_5',
                                     'L4_LBC_cACint209_1', 'L4_LBC_cACint209_2', 'L4_LBC_cACint209_3',
                                     'L4_LBC_cACint209_4', 'L4_LBC_cACint209_5',
                                     'L5_TTPC2_cADpyr232_1', 'L5_TTPC2_cADpyr232_2', 'L5_TTPC2_cADpyr232_3',
                                     'L5_TTPC2_cADpyr232_4', 'L5_TTPC2_cADpyr232_5',
                                     'L6_TPC_L4_cADpyr231_1', 'L6_TPC_L4_cADpyr231_2', 'L6_TPC_L4_cADpyr231_3',
                                     'L6_TPC_L4_cADpyr231_4', 'L6_TPC_L4_cADpyr231_5']

        self.list_NSTACK_size = [100000, 100000, 100000, 100000, 100000, \
                                 10000, 100000, 100000, 10000, 100000, \
                                 100000, 100000, 100000, 100000, 100000, \
                                 100000, 100000, 100000, 100000, 100000, \
                                 100000, 100000, 100000, 10000, 10000]
        self.cell_model_name = self.list_cell_model_name[self.cell_id - 1]
        self.NSTACK_size = self.list_NSTACK_size[self.cell_id - 1]

        self.create_cell()
        # for sec in h.allsec():
        #     if h.ismembrane('xtra', sec=sec):
        #         sec.uninsert('xtra')
        self.allSections = [sec for sec in neuron.h.cell.all]
        self.allSegments = [seg for sec in neuron.h.cell.all for seg in sec]
        self.get_cell_coordinates()
        self.get_cell_segment_coordinates()
        self.create_recordings()

    def create_cell(self):
        h.load_file("nrngui.hoc")
        h.load_file("init_snowp.hoc")
        h.cell_id = self.cell_id
        h.cell_model_name = self.cell_model_name
        h.NSTACK_size = self.NSTACK_size
        h.loadFiles()
        h.cell_chooser()
        h.load_file("steadystate_init.hoc")
        h.load_file("getes_snowp.hoc")

    def get_cell_coordinates(self):
        numSec = len(self.allSections)
        self.cell_coordinates = np.full((3, 3 * numSec), np.nan)  # shape=(3=x,y,z, numSec*3=xstart,xend,nan)
        self.cell_section_coordinates = np.zeros((numSec, 3))
        self.cell_section_coordinates_start = np.zeros((numSec, 3))
        self.cell_section_coordinates_end = np.zeros((numSec, 3))
        for i, sec in enumerate(self.allSections):
            n3d = int(neuron.h.n3d(sec=sec))
            sec_start = np.array([neuron.h.x3d(0, sec=sec), neuron.h.y3d(0, sec=sec), neuron.h.z3d(0, sec=sec)])
            sec_end = np.array(
                [neuron.h.x3d(n3d - 1, sec=sec), neuron.h.y3d(n3d - 1, sec=sec), neuron.h.z3d(n3d - 1, sec=sec)])
            self.cell_section_coordinates_start[i, :] = sec_start
            self.cell_section_coordinates_end[i, :] = sec_end
            self.cell_coordinates[:, i * 3] = sec_start
            self.cell_coordinates[:, i * 3 + 1] = sec_end
            self.cell_section_coordinates[i, :] = (sec_start + sec_end) / 2

    def get_cell_segment_coordinates(self):
        numSeg = np.array([sec.nseg for sec in self.allSections])
        numSeg = np.sum(numSeg)
        self.cell_segment_coordinates = np.zeros((numSeg, 3))
        num = 0
        for sec in self.allSections:
            for seg in sec:
                self.cell_segment_coordinates[num, :] = np.array([seg.x_xtra, seg.y_xtra, seg.z_xtra])
                num += 1

    def get_section_center(self, sec):
        n3d = int(neuron.h.n3d(sec=sec))
        sec_start = np.array([neuron.h.x3d(0, sec=sec), neuron.h.y3d(0, sec=sec), neuron.h.z3d(0, sec=sec)])
        sec_end = np.array(
            [neuron.h.x3d(n3d - 1, sec=sec), neuron.h.y3d(n3d - 1, sec=sec), neuron.h.z3d(n3d - 1, sec=sec)])
        center = (sec_start + sec_end) / 2
        return center

    def create_recordings(self):
        self.recordings = {}
        self.recordings['t'] = neuron.h.Vector().record(neuron.h._ref_t)
        self.recordings['soma(0.5)'] = neuron.h.Vector().record(neuron.h.cell.soma[0](0.5)._ref_v)

    def add_synapse_L5PC_Clone1(self, thresh=10, delay=0., weight=0.004, secindex=4, x=0.5):
        # Define the presynaptic Poisson discharge sequence
        self.spikesource = neuron.h.NetStim()
        self.spikesource.interval = 1000/50  # ms (mean) time between spikes 
        self.spikesource.number   = int(1e9) # (average) number of spikes 
        self.spikesource.start    = 1000     # ms (mean) start time of first spike (The first 1000 milliseconds are used for the membrane potential to reach the resting value.)
        self.spikesource.noise    = 1        # range 0 to 1. Fractional randomness. 0 deterministic, 1 intervals have decaying exponential distribution
        self.spikesource.seed(1)
        # Define the double-exponential synaptic input
        self.synapse = neuron.h.Exp2Syn(neuron.h.cell.apic[secindex](x)) # pyramidal neuron for apical dendrites
        self.synapse.tau1 = 2
        self.synapse.tau2 = 10
        self.synapse.e    = 0
        # Connect the presynaptic discharge sequence with the synaptic connection.
        self.connection = neuron.h.NetCon(self.spikesource, self.synapse, thresh, delay, weight)
        self.connection.delay = delay
        self.connection.weight[0] = weight
        
    def add_synapse_L5PC_Clone2(self, thresh=10, delay=0., weight=0.004, secindex=3, x=0.5):
        # Define the presynaptic Poisson discharge sequence
        self.spikesource = neuron.h.NetStim()
        self.spikesource.interval = 1000/50  # ms (mean) time between spikes 
        self.spikesource.number   = int(1e9) # (average) number of spikes 
        self.spikesource.start    = 1000     # ms (mean) start time of first spike (The first 1000 milliseconds are used for the membrane potential to reach the resting value.)
        self.spikesource.noise    = 1        # range 0 to 1. Fractional randomness. 0 deterministic, 1 intervals have decaying exponential distribution
        self.spikesource.seed(1)
        # Define the double-exponential synaptic input
        self.synapse = neuron.h.Exp2Syn(neuron.h.cell.apic[secindex](x)) # pyramidal neuron for apical dendrites
        self.synapse.tau1 = 2
        self.synapse.tau2 = 10
        self.synapse.e    = 0
        # Connect the presynaptic discharge sequence with the synaptic connection.
        self.connection = neuron.h.NetCon(self.spikesource, self.synapse, thresh, delay, weight)
        self.connection.delay = delay
        self.connection.weight[0] = weight

    def add_synapse_L5PC_Clone3(self, thresh=10, delay=0., weight=0.004, secindex=3, x=0.5):
        # Define the presynaptic Poisson discharge sequence
        self.spikesource = neuron.h.NetStim()
        self.spikesource.interval = 1000/50  # ms (mean) time between spikes 
        self.spikesource.number   = int(1e9) # (average) number of spikes 
        self.spikesource.start    = 1000     # ms (mean) start time of first spike (The first 1000 milliseconds are used for the membrane potential to reach the resting value.)
        self.spikesource.noise    = 1        # range 0 to 1. Fractional randomness. 0 deterministic, 1 intervals have decaying exponential distribution
        self.spikesource.seed(1)
        # Define the double-exponential synaptic input
        self.synapse = neuron.h.Exp2Syn(neuron.h.cell.apic[secindex](x)) # pyramidal neuron for apical dendrites
        self.synapse.tau1 = 2
        self.synapse.tau2 = 10
        self.synapse.e    = 0
        # Connect the presynaptic discharge sequence with the synaptic connection.
        self.connection = neuron.h.NetCon(self.spikesource, self.synapse, thresh, delay, weight)
        self.connection.delay = delay
        self.connection.weight[0] = weight
        
    def add_synapse_L5PC_Clone4(self, thresh=10, delay=0., weight=0.004, secindex=3, x=0.5):
        # Define the presynaptic Poisson discharge sequence
        self.spikesource = neuron.h.NetStim()
        self.spikesource.interval = 1000/50  # ms (mean) time between spikes 
        self.spikesource.number   = int(1e9) # (average) number of spikes 
        self.spikesource.start    = 1000     # ms (mean) start time of first spike (The first 1000 milliseconds are used for the membrane potential to reach the resting value.)
        self.spikesource.noise    = 1        # range 0 to 1. Fractional randomness. 0 deterministic, 1 intervals have decaying exponential distribution
        self.spikesource.seed(1)
        # Define the double-exponential synaptic input
        self.synapse = neuron.h.Exp2Syn(neuron.h.cell.apic[secindex](x)) # pyramidal neuron for apical dendrites
        self.synapse.tau1 = 2
        self.synapse.tau2 = 10
        self.synapse.e    = 0
        # Connect the presynaptic discharge sequence with the synaptic connection.
        self.connection = neuron.h.NetCon(self.spikesource, self.synapse, thresh, delay, weight)
        self.connection.delay = delay
        self.connection.weight[0] = weight
        
    def add_synapse_L5PC_Clone5(self, thresh=10, delay=0., weight=0.004, secindex=3, x=0.5):
        # Define the presynaptic Poisson discharge sequence
        self.spikesource = neuron.h.NetStim()
        self.spikesource.interval = 1000/50  # ms (mean) time between spikes 
        self.spikesource.number   = int(1e9) # (average) number of spikes 
        self.spikesource.start    = 1000     # ms (mean) start time of first spike (The first 1000 milliseconds are used for the membrane potential to reach the resting value.)
        self.spikesource.noise    = 1        # range 0 to 1. Fractional randomness. 0 deterministic, 1 intervals have decaying exponential distribution
        self.spikesource.seed(1)
        # Define the double-exponential synaptic input
        self.synapse = neuron.h.Exp2Syn(neuron.h.cell.apic[secindex](x)) # pyramidal neuron for apical dendrites
        self.synapse.tau1 = 2
        self.synapse.tau2 = 10
        self.synapse.e    = 0
        # Connect the presynaptic discharge sequence with the synaptic connection.
        self.connection = neuron.h.NetCon(self.spikesource, self.synapse, thresh, delay, weight)
        self.connection.delay = delay
        self.connection.weight[0] = weight

    def run_simulation_withEF(self, dt=0.025, tstop=20.0, tvar=0., Evar=0.):
        h.dt = dt
        h.tstop = tstop
        h.setstim_snowp()
        h.stim_amp.from_python(Evar)
        h.stim_time.from_python(tvar)
        h.attach_stim()
        neuron.h.finitialize(-70)
        neuron.h.run()

cell = Cell(19)


def calc_entrainment_of_ROISurf_uniformEF(para_index, para_prefr, para_weight, para_seed, para_Evec):
    current_element_index = para_index  # not used
    if not hasattr(cell, 'synapse'):
        cell.add_synapse_L5PC_Clone4()
    cell.spikesource.interval = 1000 / para_prefr
    cell.spikesource.start = 1000
    cell.spikesource.noise = 1
    cell.spikesource.seed(para_seed)
    cell.connection.weight[0] = para_weight
    calc_uniformE_nonUnitEvec_direction(cell.allSections, para_Evec)
    # run simulation
    dt       = 0.05
    DEL, DUR = 1e3, 125e3
    tstop    = DEL + DUR + 0
    tvar     = np.arange(0, tstop, dt)
    Eamp, Efreq, Ephi = 2.0, 10.0, 0.0
    Evar_DEL    = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR    = Eamp * np.sin(2*np.pi*Efreq*np.arange(0, DUR, dt) / 1e3 + Ephi)
    Evar_tstop  = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar        = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))
    cell.run_simulation_withEF(dt=dt, tstop=tstop, tvar=tvar, Evar=Evar)
    t      = np.array(cell.recordings['t'].to_python()[50:])
    vsoma  = np.array(cell.recordings['soma(0.5)'].to_python()[50:])
    # calc PLV, firing rate, et al.
    Evar = np.sin(2*np.pi*Efreq*np.arange(0, DUR, dt) / 1e3 + Ephi)
    fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber = calc_field_spike_entrainment(vsoma=vsoma, Evar=Evar, tvar=t, Vth=0.0)

    return fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber

def calc_entrainment_sensitivity_to_field_direction(para_theta, para_phi, para_seed, para_weight, para_prefr):
    if not hasattr(cell, 'synapse'):
        cell.add_synapse_L5PC_Clone4()
    cell.spikesource.interval = 1000 / para_prefr
    cell.spikesource.start = 1000
    cell.spikesource.noise = 1
    cell.spikesource.seed(para_seed)
    cell.connection.weight[0] = para_weight
    calc_uniformE_theta_phi(cell.allSections, theta=para_theta, phi=para_phi)
    dt       = 0.05
    DEL, DUR = 1e3, 125e3
    tstop    = DEL + DUR + 0
    tvar     = np.arange(0, tstop, dt)
    Eamp, Efreq, Ephi = 1.0, 10.0, 0.0
    Evar_DEL    = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR    = Eamp * np.sin(2*np.pi*Efreq*np.arange(0, DUR, dt) / 1e3 + Ephi)
    Evar_tstop  = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar        = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))
    cell.run_simulation_withEF(dt=dt, tstop=tstop, tvar=tvar, Evar=Evar)
    t      = np.array(cell.recordings['t'].to_python()[50:])
    vsoma  = np.array(cell.recordings['soma(0.5)'].to_python()[50:])
    Evar = np.sin(2*np.pi*Efreq*np.arange(0, DUR, dt) / 1e3 + Ephi)
    fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber = calc_field_spike_entrainment(vsoma=vsoma, Evar=Evar, tvar=t, Vth=0.0)
    return fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber


def calc_polarization_sensitivity_to_field_direction(para_theta, para_phi, para_seed, para_weight, para_prefr):
    tempseed, tempweight, tempprefr = para_seed, para_weight, para_prefr
    if hasattr(cell, 'synapse'):
        cell.spikesource.start = 1e9 # close synaptic inputs
    # define recordings for all sections, x=0.5
    allSectionNames = ['sec_%d' % i for i in range(len(cell.allSections))]
    for i, item in enumerate(allSectionNames):
        if item not in cell.recordings.keys():
            cell.recordings[item] = neuron.h.Vector().record(cell.allSections[i](0.5)._ref_v)
    calc_uniformE_theta_phi(cell.allSections, theta=para_theta, phi=para_phi)
    dt       = 0.05
    DEL, DUR = 1e3, 1e3
    tstop    = DEL + DUR + 0
    tvar     = np.arange(0, tstop, dt)
    Eamp, Efreq, Ephi = 1.0, 10.0, 0.0
    Evar_DEL    = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR    = Eamp * np.sin(2*np.pi*Efreq*np.arange(0, DUR, dt) / 1e3 + Ephi)
    Evar_tstop  = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar        = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))
    cell.run_simulation_withEF(dt=dt, tstop=tstop, tvar=tvar, Evar=Evar)
    t = np.array(cell.recordings['t'].to_python()[50:])
    polar_length = np.zeros(len(allSectionNames))
    for i, item in enumerate(allSectionNames):
        Vmem = np.array(cell.recordings[item].to_python()[50:])
        idx_peaks = scipy.signal.find_peaks(Vmem)[0][-2:]
        idx_troughs = scipy.signal.find_peaks(-Vmem)[0][-2:]
        polar_length[i] = np.mean(Vmem[idx_peaks] - Vmem[idx_troughs]) / 2
    polar_soma = polar_length[0]
    polar_cell = np.max(polar_length)
    return polar_length, polar_soma, polar_cell


def calc_entrainment_sensitivity_to_field_intensity_at_a_specific_direction(para_amp, para_theta, para_phi, para_seed, para_weight, para_prefr):
    if not hasattr(cell, 'synapse'):
        cell.add_synapse_L5PC_Clone4()
    cell.spikesource.interval = 1000 / para_prefr
    cell.spikesource.start = 1000
    cell.spikesource.noise = 1
    cell.spikesource.seed(para_seed)
    cell.connection.weight[0] = para_weight
    calc_uniformE_theta_phi(cell.allSections, theta=para_theta, phi=para_phi)
    dt       = 0.05
    DEL, DUR = 1e3, 125e3
    tstop    = DEL + DUR + 0
    tvar     = np.arange(0, tstop, dt)
    Eamp, Efreq, Ephi = para_amp, 10.0, 0.0
    Evar_DEL    = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR    = Eamp * np.sin(2*np.pi*Efreq*np.arange(0, DUR, dt) / 1e3 + Ephi)
    Evar_tstop  = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar        = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))
    cell.run_simulation_withEF(dt=dt, tstop=tstop, tvar=tvar, Evar=Evar)
    t      = np.array(cell.recordings['t'].to_python()[50:])
    vsoma  = np.array(cell.recordings['soma(0.5)'].to_python()[50:])
    Evar = np.sin(2*np.pi*Efreq*np.arange(0, DUR, dt) / 1e3 + Ephi)
    fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber = calc_field_spike_entrainment(vsoma=vsoma, Evar=Evar, tvar=t, Vth=0.0)
    return fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber








if __name__ == '__main__':
    start_time = time.time()
    cell_name = 'L5PC_Clone4'
    casemark = 100

    '''calculating polarization-direction sensitivity'''
    seed, weight, prefr = 1328, 0.035, 50
    list_theta = np.arange(0, 181, 10)
    list_phi = np.arange(0, 360, 15)
    paras_all_410_directions = []  # 410 E-Field directions
    paras_all_410_directions.append([list_theta[0], 0, seed, weight, prefr])
    for i in range(1, list_theta.shape[0] - 1):
        for j in range(list_phi.shape[0]):
            paras_all_410_directions.append([list_theta[i], list_phi[j], seed, weight, prefr])
    paras_all_410_directions.append([list_theta[-1], 0, seed, weight, prefr])

    # casemark = 'polarization'
    if casemark == 'polarization':
        paras = paras_all_410_directions
        pool = multiprocessing.Pool(processes=20)
        res = pool.starmap(calc_polarization_sensitivity_to_field_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_polarization_sensitivity_to_Efield_410_directions.p' % cell_name, 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    start_time = time.time()
    '''calculating PLV-direction sensitivity'''
    # casemark = 1
    if casemark == 1:
        idx1, idx2 = 0, 100
        paras = paras_all_410_directions[idx1:idx2]
        pool = multiprocessing.Pool(processes=20)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Efield_410_directions_index%d-%d.p' % (cell_name, idx1, idx2), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    # casemark = 2
    if casemark == 2:
        idx1, idx2 = 100, 200
        paras = paras_all_410_directions[idx1:idx2]
        pool = multiprocessing.Pool(processes=20)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Efield_410_directions_index%d-%d.p' % (cell_name, idx1, idx2), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    # casemark = 3
    if casemark == 3:
        idx1, idx2 = 200, 300
        paras = paras_all_410_directions[idx1:idx2]
        pool = multiprocessing.Pool(processes=20)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Efield_410_directions_index%d-%d.p' % (cell_name, idx1, idx2), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    # casemark = 4
    if casemark == 4:
        idx1, idx2 = 300, 410
        paras = paras_all_410_directions[idx1:idx2]
        pool = multiprocessing.Pool(processes=22)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Efield_410_directions_index%d-%d.p' % (cell_name, idx1, idx2), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)


    ############################################
    '''calculating PLV-direction sensitivity'''
    ############################################
    seed, weight, prefr = 1328, 0.035, 50
    list_theta = [0, 60, 100, 150, 180]
    list_phi = [0, 75, 210, 300, 0]
    list_Eamp = np.arange(0.1, 2.01, 0.1)

    start_time = time.time()
    # casemark = 'intensity_point_0'
    if casemark == 'intensity_point_0':
        idx = 0
        paras = [[i, list_theta[idx], list_phi[idx], seed, weight, prefr] for i in list_Eamp]
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_intensity_at_a_specific_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Eamp0-2mV_theta%d_phi%d.p' % (cell_name, list_theta[idx], list_phi[idx]), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    # casemark = 'intensity_point_1'
    if casemark == 'intensity_point_1':
        idx = 1
        paras = [[i, list_theta[idx], list_phi[idx], seed, weight, prefr] for i in list_Eamp]
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_intensity_at_a_specific_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Eamp0-2mV_theta%d_phi%d.p' % (cell_name, list_theta[idx], list_phi[idx]), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    # casemark = 'intensity_point_2'
    if casemark == 'intensity_point_2':
        idx = 2
        paras = [[i, list_theta[idx], list_phi[idx], seed, weight, prefr] for i in list_Eamp]
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_intensity_at_a_specific_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Eamp0-2mV_theta%d_phi%d.p' % (cell_name, list_theta[idx], list_phi[idx]), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    # casemark = 'intensity_point_3'
    if casemark == 'intensity_point_3':
        idx = 3
        paras = [[i, list_theta[idx], list_phi[idx], seed, weight, prefr] for i in list_Eamp]
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_intensity_at_a_specific_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Eamp0-2mV_theta%d_phi%d.p' % (cell_name, list_theta[idx], list_phi[idx]), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

    # casemark = 'intensity_point_4'
    if casemark == 'intensity_point_4':
        idx = 4
        paras = [[i, list_theta[idx], list_phi[idx], seed, weight, prefr] for i in list_Eamp]
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_entrainment_sensitivity_to_field_intensity_at_a_specific_direction, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_sensitivity_to_Eamp0-2mV_theta%d_phi%d.p' % (cell_name, list_theta[idx], list_phi[idx]), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

