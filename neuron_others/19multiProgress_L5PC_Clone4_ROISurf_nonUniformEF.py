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

def trans_surfCell_nonUniformEvectorOfSegments_to_origin(nonUniformEvecs, elm_normal, elm_phi=None):
    rotMatrix_normal = getRotationMatrix(vec0=elm_normal, vec1=[0, 0, 1])
    origin_segmentEvectors = np.zeros_like(nonUniformEvecs)
    for i in range(nonUniformEvecs.shape[0]):
        origin_segmentEvectors[i, :] = np.dot(rotMatrix_normal, nonUniformEvecs[i, :])
    return origin_segmentEvectors

def calc_nonUniformEvector_Of_Segment(allSectionLists, segmentEvectors, setVsoma0=False):
    iseg = 0
    soma_potential = 0
    for sec in allSectionLists:
        if h.ismembrane('xtra', sec=sec) and h.ismembrane('extracellular', sec=sec):
            for seg in sec:
                Evec = segmentEvectors[iseg, :]
                Ex, Ey, Ez = Evec[0], Evec[1], Evec[2]
                seg.es_xtra = -(Ex*seg.x_xtra + Ey*seg.y_xtra + Ez*seg.z_xtra) * 1e-3 
                if iseg == 0:
                    soma_potential = seg.es_xtra
                iseg += 1
    if setVsoma0 == True:
        # set soma extracellular potential to zero
        if np.abs(soma_potential) > 1e-5:
            for sec in allSectionLists:
                if h.ismembrane('xtra', sec=sec) and h.ismembrane('extracellular', sec=sec):
                    for seg in sec:
                        seg.es_xtra = seg.es_xtra - soma_potential


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


def calc_entrainment_of_ROISurf_nonUniformEF(para_index, para_prefr, para_weight, para_seed, para_nonUniformEvecs, para_normal):
    current_element_index = para_index  # not used
    if not hasattr(cell, 'synapse'):
        cell.add_synapse_L5PC_Clone4()
    cell.spikesource.interval = 1000 / para_prefr
    cell.spikesource.start = 1000
    cell.spikesource.noise = 1
    cell.spikesource.seed(para_seed)
    cell.connection.weight[0] = para_weight
    nonUniformEvector_ofSegment_atOrigin = trans_surfCell_nonUniformEvectorOfSegments_to_origin(para_nonUniformEvecs, para_normal)
    calc_nonUniformEvector_Of_Segment(cell.allSections, nonUniformEvector_ofSegment_atOrigin)
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

def calc_polarization_of_ROISurf_nonUniformEF(para_index, para_nonUniformEvecs, para_normal):
    current_element_index = para_index
    if hasattr(cell, 'synapse'):
        cell.spikesource.start = 1e9
    # define recordings for all sections, x=0.5
    allSectionNames = ['sec_%d' % i for i in range(len(cell.allSections))]
    for i, item in enumerate(allSectionNames):
        if item not in cell.recordings.keys():
            cell.recordings[item] = neuron.h.Vector().record(cell.allSections[i](0.5)._ref_v)
    nonUniformEvector_ofSegment_atOrigin = trans_surfCell_nonUniformEvectorOfSegments_to_origin(para_nonUniformEvecs, para_normal)
    calc_nonUniformEvector_Of_Segment(cell.allSections, nonUniformEvector_ofSegment_atOrigin)
    dt       = 0.05
    DEL, DUR = 1e3, 1e3
    tstop    = DEL + DUR + 0
    tvar     = np.arange(0, tstop, dt)
    Eamp, Efreq, Ephi = 2.0, 10.0, 0.0 # Eamp = 2mV/mm, Efreq = 10 Hz, Ephi = 0°
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



if __name__ == '__main__':
    # non-uniform Efield
    num_element = 1000
    dataPath = 'data_ROISurf/Layer5_popCell_interpolate_Evector_forSectionAndSegment.npy'
    data = np.load(dataPath, allow_pickle=True).item()
    popCell_Einterp_section = data['popCell_Einterp_section']
    popCell_Einterp_segment = data['popCell_Einterp_segment']  # shape=(1000, 1808, 3)

    # load population information
    dataPath = 'data_ROISurf/Layer5_popCell_sectionAndSegmentCoordinates.npy'
    popCellInfo = np.load(dataPath, allow_pickle=True).item()
    cell_normals = popCellInfo['cell_normals']
    cell_origins = popCellInfo['cell_origins'] * 1e3  # mm to μm
    list_segment_Evectors = [popCell_Einterp_segment[i] for i in range(num_element)]
    list_cell_normals = [cell_normals[j] for j in range(num_element)]

    # simulation parameter
    prefr, weight, seed = 50, 0.035, 1328

    #######################################
    start_time = time.time()
    cell_name = 'L5PC_Clone4'
    casemark = 100
    #######################################
    num_cpu = 20
    # casemark = 1
    if casemark == 1:
        idx1, idx2 = 0, 1000
        paras = [[i, prefr, weight, seed, list_segment_Evectors[i], list_cell_normals[i]] for i in range(idx1, idx2)]
        pool = multiprocessing.Pool(processes=num_cpu)
        res = pool.starmap(calc_entrainment_of_ROISurf_nonUniformEF, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_PLV_ROISurf_nonUniform_tACS2mA10Hz_scalingFactor2_index%d-%d.p' % (cell_name, idx1, idx2), 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)


    #######################################################
    '''polarization of ROI surface with 1000 elements'''
    #######################################################
    casemark = 'polarization'
    if casemark == 'polarization':
        paras = [[i, list_segment_Evectors[i], list_cell_normals[i]] for i in range(num_element)]
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_polarization_of_ROISurf_nonUniformEF, paras)
        pool.close()
        pool.join()
        pickle.dump(res, open('data_ROISurf/%s_polarization_ROISurf_nonUniform_tACS2mA10Hz_scalingFactor2.p' % cell_name, 'wb'), protocol=5)
        print('Running time:', time.time() - start_time)

