import numpy as np
import neuron
from neuron import h
import time, os
import subprocess
import pickle, glob
import scipy.signal
from scipy.signal import find_peaks
import copy

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

'''
Return the neuronal section and segment coordinates: start, mid, and end positions
'''
def calc_cell_section_and_segment_coordinates(cellAllSections):
    num_section = len(cellAllSections)
    num_segment = sum([sec.nseg for sec in cellAllSections])
    cell_section_names = []
    cell_section_coordinates_start = np.zeros((num_section, 3))
    cell_section_coordinates_end   = np.zeros((num_section, 3))
    cell_section_coordinates_mid   = np.zeros((num_section, 3))
    cell_section_diameter = np.zeros(num_section)
    cell_segment_names = []
    cell_segment_coordinates_start = np.zeros((num_segment, 3))
    cell_segment_coordinates_end   = np.zeros((num_segment, 3))
    cell_segment_coordinates_mid   = np.zeros((num_segment, 3))
    cell_segment_diameter = np.zeros(num_segment)
    
    num_sec_seg = 0
    for i, sec in enumerate(cellAllSections):
        if 'soma' in sec.name():
            cell_section_names.append('soma')
        elif 'dend' in sec.name():
            cell_section_names.append('basal')
        elif 'apic' in sec.name():
            cell_section_names.append('apic')
        else:
            cell_section_names.append('axon') 

        cell_section_diameter[i] = sec.diam
        n3d = int(sec.n3d())
        xx, yy, zz = np.zeros(n3d), np.zeros(n3d), np.zeros(n3d) 
        secLength = np.zeros(n3d)
        for j in range(n3d):
            secLength[j] = sec.arc3d(j)
            xx[j], yy[j], zz[j] = sec.x3d(j), sec.y3d(j), sec.z3d(j)
        secLength_norm = secLength / sec.arc3d(n3d-1) 
        nseg = int(sec.nseg)
        for m, segment in enumerate(sec):
            cell_segment_diameter[num_sec_seg+m] = segment.diam
            cell_segment_names.append(cell_section_names[-1])
        secRange = np.arange(0., (nseg+1)/nseg + 1/(2*nseg), 1/nseg) 
        segmentOfSection_xinterp_start = np.interp(secRange[:-2], secLength_norm, xx)
        segmentOfSection_yinterp_start = np.interp(secRange[:-2], secLength_norm, yy)
        segmentOfSection_zinterp_start = np.interp(secRange[:-2], secLength_norm, zz)
        segmentOfSection_xinterp_end = np.interp(secRange[1:-1], secLength_norm, xx)
        segmentOfSection_yinterp_end = np.interp(secRange[1:-1], secLength_norm, yy)
        segmentOfSection_zinterp_end = np.interp(secRange[1:-1], secLength_norm, zz)
        secRange -= 1 / (2 * nseg)
        secRange[0] = 0.0
        secRange[-1] = 1.0
        segmentOfSection_xinterp_mid = np.interp(secRange, secLength_norm, xx)
        segmentOfSection_yinterp_mid = np.interp(secRange, secLength_norm, yy)
        segmentOfSection_zinterp_mid = np.interp(secRange, secLength_norm, zz)

        sec_start = np.array([segmentOfSection_xinterp_mid[0], segmentOfSection_yinterp_mid[0], segmentOfSection_zinterp_mid[0]])
        sec_end = np.array([segmentOfSection_xinterp_mid[-1], segmentOfSection_yinterp_mid[-1], segmentOfSection_zinterp_mid[-1]])
        sec_mid = (sec_start + sec_end) / 2
        cell_section_coordinates_start[i, :] = sec_start
        cell_section_coordinates_end[i, :]   = sec_end
        cell_section_coordinates_mid[i, :]   = sec_mid

        seg_start = np.vstack((segmentOfSection_xinterp_start, segmentOfSection_yinterp_start, segmentOfSection_zinterp_start)).transpose()
        seg_end = np.vstack((segmentOfSection_xinterp_end, segmentOfSection_yinterp_end, segmentOfSection_zinterp_end)).transpose()
        seg_mid = np.vstack((segmentOfSection_xinterp_mid[1:-1], segmentOfSection_yinterp_mid[1:-1], segmentOfSection_zinterp_mid[1:-1])).transpose()
        cell_segment_coordinates_start[num_sec_seg:num_sec_seg + nseg, :] = seg_start
        cell_segment_coordinates_end[num_sec_seg:num_sec_seg + nseg, :] = seg_end
        cell_segment_coordinates_mid[num_sec_seg:num_sec_seg + nseg, :] = seg_mid
        num_sec_seg += nseg
    

    index_soma  = np.array([i for i, item in enumerate(cell_section_names) if item=='soma'])
    index_basal = np.array([i for i, item in enumerate(cell_section_names) if item=='basal'])
    index_apic  = np.array([i for i, item in enumerate(cell_section_names) if item=='apic'])
    index_axon  = np.array([i for i, item in enumerate(cell_section_names) if item=='axon'])
    section_coords = {
        'cell_section_names': cell_section_names,
        'cell_section_coordinates_start': cell_section_coordinates_start,
        'cell_section_coordinates_end': cell_section_coordinates_end,
        'cell_section_coordinates_mid': cell_section_coordinates_mid,
        'index_soma_section': index_soma,
        'index_basal_section': index_basal,
        'index_apic_section': index_apic,
        'index_axon_section': index_axon,
        'cell_section_diameter': cell_section_diameter}
  
    index_soma  = np.array([i for i, item in enumerate(cell_segment_names) if item=='soma'])
    index_basal = np.array([i for i, item in enumerate(cell_segment_names) if item=='basal'])
    index_apic  = np.array([i for i, item in enumerate(cell_segment_names) if item=='apic'])
    index_axon  = np.array([i for i, item in enumerate(cell_segment_names) if item=='axon'])
    segment_coords = {
        'cell_segment_names': cell_segment_names,
        'cell_segment_coordinates_start': cell_segment_coordinates_start,
        'cell_segment_coordinates_end': cell_segment_coordinates_end,
        'cell_segment_coordinates_mid': cell_segment_coordinates_mid,
        'index_soma_segment': index_soma,
        'index_basal_segment': index_basal,
        'index_apic_segment': index_apic,
        'index_axon_segment': index_axon,
        'cell_segment_diameter': cell_segment_diameter }
    return section_coords, segment_coords



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


def CellPt3dchange_ByPhiNormalOrigin_NEURON(cellSectionLists, cell_phi, cell_normal, cell_origin):
    rotMatrix_phi = getRotationMatrix(vecAx=[0, 0, 1], angle=cell_phi)
    rotMatrix_normal = getRotationMatrix(vec0=[0, 0, 1], vec1=cell_normal)
    for sec_hObj in cellSectionLists:
        for j in range(sec_hObj.n3d()):
            x0, y0, z0 = sec_hObj.x3d(j), sec_hObj.y3d(j), sec_hObj.z3d(j)
            vec_xyz = np.dot(rotMatrix_phi, np.array([x0, y0, z0]))
            vec_xyz = np.dot(rotMatrix_normal, vec_xyz)
            vec_xyz = vec_xyz + cell_origin
            sec_hObj.pt3dchange(j, vec_xyz[0], vec_xyz[1], vec_xyz[2], sec_hObj.diam3d(j))   

def placeCellSection_ByPhiNormalOrigin(secCoordStart, secCoordEnd, secCoordMid, cell_phi=None, cell_normal=None, cell_origin=None):
    # secCoordStart: shape is (numSection, 3=x,y,z)
    secCoordStart_new = np.zeros_like(secCoordStart)
    secCoordEnd_new   = np.zeros_like(secCoordEnd)
    secCoordMid_new   = np.zeros_like(secCoordMid)
    if cell_phi is None and cell_normal is None and cell_origin is None:
        print('Don\'t change cell coordinates.')
        return secCoordStart, secCoordEnd, secCoordMid
        
    if cell_phi is not None:
        rotMatrix_phi = getRotationMatrix(vecAx=[0, 0, 1], angle=cell_phi)
    if cell_normal is not None:
        rotMatrix_normal = getRotationMatrix(vec0=[0, 0, 1], vec1=cell_normal)
    for i in range(secCoordStart.shape[0]):
        xyz_start = secCoordStart[i, :]
        xyz_end   = secCoordEnd[i, :]
        xyz_mid   = secCoordMid[i, :]
        if cell_phi is not None:
            xyz_start = np.dot(rotMatrix_phi, xyz_start)
            xyz_end   = np.dot(rotMatrix_phi, xyz_end)
            xyz_mid   = np.dot(rotMatrix_phi, xyz_mid)
        if cell_normal is not None:
            xyz_start = np.dot(rotMatrix_normal, xyz_start)
            xyz_end   = np.dot(rotMatrix_normal, xyz_end)
            xyz_mid   = np.dot(rotMatrix_normal, xyz_mid)
        if cell_origin is not None:
            xyz_start += cell_origin
            xyz_end   += cell_origin
            xyz_mid   += cell_origin
        secCoordStart_new[i, :] = xyz_start
        secCoordEnd_new[i, :]   = xyz_end
        secCoordMid_new[i, :]   = xyz_mid

    return secCoordStart_new, secCoordEnd_new, secCoordMid_new

def interpolate_field_ofCellSecAndSeg(sectionCoords, segmentCoords, E_position, E_vector, method='section', numClosedPoints=10, outlier_threshold=2.0):
    # inputs - coords: (num_sec/num_seg, 3) unit is mm
    # inputs - E_position: (num_nodes, 3) unit is mm
    # inputs - E_vector: (num_nodes, 3) unit is V/m or mV/mm
    '''
    # (1) Find the 10 closest nodes around the cell
    num_section = sectionCoords.shape[0]
    num_segment = segmentCoords.shape[0]
    if method == 'segment':
        indices_10closed_points_around_cell = np.zeros((num_segment, numClosedPoints))
        for i in range(num_segment):
            distance = E_position - segmentCoords[i]
            distance = np.linalg.norm(distance, axis=1)
            indices_10closed_points_around_cell[i, :] = np.argsort(distance)[:numClosedPoints]
    elif method == 'section':
        # as the same with matlab function knnsearch(E_position, sectionCoords[i],'k',10),
        # but the matlab code run very fast than python code
        indices_10closed_points_around_cell = np.zeros((num_section, numClosedPoints))
        for i in range(num_section):
            distance = E_position - sectionCoords[i]
            distance = np.linalg.norm(distance, axis=1)
            indices_10closed_points_around_cell[i, :] = np.argsort(distance)[:numClosedPoints]
    unique_indices = np.unique(indices_10closed_points_around_cell).astype(int)
    nodes_near_cell = E_position[unique_indices, :]
    Evector_near_cell = E_vector[unique_indices, :]
    logical_indices_normal_Evector = exclude_outlier_Evector(Evector_near_cell, threshold=outlier_threshold)
    unique_indices = unique_indices[logical_indices_normal_Evector]
    nodes_near_cell = nodes_near_cell[logical_indices_normal_Evector, :]
    Evector_near_cell = Evector_near_cell[logical_indices_normal_Evector, :]
    '''
    numClosedPoints = 10
    while_flag = True
    while while_flag:
        unique_indices, nodes_near_cell, Evector_near_cell = find_closest_points_around_cell(sectionCoords, segmentCoords, \
                                                E_position, E_vector, method=method, numClosedPoints=numClosedPoints, outlier_threshold=outlier_threshold)
        Ex_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 0], segmentCoords, method='linear')
        Ey_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 1], segmentCoords, method='linear')
        Ez_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 2], segmentCoords, method='linear')
        Einterp_segment = np.column_stack((Ex_interp, Ey_interp, Ez_interp))
        if np.any(np.isnan(Einterp_segment)):
            numClosedPoints += 5
        else:
            while_flag = False
            
    # (2) Interpolate the extracellular electric field vector for each section and segment of the neuron
    #     based on the closest nodes information around the cell
    # The results are the same with the matlab function: scatteredInterpolant
    Ex_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 0], segmentCoords, method='linear')
    Ey_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 1], segmentCoords, method='linear')
    Ez_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 2], segmentCoords, method='linear')
    Einterp_segment = np.column_stack((Ex_interp, Ey_interp, Ez_interp))
    Ex_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 0], sectionCoords, method='linear')
    Ey_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 1], sectionCoords, method='linear')
    Ez_interp = scipy.interpolate.griddata(nodes_near_cell, Evector_near_cell[:, 2], sectionCoords, method='linear')
    Einterp_section = np.column_stack((Ex_interp, Ey_interp, Ez_interp))
    return Einterp_section, Einterp_segment, Evector_near_cell, nodes_near_cell, unique_indices


def find_closest_points_around_cell(sectionCoords, segmentCoords, E_position, E_vector, method='section', numClosedPoints=10, outlier_threshold=2.0):
    # (1) Find the 10 closest nodes around the cell
    num_section = sectionCoords.shape[0]
    num_segment = segmentCoords.shape[0]
    if method == 'segment':
        indices_10closed_points_around_cell = np.zeros((num_segment, numClosedPoints))
        for i in range(num_segment):
            distance = E_position - segmentCoords[i]
            distance = np.linalg.norm(distance, axis=1)
            indices_10closed_points_around_cell[i, :] = np.argsort(distance)[:numClosedPoints]
    elif method == 'section':
        # as the same with matlab function knnsearch(E_position, sectionCoords[i],'k',10),
        # but the matlab code run very fast than python code
        indices_10closed_points_around_cell = np.zeros((num_section, numClosedPoints))
        for i in range(num_section):
            distance = E_position - sectionCoords[i]
            distance = np.linalg.norm(distance, axis=1)
            indices_10closed_points_around_cell[i, :] = np.argsort(distance)[:numClosedPoints]
    unique_indices = np.unique(indices_10closed_points_around_cell).astype(int)
    nodes_near_cell = E_position[unique_indices, :]
    Evector_near_cell = E_vector[unique_indices, :]
    logical_indices_normal_Evector = exclude_outlier_Evector(Evector_near_cell, threshold=outlier_threshold)
    unique_indices = unique_indices[logical_indices_normal_Evector]
    nodes_near_cell = nodes_near_cell[logical_indices_normal_Evector, :]
    Evector_near_cell = Evector_near_cell[logical_indices_normal_Evector, :]
    return unique_indices, nodes_near_cell, Evector_near_cell


def exclude_outlier_Evector(Evector_near_cell, threshold=2.0):
    Emagn = np.linalg.norm(Evector_near_cell, axis=1)
    mean_Emag, std_Emag = np.mean(Emagn), np.std(Emagn)
    # calculate the z_score of each point
    z_scores = (Emagn - mean_Emag) / std_Emag
    logical_indices_exclude_outlier = np.abs(z_scores) < threshold
    return logical_indices_exclude_outlier
