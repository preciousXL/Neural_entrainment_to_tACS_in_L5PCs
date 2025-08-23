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
from neuron_help_functions import Cell, calc_cell_section_and_segment_coordinates
from neuron_help_functions import placeCellSection_ByPhiNormalOrigin
from neuron_help_functions import CellPt3dchange_ByPhiNormalOrigin_NEURON
from neuron_help_functions import getRotationMatrix, getRotation_RotVecDegree

# change the cell id
cell_id = 17
cell = Cell(cell_id)


def calc_ROISurfEvec_to_OriginEvec(elmEvecs, elmNormals):
    # rotate Efield vector from layer surface to the origin (0, 0, 0),
    # rather than place cell in the layer surface
    num_elem    = elmEvecs.shape[0]
    originEvecs = np.zeros((num_elem, 3))
    for i in range(num_elem):
        Evec, normal = elmEvecs[i,:], elmNormals[i,:]
        rotMatrix_normal = getRotationMatrix(vec0=normal, vec1=[0, 0, 1])
        originEvecs[i, :] = np.dot(rotMatrix_normal, elmEvecs[i, :])
    return originEvecs

def calc_uniformE_nonUnitEvec_direction(allSectionLists, Evec):
    Ex, Ey, Ez = Evec[0], Evec[1], Evec[2]
    for sec in allSectionLists:
        if h.ismembrane('xtra', sec=sec) and h.ismembrane('extracellular', sec=sec):
            for seg in sec:
                # potential units in NEURON is mV, and thus multiply a factor of 1e-3 to transfer
                seg.es_xtra = -(Ex*seg.x_xtra + Ey*seg.y_xtra + Ez*seg.z_xtra) * 1e-3

def calcSpikeTime_spikeIndex_FiringRate_spikeNumber_fromVth(vsoma, tvar, Vth=0.0):
    ''' calculate firing rate, spike timing and its index, spike number.'''
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
    # calculate Efield instantanous phase using hilbert transfer
    analyticSignal = scipy.signal.hilbert(Evar)
    tacsInstantaneousPhase = np.angle(analyticSignal)  # -π~π
    spikeTime, spikeIndex, firingRate, spikeNumber = calcSpikeTime_spikeIndex_FiringRate_spikeNumber_fromVth(vsoma, tvar, Vth=Vth)
    spikePhaseRadian = tacsInstantaneousPhase[spikeIndex]
    spikePhaseRadian[spikePhaseRadian < 0] += 2*np.pi
    pluralPLV = np.mean(np.exp(1j*spikePhaseRadian))
    spikePhaseDegree = np.rad2deg(spikePhaseRadian)
    fieldSpikePLV = np.abs(pluralPLV)

    return fieldSpikePLV, spikePhaseDegree, pluralPLV, spikeTime, spikeIndex, firingRate, spikeNumber


def calc_entrainment_of_ROISurf_uniformEF(cell_name, para_index, para_prefr, para_weight, para_seed, para_Evec):
    current_element_index = para_index  # not used
    if not hasattr(cell, 'synapse'):
        if cell_name == 'L5PC_Clone1':
            cell.add_synapse_L5PC_Clone1()
        elif cell_name == 'L5PC_Clone2':
            cell.add_synapse_L5PC_Clone2()
        elif cell_name == 'L5PC_Clone3':
            cell.add_synapse_L5PC_Clone3()
        elif cell_name == 'L5PC_Clone4':
            cell.add_synapse_L5PC_Clone4()
        elif cell_name == 'L5PC_Clone5':
            cell.add_synapse_L5PC_Clone5()
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


if __name__ == '__main__':
    cell_name = cell.cell_clone_name
    print('Cell clone name: ', cell_name)
    data_dir = '../data'
    num_element = 1000

    # Load cell Efield information in the layer population
    dataPath = os.path.join(data_dir, '%s_popCell_interpolate_Evector_forSectionAndSegment.npy' % cell_name)
    data = np.load(dataPath, allow_pickle=True).item()
    popCell_Einterp_section = data['popCell_Einterp_section']
    uniformEvector_by_soma = popCell_Einterp_section[:, 0, :]

    # Load normals and coordinates of each element in the layer SOI mesh
    dataPath = os.path.join(data_dir, '%s_popCell_sectionAndSegmentCoordinates.npy' % cell_name)
    popCellInfo = np.load(dataPath, allow_pickle=True).item()
    cell_normals = popCellInfo['cell_normals']
    cell_origins = popCellInfo['cell_origins'] * 1e3  # mm to μm

    # Rotate interpolated Efield from layer surface to the origin
    originEvectors = calc_ROISurfEvec_to_OriginEvec(uniformEvector_by_soma, cell_normals)
    list_Evec = [originEvectors[i, :] for i in range(num_element)]

    # initialize the synaptic parameter for corresponding cell type
    prefr, weight, seed = None, None, None
    if cell_name == 'L5PC_Clone1':
        seed, weight, prefr = 3722, 0.031, 50
    elif cell_name == 'L5PC_Clone2':
        seed, weight, prefr = 4784, 0.035, 50
    elif cell_name == 'L5PC_Clone3':
        seed, weight, prefr = 4784, 0.035, 50
    elif cell_name == 'L5PC_Clone4':
        seed, weight, prefr = 1328, 0.035, 50
    elif cell_name == 'L5PC_Clone5':
        seed, weight, prefr = 3166, 0.035, 50

    # Run simulations
    print('Synaptic parameters: ', seed, weight, prefr)
    start_time = time.time()
    num_cpu = 20
    # casemark = None

    casemark = 1
    if casemark == 1:
        idx1, idx2 = 0, 1000
        save_path = os.path.join(data_dir, '%s_PLV_ROISurf_uniform_tACS2mA10Hz_index%d-%d.p' % (cell_name, idx1, idx2))
        if not os.path.exists(save_path):
            print('save path: ', save_path)
            paras = [[cell_name, i, prefr, weight, seed, list_Evec[i]] for i in range(idx1, idx2)]
            pool = multiprocessing.Pool(processes=num_cpu)
            res = pool.starmap(calc_entrainment_of_ROISurf_uniformEF, paras)
            pool.close()
            pool.join()
            pickle.dump(res, open(save_path, 'wb'), protocol=5)
            print('Running time:', time.time() - start_time)
