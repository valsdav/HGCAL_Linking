import uproot
import os
import numpy as np
import pandas as pd 
import awkward as ak 
import glob
from multiprocessing import Pool
import numba
from itertools import islice
    

def load_file(file):
    try:
        print('.', end="")
        f = uproot.open(file)
        t =  f["ticlNtuplizer/tracksters"]
        calo = f["ticlNtuplizer/simtrackstersCP"]
        ass = f["ticlNtuplizer/associations"]
        A = calo.arrays(["stsCP_trackster_barycenter_eta","stsCP_trackster_barycenter_phi",
                                  "stsCP_barycenter_x","stsCP_barycenter_y","stsCP_barycenter_z","stsCP_raw_energy"])
        B = t.arrays(["raw_energy","raw_em_energy", "trackster_barycenter_eta","trackster_barycenter_phi",
                                    "barycenter_x","barycenter_y","barycenter_z","id_probabilities",
                                    "EV1", "EV2", "EV3", "eVector0_x", "eVector0_y","eVector0_z", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3"])
        C = ass.arrays([ "tsCLUE3D_recoToSim_CP", "tsCLUE3D_recoToSim_CP_score"])
        return A, B, C
    except:
        print("error ", file)



@numba.njit
def in_window(calo_eta, calo_phi, calo_z, track_eta, track_phi, track_z, builder):
    deta = 0.10
    dphi = 0.25
    for c_eta, c_phi, c_z, t_eta, t_phi, t_z in zip(calo_eta, calo_phi, calo_z, track_eta, track_phi, track_z):
        builder.begin_list()
        for (ceta, cphi, cz, teta,tphi, tz) in zip(c_eta, c_phi, c_z, t_eta, t_phi, t_z):
            #print(cz, tz, ceta, teta, cphi,tphi)
            same_z = np.sign(cz) == np.sign(tz)
            in_eta = abs(ceta- teta) < deta
            in_phi = abs(((cphi - tphi + np.pi) % (2 * np.pi) - np.pi)) < dphi
            builder.append(same_z & in_eta & in_phi)
        builder.end_list()
    return builder


import numba
@numba.njit
def calo_match_mask(tcks_indices, indices_to_match, builder):
    for tr_ind, index_to_match in zip(tcks_indices, indices_to_match):
        builder.begin_list()
        #Looping other all the calolist for each trackers
        for trInd in range(len(tr_ind)):    
            calo_indices_for_this_track = tr_ind[trInd]
            #print(calo_indices_for_this_track)
            found = False
            for i in range(len(calo_indices_for_this_track)):
                # checking if the index is the one to keep
                #print(index_to_match[trInd])
                if i == index_to_match[trInd]:
                    builder.append(calo_indices_for_this_track[i])
                    found =True
            if not found:
                builder.append(None)
            
        builder.end_list()
    return builder



def save_dataset(files, i, output):

    p = Pool()
    results = p.map(load_file, files)
    calos = [ ]
    tracksters = [ ]
    associations = [ ]
    for c,t, a in [r for r in results if r ]:
        calos.append(c)
        tracksters.append(t)
        associations.append(a)
        
        
    df_calo = ak.concatenate(calos)
    df_track = ak.concatenate(tracksters)
    df_ass = ak.concatenate(associations)

    EM_mask =  (df_track.raw_em_energy / df_track.raw_energy ) > 0.9
    df_track_EM = df_track[EM_mask]

    # Pairs are already only with EM masked trackers
    pairs = ak.argcartesian([df_calo.stsCP_trackster_barycenter_eta, df_track_EM.trackster_barycenter_eta], axis=1)
    calo_idx, track_idx = ak.unzip(pairs)

    df_calo.eta = df_calo.stsCP_trackster_barycenter_eta
    df_calo.phi = df_calo.stsCP_trackster_barycenter_phi
    df_calo.z = df_calo.stsCP_barycenter_z
    df_track.eta = df_track_EM.trackster_barycenter_eta
    df_track.phi = df_track_EM.trackster_barycenter_phi
    df_track.z = df_track_EM.barycenter_z

    all_calo_eta = df_calo.stsCP_trackster_barycenter_eta[calo_idx]
    all_calo_phi = df_calo.stsCP_trackster_barycenter_phi[calo_idx]
    all_calo_z = df_calo.stsCP_barycenter_z[calo_idx]
    
    all_track_eta = df_track_EM.trackster_barycenter_eta[track_idx]
    all_track_phi = df_track_EM.trackster_barycenter_phi[track_idx]
    all_track_z = df_track_EM.barycenter_z[track_idx]

    out = in_window(all_calo_eta, all_calo_phi, all_calo_z, 
                all_track_eta, all_track_phi,  all_track_z, ak.ArrayBuilder())

    goodpairs = pairs[out]

    calo_idx_inwindow, track_idx_inwindow = ak.unzip(goodpairs)

    min_score = 0.9
    masked_score  = ak.mask(df_ass.tsCLUE3D_recoToSim_CP_score, df_ass.tsCLUE3D_recoToSim_CP_score<min_score)
    armin = ak.fill_none(ak.argmin(masked_score, axis=2), -1)
    trackers_to_calo_ = calo_match_mask(df_ass.tsCLUE3D_recoToSim_CP, armin, ak.ArrayBuilder()).snapshot()
    # remember to apply the EM_Mask
    trackers_to_calo = trackers_to_calo_[EM_mask]

    # mis-alignment calculation
    norm_PCAVect = (df_track_EM.eVector0_x **2 + df_track_EM.eVector0_y**2  + df_track_EM.eVector0_z**2)**0.5
    norm_baryVect = (df_track_EM.barycenter_x **2 + df_track_EM.barycenter_y**2  + df_track_EM.barycenter_z**2)**0.5
    prod_scalar = (df_track_EM.barycenter_x * df_track_EM.eVector0_x + df_track_EM.barycenter_y * df_track_EM.eVector0_y + \
                        df_track_EM.barycenter_z * df_track_EM.eVector0_z) /(norm_PCAVect*norm_baryVect)

    X = [ ]
    Y = [ ] 
    Y_meta = [ ]

    mask_events = ak.num(df_calo.stsCP_raw_energy, axis=1) == 2
    padding = 30

    for calo_idx in [0, 1]:
        #mask everything
        calo_idx_inwindow_M = calo_idx_inwindow[mask_events]
        track_idx_inwindow_M = track_idx_inwindow[mask_events]
        df_track_EM_M = df_track_EM[mask_events]
        trackers_to_calo_M = trackers_to_calo[mask_events]
        # Get the tracksters in the window of each calo particles
        tracks_in_window = track_idx_inwindow_M[calo_idx_inwindow_M == calo_idx]
        # Get only those trackers by ID
        trk_data = df_track_EM_M[tracks_in_window]
        # Save the input variables
        x = ak.zip({"raw_en": trk_data.raw_energy, 
                         "barycenter_x": trk_data.barycenter_x,
                         "barycenter_y": trk_data.barycenter_y,
                         "barycenter_z": trk_data.barycenter_z,
                         "EV1": trk_data.EV1,
                         "EV2": trk_data.EV2,
                         "EV3": trk_data.EV3,
                         "sigmaPCA1": trk_data.sigmaPCA1,
                         "sigmaPCA2": trk_data.sigmaPCA2,
                         "sigmaPCA3": trk_data.sigmaPCA3,
                         "misalign":  prod_scalar[mask_events][tracks_in_window],
                        }  )

        trackers_in_window_truth = ak.fill_none(trackers_to_calo_M[ tracks_in_window ] == calo_idx, False)

        # Saving also the caloparticle metadata
        m = df_calo[mask_events,calo_idx]
        X.append(x)
        Y.append(trackers_in_window_truth)
        Y_meta.append(m)


    print(X)
    dataset = ak.zip({"X": ak.concatenate(X, axis=0), 
                      "Y": ak.concatenate(Y, axis=0), 
                      "meta": ak.concatenate(Y_meta, axis=0)})
    ak.to_parquet(dataset, f"{output}/dataset_{i}.parquet")
        



if __name__ == "__main__":
    import sys
    input_folder = sys.argv[1]
    all_files = glob.glob(f"{input_folder}/ntuples_13845741_*.root")

    N = len(all_files) 
    print(N)
    for i, k in enumerate(range(0, N, 200)):
        print(i)
        save_dataset(all_files[k:k+200], i, "dataset")
	

