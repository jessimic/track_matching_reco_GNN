import ROOT as rt
import uproot
from larcv import larcv
import math
import numpy as np
import glob
import os
from shutil import move
import random
import matplotlib.pyplot as plt

#This is code that takes all the MINERvA root files in a directory and creates a LARCV files that can be input into GrapPA

#Class for the data of the current file being input
class Mx2Data:
    def __init__(self, filename):

        # setup input
        
        self.file = uproot.open(filename)
        self.num_entries = self.file["minerva"].num_entries
        
        self.offsetX = self.file["minerva"]["offsetX"].array(library="np")
        self.offsetY = self.file["minerva"]["offsetY"].array(library="np")
        self.offsetZ = self.file["minerva"]["offsetZ"].array(library="np")
 
        self.n_tracks = self.file["minerva"]["n_tracks"].array(library="np") 
        self.n_blobs_id = self.file["minerva"]["n_blobs_id"].array(library="np")
        self.trk_vis_energy = self.file["minerva"]["trk_vis_energy"].array(library="np")
        self.trk_type = self.file["minerva"]["trk_type"].array(library="np")
        self.trk_patrec = self.file["minerva"]["trk_patrec"].array(library="np")
        self.trk_node_X = self.file["minerva"]["trk_node_X"].array(library="np")
        self.trk_node_Y = self.file["minerva"]["trk_node_Y"].array(library="np")
        self.trk_node_Z = self.file["minerva"]["trk_node_Z"].array(library="np")
        self.trk_index = self.file["minerva"]["trk_index"].array(library="np")
        self.trk_nodes = self.file["minerva"]["trk_nodes"].array(library="np")
        self.trk_time_slice = self.file["minerva"]["trk_time_slice"].array(library="np")
        self.trk_node_qOverP = self.file["minerva"]["trk_node_qOverP"].array(library="np")
        self.trk_node_cluster_idx = self.file["minerva"]["trk_node_cluster_idx"].array(library="np")

        self.clus_id_coord = self.file["minerva"]["clus_id_coord"].array(library="np")
        self.clus_id_z = self.file["minerva"]["clus_id_z"].array(library="np")
        self.clus_id_module = self.file["minerva"]["clus_id_module"].array(library="np")
        self.clus_id_strip = self.file["minerva"]["clus_id_strip"].array(library="np")
        self.clus_id_view = self.file["minerva"]["clus_id_view"].array(library="np")
        self.clus_id_pe = self.file["minerva"]["clus_id_pe"].array(library="np")
        self.clus_id_energy = self.file["minerva"]["clus_id_energy"].array(library="np")
        self.clus_id_time_slice = self.file["minerva"]["clus_id_time_slice"].array(library="np")
        self.clus_id_time = self.file["minerva"]["clus_id_time"].array(library="np")
        self.clus_id_type = self.file["minerva"]["clus_id_type"].array(library="np")
        self.clus_id_hits_idx = self.file["minerva"]["clus_id_hits_idx"].array(library="np")
        self.clus_id_size = self.file["minerva"]["clus_id_size"].array(library="np")

        self.mc_id_nmchit = self.file["minerva"]["mc_id_nmchit"].array(library="np")
        self.mc_id_mchit_x = self.file["minerva"]["mc_id_mchit_x"].array(library="np")
        self.mc_id_mchit_y = self.file["minerva"]["mc_id_mchit_y"].array(library="np")
        self.mc_id_mchit_z = self.file["minerva"]["mc_id_mchit_z"].array(library="np")
        self.mc_id_mchit_trkid = self.file["minerva"]["mc_id_mchit_trkid"].array(library="np")
        self.mc_id_mchit_dE = self.file["minerva"]["mc_id_mchit_dE"].array(library="np")
        self.mc_id_mchit_dL = self.file["minerva"]["mc_id_mchit_dL"].array(library="np")

        self.vtx_x = self.file["minerva"]["vtx_x"].array(library="np")
        self.vtx_y = self.file["minerva"]["vtx_y"].array(library="np")
        self.vtx_z = self.file["minerva"]["vtx_z"].array(library="np")
        self.vtx_tracks_idx = self.file["minerva"]["vtx_tracks_idx"].array(library="np")

        self.mc_id_module = self.file["minerva"]["mc_id_module"].array(library="np")        
        self.mc_id_strip = self.file["minerva"]["mc_id_strip"].array(library="np")        
        self.mc_id_view = self.file["minerva"]["mc_id_view"].array(library="np")
        self.mc_id_dE = self.file["minerva"]["mc_id_dE"].array(library="np")
        self.mc_id_pe = self.file["minerva"]["mc_id_pe"].array(library="np")

        self.mc_traj_edepsim_trkid = self.file["minerva"]["mc_traj_edepsim_trkid"].array(library="np")
        self.mc_traj_trkid = self.file["minerva"]["mc_traj_trkid"].array(library="np")
        self.mc_traj_edepsim_eventid = self.file["minerva"]["mc_traj_edepsim_eventid"].array(library="np")
        self.mc_traj_pdg = self.file["minerva"]["mc_traj_pdg"].array(library="np")
        self.mc_traj_point_x = self.file["minerva"]["mc_traj_point_x"].array(library="np")
        self.mc_traj_point_y = self.file["minerva"]["mc_traj_point_y"].array(library="np")
        self.mc_traj_point_z = self.file["minerva"]["mc_traj_point_z"].array(library="np")
        self.mc_traj_overflow = self.file["minerva"]["mc_traj_overflow"].array(library="np")
        self.mc_traj_point_t = self.file["minerva"]["mc_traj_point_z"].array(library="np")
        

if __name__ == "__main__":
    #larcv_directory = "/n/holystore01/LABS/iaifi_lab/Users/jmicallef/data_2x2/minerva/larcv/validation_set/"
    #check_directory = "/n/holystore01/LABS/iaifi_lab/Users/jmicallef/data_2x2/minerva/validation_set/"
    larcv_directory = "/n/holystore01/LABS/iaifi_lab/Users/jmicallef/data_2x2/minerva/larcv/"
    check_directory = "/n/holystore01/LABS/iaifi_lab/Users/jmicallef/data_2x2/minerva/"

    # List all .root files in the specified directory
    input_files = glob.glob(os.path.join(larcv_directory, "out*.root"))
    
    save_true_num_tracks = []
    save_split_num_tracks = []
    save_connected_num_tracks = []
    maxe = 0
    filecount=0
    for i in input_files:
        base_name = os.path.basename(i)
        input_file = check_directory+base_name[4:]
        print(base_name)

        try:
            Mx2Hits = Mx2Data(input_file)
            filecount+=1
        except:
            print("skipping base_name")
            continue
        maxe+=Mx2Hits.num_entries
        for entry in range(0,Mx2Hits.num_entries):
            offsetZ = Mx2Hits.offsetZ[entry]
            orig_num_tracks = Mx2Hits.n_tracks[entry]
            for idx in Mx2Hits.trk_index[entry]:

                n_nodes = Mx2Hits.trk_nodes[entry][idx]
                # print(n_nodes)
                if ((n_nodes >0)):
                    # print( "t", Mx2Hits.trk_node_Z[entry][idx][0])
                    z_nodes = Mx2Hits.trk_node_Z[entry][idx][:n_nodes] - offsetZ
                    has_pos = np.any(z_nodes > 0)
                    has_neg = np.any(z_nodes < 0)
                    if has_pos and has_neg:
                        orig_num_tracks +=1
            save_true_num_tracks.append(Mx2Hits.n_tracks[entry])
            save_split_num_tracks.append(orig_num_tracks)
            save_connected_num_tracks.append(orig_num_tracks-Mx2Hits.n_tracks[entry])
    
print(filecount)    
print(sum(save_true_num_tracks),sum(save_split_num_tracks),sum(save_connected_num_tracks))
print(sum(save_true_num_tracks)/maxe,sum(save_split_num_tracks)/maxe,sum(save_connected_num_tracks)/maxe)
print(save_connected_num_tracks.count(0),maxe,max(save_connected_num_tracks),max(save_true_num_tracks),max(save_split_num_tracks))

sample = "train_"
max_truth = max(save_true_num_tracks)
plt.figure(figsize=(10,8))
plt.hist(save_true_num_tracks,bins=max_truth)
plt.xlabel("True Number of Tracks")
plt.title("True Number of Tracks")
plt.savefig(sample+"true_num_tracks.png")

plt.figure(figsize=(10,8))
max_split = max(save_split_num_tracks)
plt.hist(save_split_num_tracks,bins=max_split)
plt.xlabel("Number of Track Segments")
plt.title("True Split Number of Tracks")
plt.savefig(sample+"split_num_tracks.png")

plt.figure(figsize=(10,8))
max_connected = max(save_connected_num_tracks)
plt.hist(save_connected_num_tracks,bins=max_connected)
plt.xlabel("Number of Connected Track Segments Tracks")
plt.title("True Connected Number of Tracks")
plt.savefig(sample+"connected_num_tracks.png")


plt.figure(figsize=(10,8))
counts, xedges, yedges, im = plt.hist2d(save_true_num_tracks,save_connected_num_tracks, 
                                        bins=[max_truth, max_connected], 
                                        cmap='viridis')
plt.colorbar(im, label='Counts')
plt.xlabel("Number of Full Tracks")
plt.ylabel("Number of Connected Track Segments Tracks")
plt.title("Total Tracks vs. Number of Connected Segements")
plt.savefig(sample+"true_vs_connected_num_tracks.png")

plt.figure(figsize=(10,8))
counts, xedges, yedges, im = plt.hist2d(save_true_num_tracks,save_split_num_tracks, 
                                        bins=[max_truth, max_split], 
                                        cmap='viridis')
plt.colorbar(im, label='Counts')
plt.xlabel("Number of Full Tracks")
plt.ylabel("Number of Track Segments")
plt.title("Total Tracks vs. Number of Track Segements")
plt.savefig(sample+"true_vs_split_num_tracks.png")
