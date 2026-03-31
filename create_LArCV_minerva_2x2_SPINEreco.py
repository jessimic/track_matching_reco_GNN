import ROOT as rt
import uproot
from larcv import larcv
import math
import numpy as np
import glob
import os
from shutil import move
import random
import re
import os,sys
import yaml
from spine.io.core.read.hdf5 import HDF5Reader
from spine.post.truth.match import MatchProcessor
from spine.utils.match import overlap_iou
#from spine.driver import Driver
#This is code that takes all the MINERvA root files in a directory and creates a LARCV files that can be input into GrapPA


#Class for the data of the current file being input
class Mx2Data:
    def __init__(self, filename): #, output_filename):

        # setup input
        
        self.file = uproot.open(filename)
        
        self.mc_run = self.file["minerva"]["mc_run"].array(library="np")
        self.mc_subrun = self.file["minerva"]["mc_subrun"].array(library="np")
        
        self.offsetX = self.file["minerva"]["offsetX"].array(library="np")
        self.offsetY = self.file["minerva"]["offsetY"].array(library="np")
        self.offsetZ = self.file["minerva"]["offsetZ"].array(library="np")
 
        self.trk_node_X = self.file["minerva"]["trk_node_X"].array(library="np")
        self.trk_node_Y = self.file["minerva"]["trk_node_Y"].array(library="np")
        self.trk_node_Z = self.file["minerva"]["trk_node_Z"].array(library="np")
        self.trk_index = self.file["minerva"]["trk_index"].array(library="np")
        self.trk_nodes = self.file["minerva"]["trk_nodes"].array(library="np")
        self.n_tracks = self.file["minerva"]["n_tracks"].array(library="np") 

        # Get most energetic particle
        self.trk_node_cluster_idx = self.file["minerva"]["trk_node_cluster_idx"].array(library="np")
        self.clus_id_z = self.file["minerva"]["clus_id_z"].array(library="np")
        self.clus_id_hits_idx = self.file["minerva"]["clus_id_hits_idx"].array(library="np")
        self.clus_id_size = self.file["minerva"]["clus_id_size"].array(library="np")
        self.mc_id_nmchit = self.file["minerva"]["mc_id_nmchit"].array(library="np")
        self.mc_id_mchit_trkid = self.file["minerva"]["mc_id_mchit_trkid"].array(library="np")
        self.mc_id_mchit_dE = self.file["minerva"]["mc_id_mchit_dE"].array(library="np")
        self.mc_traj_edepsim_trkid = self.file["minerva"]["mc_traj_edepsim_trkid"].array(library="np")
        self.mc_traj_trkid = self.file["minerva"]["mc_traj_trkid"].array(library="np")
        self.mc_traj_edepsim_eventid = self.file["minerva"]["mc_traj_edepsim_eventid"].array(library="np")
        self.mc_traj_pdg = self.file["minerva"]["mc_traj_pdg"].array(library="np")
        

        # setup iomanager for output mode
        # IOManager class declaration in larcv/core/DataFormat/IOManager.h        
        #self.out_larcv = larcv.IOManager( larcv.IOManager.kWRITE )
        # use the following for more verbose description of iomanager's actions
        ##self.out_larcv.set_verbosity( larcv.msg.kINFO )
        ##self.out_larcv.set_verbosity( larcv.msg.kDEBUG )
        #self.out_larcv.set_out_file( output_filename )
        #self.out_larcv.initialize()

    def get_most_energetic_particle(self, entry,track_index):
        clus_id_z = (self.clus_id_z[entry])

        cl_list = self.trk_node_cluster_idx[entry][track_index] # select the clusters associated with the nodes of the track
        cl_list = cl_list[cl_list>=0]
        hit_list = self.clus_id_hits_idx[entry][cl_list] # Lists of digits that were clustered for each of those clusters
        hit_list = hit_list[hit_list>=0]

        hit_energy_list = self.mc_id_mchit_dE[entry][hit_list] # Energy deposited
        nhits = self.mc_id_nmchit[entry][hit_list]
        nhits = np.where(nhits>2,2,nhits) # Each digits is connsidered to be at most 2 true MC hits

        traj_list = np.concatenate([self.mc_id_mchit_trkid[entry][hit_list][i][:nhits[i]] for i in range(len(nhits))]) # Get the list of all trajectories that contributed to the track.
        # traj_list = Mx2Hits.mc_id_mchit_trkid[entry][hit_list][:,0]

        # Getting the trajectories that contributed
        traj_list = traj_list[traj_list>0]
        hit_energy_list = hit_energy_list[hit_energy_list>0]
        particle_energy = {}
        for i in range(len(traj_list)):
            particle_id = traj_list[i]
            energy = hit_energy_list[i]
            if particle_id in particle_energy:
                particle_energy[particle_id] += energy
            else:
                particle_energy[particle_id] = energy
        #if len(particle_energy) == 0:
            #print("Is this the problem?")
        if len(particle_energy) > 0:
            max_energy_particle_id = max(particle_energy, key=particle_energy.get)
            edep_traj_name = self.mc_traj_edepsim_trkid[entry][max_energy_particle_id]
            edep_traj_evtid = self.mc_traj_edepsim_eventid[entry][max_energy_particle_id]
            mc_traj_trkid = self.mc_traj_trkid[entry][max_energy_particle_id]
            mc_traj_pdg = Mx2Hits.mc_traj_pdg[entry][max_energy_particle_id]
        else:
            mc_traj_pdg = 0
            edep_traj_name = 0 
            edep_traj_evtid = 0
        #print(edep_traj_evtid, mc_traj_trkid, mc_traj_pdg)

        return mc_traj_pdg, edep_traj_name, edep_traj_evtid

    def find_Mx2_uniqueID(self, entry):

        unique_id_store = []
        vertexid_store = []
        unique_id_dict = {}
        for idx in self.trk_index[entry]:        
            n_nodes = self.trk_nodes[entry][idx]
            if ((n_nodes >0)):
                edep_pdg, edep_traj_name, edep_traj_evtid = self.get_most_energetic_particle(entry, idx)
                fullid = str(edep_pdg)+str(edep_traj_evtid)+str(edep_traj_name)
                vertexid_store.append(edep_traj_evtid)
                unique_id_store.append(fullid)
         
        minerva_vertices = [str(val) for val in set(vertexid_store)]
        unique_id_dict = {str(val): i for i, val in enumerate(set(unique_id_store))}

        return minerva_vertices, unique_id_dict

    #def process_entries(self):
    #    """
    #    function responsible for managing loop over entries
    #    """
    #    nentries = len(self.trk_node_X) # made up
    #    run = 0 #self.mc_run
    #    subrun = 0 #self.mc_subrun

     #   for entry in range(nentries):
            # convert input data into larcv data for one entry
     #       fail = self.process_one_minvera_entry(entry)
     #       if not fail:
                # set the entry id
     #           self.out_larcv.set_id( run, subrun, entry )
                # save the entry data
     #           self.out_larcv.save_entry()
            #else:
            #    print("Not writing", entry)

    def process_one_minerva_entry(self,entry,out_larcv, all_uniqueID_dict, vsa_in, vox3dmeta, fragment_counter):
        """
        extract information from minerva trees and store in lartpc_mlreco classes for one entry

        from (old) grappa parser config
        schema:
          clust_label:
            - parse_cluster3d_full
            - cluster3d_pcluster_highE
            - particle_corrected
          coords:
            - parse_particle_coords
            - particle_corrected
            - cluster3d_pcluster_highE
        """

        # how larcv represents clusters and match to objects
        # individual voxel of energy deposition <---> larcv.Voxel3D
        # tracks <---> larcv.VoxelSet <---> a list of indices within a 3D array of voxels
        # container of tracks <---> larcv.VoxelSetArray
        # ClusterVoxel3D  = larcv.VoxelSetArray + larcv.Voxel3DMeta (the latter providing map from voxel indices to 3D position

        # class declaration of Voxel3D and VoxelSet is in larcv/core/DataFormat/Voxel.h
        # class declaration of Voxel3DMeta is in larcv/core/DataFormat/Voxel3DMeta.h        
        # class declaration of ClusterVoxel3D is in larcv/core/DataFormat/ClusterVoxel3D.h

        # one wonky thing is that each voxel is imagined to have an ID represented by a single integer
        # i.e. for a 3d array of voxels, you have to imagine assigning a sequatial ID to each voxel after "unrolling" it.
        # the way to convert from (a more natural) triplet-index (e.g. (i,j,k) is to employ the Voxel3DMeta functions:
        # VoxelID_t index(const size_t i_x, const size_t i_y, const size_t i_z) const;
        # inline VoxelID_t id(const Point3D& pt) const
        # there are also functions to go from sequential index (VoxelID_t) to triplet-index: 
        #  // Find x index that corresponds to a specified index
        #  size_t id_to_x_index(VoxelID_t id) const;
        #  // Find y index that corresponds to a specified index
        #  size_t id_to_y_index(VoxelID_t id) const;
        #  // Find z index that corresponds to a specified index
        #  size_t id_to_z_index(VoxelID_t id) const;
        #  // Find xyz index that corresponds to a specified index
        #  void id_to_xyz_index(VoxelID_t id, size_t& x, size_t& y, size_t& z) const;

        #Saved offset of the MINERvA origin definition, compared to the 2x2
        offsetX = self.offsetX[entry]
        offsetY = self.offsetY[entry]
        offsetZ = self.offsetZ[entry]
        

        #print("Entry: ", entry)
        for idx in self.trk_index[entry]:
            n_nodes = self.trk_nodes[entry][idx]
            if ((n_nodes >0)):
               #This array keeps track of up vs downstream nodes, helps to distinguish between clusters
                Us_Ds=[]
        
                x_nodes = self.trk_node_X[entry][idx][:n_nodes] - offsetX
                y_nodes = self.trk_node_Y[entry][idx][:n_nodes] - offsetY 
                z_nodes = self.trk_node_Z[entry][idx][:n_nodes] - offsetZ
                any_us = sum(z_nodes < 0) > 0 #Any nodes upstream?
                any_ds = sum(z_nodes > 0) > 0 #Any nodes downstream?
                assert sum(z_nodes< 0)+sum(z_nodes > 0) == len(z_nodes), "Any upstream vs downstream should be mutually exclusive"

                #Finding the trajectory that contributed the most
                edep_pdg, edep_traj_name, edep_traj_evtid = self.get_most_energetic_particle(entry, idx)
                fullid = str(edep_pdg)+str(edep_traj_evtid)+str(edep_traj_name)
                if fullid == "000":
                    print("Skipping empty Mx2 track in minerva spill ID", entry)
                    continue
                
                fragment_id=fragment_counter
                group_id= all_uniqueID_dict[str(fullid)] 
                fragment_counter += 1
                print("MINERVA Fragment: ", fragment_id, "Group: ", group_id, "Unique ID", str(fullid))
                
                #Create particle larcv object. Each cluster will need its own particle object, even if the clusters belong to one particle.
                #Since this code only consider MINERvA tracks, every particle is assigned a track shape
                particle = larcv.Particle(larcv.ShapeType_t.kShapeTrack)
                particle.id(int(fragment_id))
                particle.group_id(int(group_id))

                #A voxel set corresponds to all the voxels for one cluster 
                # fill voxelset 
                track_as_voxelset = larcv.VoxelSet()
                track_as_voxelset.id(int(fragment_id))

                    
                for edepvoxels in range(n_nodes):
                    voxelid = vox3dmeta.id( x_nodes[edepvoxels], y_nodes[edepvoxels], z_nodes[edepvoxels] )
                    voxel = larcv.Voxel( voxelid )
                    if voxelid!=larcv.kINVALID_VOXELID: #Skip if invalid
                        track_as_voxelset.add( voxel )

                # add voxelset to container
                vsa_in.insert( track_as_voxelset )

                # get the cluster3d entry container, by contributing VoxelSetArray and the Voxel3Dmeta
                entry_clust3d = out_larcv.get_data( "cluster3d", "pcluster" )
                entry_clust3d.set( vsa, vox3dmeta )
                
                #get particle container and fill
                entry_particles = out_larcv.get_data( "particle", "corrected")
                entry_particles.append(particle)
 
        return fragment_counter, vsa_in, vox3dmeta, out_larcv, all_uniqueID_dict

    #def write_and_close(self):
    #    self.out_larcv.finalize()

def compute_iou_matrix(reco_clusters, truth_particles, get_index_fn):
    """
    Compute IoU overlap matrix between reco index clusters and truth particles. From Claude after reading spine matching.

    Parameters
    ----------
    reco_clusters : List[np.ndarray]
        List of index arrays for reco objects (your raw clusters)
    truth_particles : List[object]
        List of full truth particle objects
    get_index_fn : callable
        The processor's self.get_index method to extract indexes from truth objects
    """
    # Build truth index list using the existing get_index method
    reco_input = [] #typed.List.empty_list(nb.int64[:])
    for cluster in reco_clusters:
        reco_input.append(cluster.astype(np.int64))

    truth_input = [] #typed.List.empty_list(nb.int64[:])
    for p in truth_particles:
        truth_input.append(get_index_fn(p))

    # Compute IoU matrix (N_reco x N_truth)
    if len(reco_input) and len(truth_input):
        ovl_matrix = overlap_iou(reco_input, truth_input)
    else:
        ovl_matrix = np.empty((len(reco_input), len(truth_input)))

    return ovl_matrix

def match_reco_clusters_to_truth(reco_clusters, particle_shapes, truth_particles, get_index_fn,
                                  min_overlap=0.0):
    """
    Match reco index clusters to truth particles using IoU. From Claude after reading spine matching.

    Parameters
    ----------
    reco_clusters : List[np.ndarray]
        List of raw index arrays for reco objects
    particle_shapes : List or np.ndarray
        Shape label for each reco cluster, same length as reco_clusters
    truth_particles : List[object]
        List of full truth particle objects
    get_index_fn : callable
        The processor's self.get_index method
    min_overlap : float
        Minimum IoU threshold to consider a valid match

    Returns
    -------
    pairs : List[tuple]
        (reco_cluster, truth_particle or None) for each reco cluster
    overlaps : List[float]
        Best overlap score per reco cluster (-1.0 if unmatched)
    """
    assert len(reco_clusters) == len(particle_shapes), "reco_clusters (len" + len(reco_clusters) + " ) and particle_shapes len(" + len(particle_shapes) + ") must have the same length."
    
    ovl_matrix = compute_iou_matrix(reco_clusters, truth_particles, get_index_fn)
    ovl_valid = ovl_matrix > min_overlap

    pairs, pair_overlaps = [], []
    for i, (cluster, shape) in enumerate(zip(reco_clusters, particle_shapes)):
        match_idxs = np.where(ovl_valid[i])[0]

        if not len(match_idxs):
            pairs.append((cluster, None))
            pair_overlaps.append(-1.0)
        else:
            overlaps = ovl_matrix[i, match_idxs]
            perm = np.argsort(overlaps)[::-1]
            sorted_idxs = match_idxs[perm]

            best_truth = truth_particles[sorted_idxs[0]]
            pairs.append((cluster, shape, best_truth))
            pair_overlaps.append(overlaps[perm[0]])

    return pairs, pair_overlaps
    
def get_pair_data(i, pairs, spine_data):
    
    

    pair_here = pairs[i]

    reco_cluster = pair_here[0]
    reco_shape = pair_here[1]
    mask = reco_cluster
    points_particle = spine_data['points'][mask]
    shape = int(reco_shape) if reco_shape is not None else -1

    tp = None
    if len(pair_here) > 2:
        tp = pair_here[2]
        
    return shape, mask, points_particle, tp

def process_one_2x2_entry(spine_data, entry,out_larcv, all_uniqueID_dict, vsa, vox3dmeta, fragment_counter, size_threshold=5):
    
    #print("Processing ND entry", entry)
    
    no_truth_match = 1
    
    #print("")
    #print("Particles:")
    dontuse, dontuse2, int_dict = find_2x2_uniqueIDs_remaining(spine_data) #get the interaction ID to vertex ID conversion
    #print("2x2 interaction ID dict", int_dict)
    for cluster in range(len(pairs)):
        shape, mask, points_particle, tp = get_pair_data(cluster, pairs, spine_data)
        if shape == 1: #SAVE TRACK ONLY!

            fragment_id=fragment_counter #Start from MINERvA fragments
            fragment_counter += 1

            if tp is not None: #Check if it was matched with a truth particle
                fullid = str(tp.pdg_code)+str(int_dict[tp.interaction_id])+str(tp.track_id) 
            else:
                #No uniqueID since that is calculated base on truth
                #Create unique ID
                fullid = "0" + str(no_truth_match) + "0"
            group_id= all_uniqueID_dict[str(fullid)] 
        

            #Create threshold
            #if len(p.points[:,0]) < size_threshold:
                #print("2x2 Particle under size threshold %i in event %i, skipping"%(size_threshold,entry))
            #    continue

        
            print("SPINE Fragment: ", fragment_id, "Group: ", group_id, "Unique ID", str(fullid))
            
            #Create particle larcv object
            particle = larcv.Particle(larcv.ShapeType_t.kShapeTrack)
            particle.id(int(fragment_id))
            particle.group_id(int(group_id))

            #A voxel set corresponds to all the voxels for one cluster 
            # fill voxelset 
            track_as_voxelset = larcv.VoxelSet()
            track_as_voxelset.id(int(fragment_id))

            
            for vox in range(len(points_particle[:, 0])):
                voxelid = vox3dmeta.id( points_particle[vox, 0]*10, points_particle[vox, 1]*10, points_particle[vox, 2]*10 )
                voxel = larcv.Voxel( voxelid )
                if voxelid!=larcv.kINVALID_VOXELID: #Skip if invalid
                    track_as_voxelset.add( voxel )

            # add voxelset to container
            vsa.insert( track_as_voxelset )

            # get the cluster3d entry container, by contributing VoxelSetArray and the Voxel3Dmeta
            entry_clust3d = out_larcv.get_data( "cluster3d", "pcluster" )
            entry_clust3d.set( vsa, vox3dmeta )
            
            #get particle container and fill
            entry_particles = out_larcv.get_data( "particle", "corrected")
            entry_particles.append(particle)

    return out_larcv

def setup_larcv(output_filename):
    out_larcv = larcv.IOManager( larcv.IOManager.kWRITE )
    # use the following for more verbose description of iomanager's actions
    #out_larcv.set_verbosity( larcv.msg.kINFO )
    #out_larcv.set_verbosity( larcv.msg.kDEBUG )
    out_larcv.set_out_file( output_filename )
    out_larcv.initialize()

    return out_larcv
    
def process_entry(out_larcv, run, subrun, entry):
    out_larcv.set_id( run, subrun, entry )
    # save the entry data
    out_larcv.save_entry()

def write_and_close(out_larcv):
    out_larcv.finalize()
    
def find_2x2_uniqueIDs_remaining(spine_data, uniqueID_dict = None,size_threshold=0):
    ## Takes in a dictionary that has other "int ID" to cross reference with 
    ## pdg+interaction+trackid "unique ID" per track. Use to combine Mx2 and
    ## 2x2 tracks. Or keep key_dict empty to make a new dict, the keys are
    ## meant to take unique interaction IDs to a simple int list starting
    ## from 0 for each spill, numbering by interaction+track ID
    
    vertex_ids = []
    pdgs =[]
    trackids = []
    allids = []
    reco_clusters = spine_data["particle_clusts"]
    truth_particles = spine_data['truth_particles']
    truth_interactions = spine_data['truth_interactions']
    fake_interaction = 1
    
    interaction_dict = {}
    for ixn in truth_interactions:
        idkey = ixn.id
        vertex_id = ixn.interaction_id
        interaction_dict[idkey] = vertex_id
    
    for cluster in range(len(reco_clusters)):    
        shape, mask, points_particle, tp = get_pair_data(cluster, pairs, spine_data)
        #Create threshold
        if tp is not None:
            if len(tp.points[:,0]) < size_threshold:
                continue
            #particle.interaction_id tells you int from 0 that corresponds to the vertex ID list in interaction.interaction_id
            vertex_ids.append(interaction_dict[tp.interaction_id])
            pdgs.append(tp.pdg_code)
            trackids.append(tp.track_id)
            allids.append(str(tp.pdg_code)+str(interaction_dict[tp.interaction_id])+str(tp.track_id))
        else:
            vertex_ids.append(fake_interaction) 
            pdgs.append(0)
            trackids.append(0)
            allids.append("0"+str(fake_interaction)+"0")
            fake_interaction+=1
    
    all_2x2_vertexIDs = set(vertex_ids) #just vertex ID
    unique_ids = set(allids) #includes pdg and track ID
    #print("Mx2 IDs", uniqueID_dict)
    #print("2x2 IDs", unique_ids)
    needed_new_ints = len(set(allids))
    
    if uniqueID_dict:
        new_ids = [str(val) for val in unique_ids if val not in uniqueID_dict] #key_dict
        starting_index = len(uniqueID_dict)
        #print("New 2x2 IDs", new_ids, "starting at", starting_index, "adding until", needed_ints)
    else:
        uniqueID_dict = {}
        new_ids = [str(val) for val in unique_ids]
        #print("No given Mx2 uniqueIDs", new_ids)
        starting_index = 0

    needed_ints = starting_index + len(new_ids)
    for i in range(starting_index, needed_ints):
        key = new_ids[i - starting_index]
        #if key not in unique_ids:
        uniqueID_dict[key] = i
    #print("Unique IDs saved", uniqueID_dict)
    #uniqueID_array = [uniqueID_dict[str(uid)] for uid in unique_ids]

    return uniqueID_dict, all_2x2_vertexIDs, interaction_dict
    
def find_2x2_vertices(spine_data):
    
    truth_interactions = spine_data['truth_interactions']
    vertex_ids = []
    for ixn in truth_interactions:
        vertex_ids.append(ixn.interaction_id)

    unique_ids = set(vertex_ids)
    
    new_ids = [str(val) for val in unique_ids]
   
    return new_ids

def fill_in_nonshared_events(n_array, m_array):
    
    #NOT CHECKED BUT ASSUMES THAT ALL SPINE EVENTS ARE VALID, MATCHES Mx2 to that
    
    #Works on assumption that these are sorted already in accending order
    n_is_sorted = all(n_array[i] <= n_array[i+1] for i in range(len(n_array)-1))
    m_is_sorted = all(m_array[i] <= m_array[i+1] for i in range(len(m_array)-1))
    assert n_is_sorted
    assert m_is_sorted
    
    full_range = list(range(n_array[0], n_array[-1] + 1))
    missing = sorted(set(full_range) - set(n_array))

    filled_n_indices = n_array
    filled_m_indices = m_array
    for num in missing:
        # Find where to insert the missing number
        insert_pos = next(i for i, val in enumerate(n_array) if val > num)
        filled_n_indices.insert(insert_pos, num)
        filled_m_indices.insert(insert_pos, m_array[insert_pos-1]+1)  # or -1, or something meaningful

    return filled_n_indices, filled_m_indices

def find_vertex_ids_all_events(spine_reader,Mx2Hits):
    minerva_indices_save = []
    nd_indices_save = []
    total_events_spine = len(spine_reader)
    total_events_mx2 = len(Mx2Hits.n_tracks)
    assert total_events_spine < total_events_mx2 #spine should be more "pruned" than minerva file

    #Get all vertices per event
    for i in range(0,total_events_spine):
        #print("ENTRY ", i)
        min_indices, mx2_unique_IDs = Mx2Hits.find_Mx2_uniqueID(i)
        minerva_indices_save.append(min_indices)

        spine_data = spine_reader[i]
        nd_indices = find_2x2_vertices(spine_data)
        nd_indices_save.append(nd_indices)
    for i in range(total_events_spine,total_events_mx2):
        
        min_indices, mx2_unique_IDs = Mx2Hits.find_Mx2_uniqueID(i)
        minerva_indices_save.append(min_indices)
    
    #Match vertices per event numbers
    m_indices = []
    n_indices = []
    for i, m_sub in enumerate(minerva_indices_save):
        for j, n_sub in enumerate(nd_indices_save):
            shared = set(m_sub) & set(n_sub)
            if shared:
                #print(f"m[{i}] and n[{j}] share {list(shared)}")
                m_indices.append(i)
                n_indices.append(j)

    full_n, full_m = fill_in_nonshared_events(n_indices, m_indices)

    return full_n, full_m


def make_larcv_box(xmin = (-1080.0),ymin = (-1450.0), zmin = (-2400.0),xmax = (1080.0),ymax = (1000.0),zmax = (3100.0)):
    #Manually making a bounding box around both upstream and downstream MINERvA planes, in the 2x2 coordinate frame
                
    #voxel size is 3mm (matching 2x2, I think)
    xnum = int(math.ceil(abs((xmin - xmax)/3)))
    ynum = int(math.ceil(abs((ymin - ymax)/3)))
    znum = int(math.ceil(abs((zmin - zmax)/3)))

    # Define the meta for LArCV
    # the "meta" is used to map individual 3D voxels within a 3D array to the physical positions in the detector
    vox3dmeta = larcv.Voxel3DMeta()
    # define the meta with the set(...) function
    """
    inline void set(double xmin, double ymin, double zmin,
        double xmax, double ymax, double zmax,
        size_t xnum,size_t ynum,size_t znum,
        DistanceUnit_t unit=kUnitCM)
    """
    vox3dmeta.set(xmin,ymin,zmin,xmax,ymax,zmax,xnum,ynum,znum) 
    #The voxelsetarray contains all the voxels for one event
    vsa = larcv.VoxelSetArray()

    return vsa, vox3dmeta

if __name__ == "__main__":
    directory = "/global/homes/j/jessiem/track_matching_reco_GNN/mcfiles/"

    training_output = directory+"larcv/"
    validation_output = directory+"larcv/larcv_validation_set/"

    # List all .root files in the specified directory
    training_input_files = glob.glob(os.path.join(directory+"minerva/", "*.dst.root"))
    all_files = training_input_files
    #Shuffle to make random order
    shuffled_file_order = np.arange(0, len(all_files) )
    np.random.shuffle(shuffled_file_order)

    #Set up validation saving throughout processing
    #num_training = int(0.8 * len(all_files))
    train_ratio = 1.0 #0.8
    if train_ratio < 1:
        val_every = round(1 / (1 - train_ratio))  # every Nth file for validation
    else:
        val_every = 1000000
    
    #all_files = all_files[:1] 
    print(training_input_files, len(all_files))

    #Want to overwrite? Change to true, otherwise, will check and only write new files
    rewrite_output = True
    
    n_entries =[]
    n_matches = []
    n_clusters = []
    for f_id in range(0,len(all_files)):
        base_name = os.path.basename(all_files[f_id])
        # extract the 7-digit number (or any sequence of digits)
        match = re.search(r'\.(\d{7})\.', base_name)
        if match:
            filenum = match.group(1)
            #print(filenum)  # 0000101
        spine_file = directory + "spine_reco/MiniRun6.5_1E19_RHC.flow2supera."+filenum+".LARCV_MiniRun6.5_1E19_RHC.flow2supera.0-199.LARCV_partialSPINE_withtruth.h5"
        if not os.path.isfile(spine_file) or not os.path.isfile(all_files[f_id]):
            print("Missing matching spine or minerva file, check!!!!! skipping", spine_file, all_files[f_id])
            continue
        
        training_exists = False
        validation_exists = False
        if f_id % val_every == 0:
            output_file = os.path.join(validation_output, f"out_{base_name}")
            validation_exists = os.path.isfile(output_file)
            training_bool = False
        #if f_id < num_training:
        else:
            output_file = os.path.join(training_output, f"out_{base_name}")
            training_exists = os.path.isfile(output_file)
            training_bool = True
        
        if training_exists and validation_exists:
            assert True, "SAME FILE IN TRAINING AND VALIDATION, BREAKING"
        if not rewrite_output:
            if os.path.isfile(output_file):
                print(output_file, "already exists, NOT re-writing (can change bool in code to overwrite)")
                continue

        print(base_name, os.path.basename(spine_file), output_file)
        reader = HDF5Reader(spine_file)
        Mx2Hits = Mx2Data(all_files[f_id]) #, output_file)
        spine_eventID, minerva_eventID = find_vertex_ids_all_events(reader,Mx2Hits)
        my_larcv = setup_larcv(output_file)
        #print(spine_eventID, minerva_eventID)
        for spill_id in range(0,len(spine_eventID)):
            spine_index = spine_eventID[spill_id]
            minerva_index = minerva_eventID[spill_id]

            #Find overlapping track IDs, using vertex ID + trackID (+ pdg for completeness)
            min_indices, mx2_unique_IDs = Mx2Hits.find_Mx2_uniqueID(minerva_index)
            spine_data = reader[spine_index]
            
            #Match Processor
            processor = MatchProcessor(particle=True)
            pairs, overlaps = match_reco_clusters_to_truth(
            reco_clusters=spine_data["particle_clusts"],
            truth_particles=spine_data['truth_particles'],
            particle_shapes=spine_data["particle_shapes"],
            get_index_fn=processor.get_index,   # borrow from your existing processor
            min_overlap=0.1
            )
            all_uniqueID_dict, all_2x2_vertexIDs, interaction_dict = find_2x2_uniqueIDs_remaining(spine_data, mx2_unique_IDs)
            if (int(filenum) < 100): # and training_bool:
                n_clusters.append(len(all_uniqueID_dict))
                n_matches.append((len(all_2x2_vertexIDs)+len(mx2_unique_IDs))-len(all_uniqueID_dict))
            print("Spill", spill_id, all_uniqueID_dict)
            #print("Mx2 IDs", mx2_unique_IDs)
            #print("2x2 IDs", all_2x2_vertexIDs)


            #group id is the true particle id, so up and down stream track pieces created by the same particle should have the same group id
            #fragment id is the cluster id, so up and down stream track pieces will have a different fragment id even if created by the same particle
            group_ids = []
            fragment_ids = []
            fragment_counter = 0
            groups_counter = 0
            vsa, vox3meta = make_larcv_box()
            fragment_id, vsa, vox3meta, my_larcv, all_ID_dict = Mx2Hits.process_one_minerva_entry(minerva_index, my_larcv, all_uniqueID_dict, vsa, vox3meta, fragment_counter)
            ## RUN 2x2 PROCESSING HERE #####
            my_larcv = process_one_2x2_entry(spine_data, spine_index, my_larcv, all_uniqueID_dict, vsa, vox3meta, fragment_id)
            nentries = len(all_uniqueID_dict) # made up
            run = 0 #Mx2Hits.mc_run
            subrun = 0 #Mx2Hits.mc_subrun
            my_larcv.set_id( run, subrun, spill_id )
            #my_larcv.save_entry()

        print(n_clusters,n_matches)
        #my_larcv.finalize()
        #write_and_close(my_larcv)
            
            

        #Mx2Hits.process_entries()
        #Mx2Hits.write_and_close()
    
