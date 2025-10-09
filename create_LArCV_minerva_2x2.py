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
sys.path.insert(0, '/sdf/data/neutrino/software/spine/src/')

# Necessary imports
from spine.driver import Driver
#This is code that takes all the MINERvA root files in a directory and creates a LARCV files that can be input into GrapPA

#Class for the data of the current file being input
class Mx2Data:
    def __init__(self, filename, output_filename):

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
        self.out_larcv = larcv.IOManager( larcv.IOManager.kWRITE )
        # use the following for more verbose description of iomanager's actions
        #self.out_larcv.set_verbosity( larcv.msg.kINFO )
        #self.out_larcv.set_verbosity( larcv.msg.kDEBUG )
        self.out_larcv.set_out_file( output_filename )
        self.out_larcv.initialize()

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
        max_energy_particle_id = max(particle_energy, key=particle_energy.get)
        edep_traj_name = self.mc_traj_edepsim_trkid[entry][max_energy_particle_id]
        edep_traj_evtid = self.mc_traj_edepsim_eventid[entry][max_energy_particle_id]
        mc_traj_trkid = self.mc_traj_trkid[entry][max_energy_particle_id]
        mc_traj_pdg = Mx2Hits.mc_traj_pdg[entry][max_energy_particle_id]
        #print(edep_traj_evtid, mc_traj_trkid, mc_traj_pdg)

        return mc_traj_pdg, edep_traj_name, edep_traj_evtid

    def find_Mx2_vertices(self, entry):

        unique_id_store = []
        
        for idx in self.trk_index[entry]:        
            n_nodes = self.trk_nodes[entry][idx]
            if ((n_nodes >0)):
                edep_pdg, edep_traj_name, edep_traj_evtid = self.get_most_energetic_particle(entry, idx)
                unique_id_store.append(edep_traj_evtid)
         
        minerva_vertices = [str(val) for val in set(unique_id_store)]

        return minerva_vertices

    def process_entries(self):
        """
        function responsible for managing loop over entries
        """
        nentries = len(self.trk_node_X) # made up
        run = 0 #self.mc_run
        subrun = 0 #self.mc_subrun

        for entry in range(nentries):
            # convert input data into larcv data for one entry
            fail = self.process_one_minvera_entry(entry)
            if not fail:
                # set the entry id
                self.out_larcv.set_id( run, subrun, entry )
                # save the entry data
                self.out_larcv.save_entry()
            #else:
            #    print("Not writing", entry)

    def process_one_minerva_entry(self,entry):
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
        
        #group id is the true particle id, so up and down stream track pieces created by the same particle should have the same group id
        #fragment id is the cluster id, so up and down stream track pieces will have a different fragment id even if created by the same particle
        group_ids = []
        fragment_ids = []
        
        #Manually making a bounding box around both upstream and downstream MINERvA planes, in the 2x2 coordinate frame
        xmin = (-1080.0)
        ymin = (-1450.0)
        zmin = (-2400.0)
        xmax = (1080.0) 
        ymax = (1000.0) 
        zmax = (3100.0)        
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
        
        fragment_counter = 0
        groups_counter = 0
        #print("Entry: ", entry)
        for idx in self.trk_index[entry]:
            n_nodes = self.trk_nodes[entry][idx]
            if ((n_nodes >0)):
               #This array keeps track of up vs downstream nodes, helps to distinguish between clusters
                Us_Ds=[]
        
                x_nodes = Mx2Hits.trk_node_X[entry][idx][:n_nodes] - offsetX
                y_nodes = Mx2Hits.trk_node_Y[entry][idx][:n_nodes] - offsetY 
                z_nodes = Mx2Hits.trk_node_Z[entry][idx][:n_nodes] - offsetZ
                any_us = sum(z_nodes < 0) > 0 #Any nodes upstream?
                any_ds = sum(z_nodes > 0) > 0 #Any nodes downstream?
                assert sum(z_nodes< 0)+sum(z_nodes > 0) == len(z_nodes), "Any upstream vs downstream should be mutually exclusive"

                #Finding the trajectory that contributed the most
                #get_most_energetic_particle(self, traj_list, hit_energy_list, entry)
            
                fragment_id=fragment_counter #fragment_ids_shuffled[fragment_counter]
                group_id=groups_counter
                fragment_counter += 1
                groups_counter += 1
                #print("Fragment: ", fragment_id, "Group: ", group_id)
                
                #Create particle larcv object. Each cluster will need its own particle object, even if the clusters belong to one particle.
                #Since this code only consider MINERvA tracks, every particle is assigned a track shape
                particle = larcv.Particle(larcv.ShapeType_t.kShapeTrack)
                particle.id(int(fragment_id))
                particle.group_id(int(group_id))

                #A voxel set corresponds to all the voxels for one cluster 
                # fill voxelset 
                track_as_voxelset = larcv.VoxelSet()
                track_as_voxelset.id(int(fragment_id))
                
                #Create a second particle, if needed
                particle2 = larcv.Particle(larcv.ShapeType_t.kShapeTrack)

                #A voxel set corresponds to all the voxels for one cluster 
                # fill voxelset 
                track_as_voxelset2 = larcv.VoxelSet()

                    
                for edepvoxels in range(n_nodes):
                    voxelid = vox3dmeta.id( x_nodes[edepvoxels], y_nodes[edepvoxels], z_nodes[edepvoxels] )
                    voxel = larcv.Voxel( voxelid )
                    if voxelid!=larcv.kINVALID_VOXELID: #Skip if invalid
                        track_as_voxelset.add( voxel )

                # add voxelset to container
                vsa.insert( track_as_voxelset )

                # get the cluster3d entry container, by contributing VoxelSetArray and the Voxel3Dmeta
                entry_clust3d = self.out_larcv.get_data( "cluster3d", "pcluster" )
                entry_clust3d.set( vsa, vox3dmeta )
                
                #get particle container and fill
                entry_particles = self.out_larcv.get_data( "particle", "corrected")
                entry_particles.append(particle)


                    
        
        return 0

    def write_and_close(self):
        self.out_larcv.finalize()


def find_2x2_vertices(spine_driver, entry):
    
    data = spine_driver.process(entry)
    truth_interactions = data[f'truth_interactions']
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

def find_vertex_ids_all_events(spine_driver,Mx2Hits):
    minerva_indices_save = []
    nd_indices_save = []
    total_events_spine = len(spine_driver)
    total_events_mx2 = len(Mx2Hits.n_tracks)
    assert total_events_spine < total_events_mx2 #spine should be more "pruned" than minerva file

    #Get all vertices per event
    for i in range(0,total_events_spine):
        #print("ENTRY ", i)
        min_indices = Mx2Hits.find_Mx2_vertices(i)
        minerva_indices_save.append(min_indices)

        nd_indices = find_2x2_vertices(spine_driver, i)
        nd_indices_save.append(nd_indices)
    for i in range(total_events_spine,total_events_mx2):
        min_indices = Mx2Hits.find_Mx2_vertices(i)
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


if __name__ == "__main__":
    directory = "/sdf/home/j/jessicam/Mx2/data/"

    training_output = directory+"larcv/"
    validation_output = directory+"larcv/validation_set/"

    # List all .root files in the specified directory
    training_input_files = glob.glob(os.path.join(directory+"minerva/", "*.root"))
    testing_input_files = glob.glob(os.path.join(directory+"minerva/validation_set/", "*.root"))
    all_files = training_input_files + testing_input_files 
    num_training = len(training_input_files)
    #all_files = all_files[:1] 
    print(training_input_files, len(all_files))

    for f_id in range(0,len(all_files)):
        base_name = os.path.basename(all_files[f_id])
        # extract the 7-digit number (or any sequence of digits)
        match = re.search(r'\.(\d{7})\.', base_name)
        if match:
            filenum = match.group(1)
            print(filenum)  # 0000101
        spine_file = "/sdf/data/neutrino/sindhuk/MR6.4/v2/MiniRun6.4_1E19_RHC.spine_v2."+filenum+".MLRECO_SPINE.hdf5"

        if f_id < num_training:
            output_file = os.path.join(training_output, f"out_{base_name}")
            training_bool = True
        else:
            output_file = os.path.join(validation_output, f"out_{base_name}")
            training_bool = False
        
        print(base_name, os.path.basename(spine_file), output_file)
        cfg = f'''
        base:
          verbosity: warning
        build:
          mode: both
          fragments: false
          particles: true
          interactions: true
          units: cm
        io:
          reader:
            file_keys: {spine_file}
            skip_unknown_attrs: True
            name: hdf5
        '''
        driver = Driver(yaml.safe_load(cfg))
        Mx2Hits = Mx2Data(all_files[f_id], output_file)
        spine_eventID, minerva_eventID = find_vertex_ids_all_events(driver,Mx2Hits)
        print(spine_eventID, minerva_eventID)
        #Mx2Hits.process_entries()
        #Mx2Hits.write_and_close()
    
