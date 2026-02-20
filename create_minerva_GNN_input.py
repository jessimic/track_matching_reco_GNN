import ROOT as rt
import uproot
from larcv import larcv
import math
import numpy as np
import glob
import os
from shutil import move
import random

#This is code that takes all the MINERvA root files in a directory and creates a LARCV files that can be input into GrapPA

#Class for the data of the current file being input
class Mx2Data:
    def __init__(self, filename, output_filename):

        # setup input
        
        self.file = uproot.open(filename)
        
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
        self.mc_traj_point_t = self.file["minerva"]["mc_traj_point_t"].array(library="np")
        

        # setup iomanager for output mode
        # IOManager class declaration in larcv/core/DataFormat/IOManager.h        
        self.out_larcv = larcv.IOManager( larcv.IOManager.kWRITE )
        # use the following for more verbose description of iomanager's actions
        #self.out_larcv.set_verbosity( larcv.msg.kINFO )
        #self.out_larcv.set_verbosity( larcv.msg.kDEBUG )
        self.out_larcv.set_out_file( output_filename )
        self.out_larcv.initialize()

    def process_entries(self):
        """
        function responsible for managing loop over entries
        """
        nentries = len(self.trk_node_X) # made up
        run = 0
        subrun = 0
        for entry in range(nentries):
            # convert input data into larcv data for one entry
            self.process_one_entry(entry)
            # set the entry id
            self.out_larcv.set_id( run, subrun, entry )
            # save the entry data
            self.out_larcv.save_entry()

    def process_one_entry(self,entry):
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
        offsetX = Mx2Hits.offsetX[entry]
        offsetY = Mx2Hits.offsetY[entry]
        offsetZ = Mx2Hits.offsetZ[entry]

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

        # define the meta
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
        
        #group id is the true particle id, so up and down stream track pieces created by the same particle should have the same group id
        #fragment id is the cluster id, so up and down stream track pieces will have a different fragment id even if created by the same particle
        group_id=-1
        fragment_id=-1
        group_id_comp=[]
        fragment_id_comp=[]
        
        #Loop over the tracks in the event
        for idx in Mx2Hits.trk_index[entry]:
            inverse_idx=Mx2Hits.trk_index[entry][-(idx+1)]
            #number of nodes
            n_nodes = Mx2Hits.trk_nodes[entry][idx]

            if ((n_nodes >0)):
                #This array keeps track of up vs downstream nodes, helps to distinguish between clusters
                Us_Ds=[]
        
                x_nodes = Mx2Hits.trk_node_X[entry][idx][:n_nodes] - offsetX
                y_nodes = Mx2Hits.trk_node_Y[entry][idx][:n_nodes] - offsetY 
                z_nodes = Mx2Hits.trk_node_Z[entry][idx][:n_nodes] - offsetZ
                #print("fragment_id pre increment: ", fragment_id)
                #print("frag id post increment: ", fragment_id)
                #print("group_id: ", group_id)

                track_energy=Mx2Hits.trk_vis_energy[entry][idx]
                node_energy=track_energy/n_nodes

                #selecting all the MC trajectories of the spill
                traj_x = Mx2Hits.mc_traj_point_x[entry] #each entry is one trajectory 
                traj_y = Mx2Hits.mc_traj_point_y[entry]
                traj_z = Mx2Hits.mc_traj_point_z[entry]

                #placeholders for the track_ids to consider for the track
                mc_Mx2_truth={}
                mc_Mx2_truth["track_id"] = []
                mc_Mx2_truth["traj_x"] = []
                mc_Mx2_truth["traj_y"] = []
                mc_Mx2_truth["traj_z"] = []

                trk_time_slice = (Mx2Hits.trk_time_slice[entry]) 

                clus_id_z = (Mx2Hits.clus_id_z[entry])

                cl_list = Mx2Hits.trk_node_cluster_idx[entry][idx] # select the clusters associated with the nodes of the track
                cl_list = cl_list[cl_list>=0]
                clus_z = clus_id_z[cl_list]  #  select the Z positions of the clusters associated with the nodes of the track
                cl_size = Mx2Hits.clus_id_size[entry][cl_list] # Number of digits that were clustered for each of those clusters
                hit_list = Mx2Hits.clus_id_hits_idx[entry][cl_list] # Lists of digits that were clustered for each of those clusters

                hit_list = hit_list[hit_list>=0]

                hit_energy_list = Mx2Hits.mc_id_mchit_dE[entry][hit_list] # Energy deposited
                nhits = Mx2Hits.mc_id_nmchit[entry][hit_list]
                nhits = np.where(nhits>2,2,nhits) # Each digits is connsidered to be at most 2 true MC hits

                traj_list = np.concatenate([Mx2Hits.mc_id_mchit_trkid[entry][hit_list][i][:nhits[i]] for i in range(len(nhits))]) # Get the list of all trajectories that contributed to the track.
                # traj_list = Mx2Hits.mc_id_mchit_trkid[entry][hit_list][:,0]

                # Getting the trajectories that contributed
                traj_list = traj_list[traj_list>0]
                hit_energy_list = hit_energy_list[hit_energy_list>0]
                #Finding the trajectory that contributed the most
                particle_energy = {}
                for i in range(len(traj_list)):
                    particle_id = traj_list[i]
                    energy = hit_energy_list[i]
                    if particle_id in particle_energy:
                        particle_energy[particle_id] += energy
                    else:
                        particle_energy[particle_id] = energy
                if len(particle_energy) == 0:
                    #print("Is this the problem?")
                    continue
                max_energy_particle_id = max(particle_energy, key=particle_energy.get)
                edep_traj_name = Mx2Hits.mc_traj_edepsim_trkid[entry][max_energy_particle_id]
                edep_traj_evtid = Mx2Hits.mc_traj_edepsim_eventid[entry][max_energy_particle_id]
                mc_traj_trkid = Mx2Hits. mc_traj_trkid[entry][max_energy_particle_id]
            
                # print(max_energy_particle_id, edep_traj_name,edep_traj_evtid, mc_traj_trkid)
            
                # Getting the Start and End position of the trajectory that contributed the most to the track.
                fragment_id+=1
                group_id+=1

                x_traj = [traj_x[max_energy_particle_id][0], traj_x[max_energy_particle_id][1]]
                y_traj = [traj_y[max_energy_particle_id][0], traj_y[max_energy_particle_id][1]]
                z_traj = [traj_z[max_energy_particle_id][0], traj_z[max_energy_particle_id][1]]

                x_traj = np.array(x_traj) - offsetX
                y_traj = np.array(y_traj) - offsetY
                z_traj = np.array(z_traj) - offsetZ
            
                mc_Mx2_truth["track_id"] = np.ones(len(x_traj))*edep_traj_name
                mc_Mx2_truth["traj_x"] = x_traj
                mc_Mx2_truth["traj_y"] = y_traj
                mc_Mx2_truth["traj_z"] = z_traj
            
                start_traj=[x_traj[0], y_traj[0], z_traj[0]]
                end_traj=[x_traj[1], y_traj[1], z_traj[1]]
                
                #Create particle larcv object. Each cluster will need its own particle object, even if the clusters belong to one particle.
                #Since this code only consider MINERvA tracks, every particle is assigned a track shape
                particle = larcv.Particle(larcv.ShapeType_t.kShapeTrack)
                particle.id(int(fragment_id))
                particle.group_id(int(group_id))

                #A voxel set corresponds to all the voxels for one cluster 
                # fill voxelset 
                track_as_voxelset = larcv.VoxelSet()
                track_as_voxelset.id(int(fragment_id))

                fragment_id_comp.append(track_as_voxelset.id())
                fragment_id_comp.append(particle.id())

                group_id_comp.append(particle.group_id())

                #print("Fragment ID in voxelset (or fragment): ", track_as_voxelset.id())
                #print("Fragment ID in particle: ", particle.id())
                #print()
                # print(idx)

                #Loop over the nodes (or hits) and 
                for edepvoxels in range(n_nodes):
                    voxelid = vox3dmeta.id( x_nodes[edepvoxels], y_nodes[edepvoxels], z_nodes[edepvoxels] )

                    #if voxelid==18446744073709551615:    
                        #print("x,y,z: ", x_nodes[edepvoxels], y_nodes[edepvoxels], z_nodes[edepvoxels])
                    
                    if z_nodes[edepvoxels]>0:
                        #downstream
                        Us_Ds.append(1)
                    if z_nodes[edepvoxels]<0:
                        #upstream
                        Us_Ds.append(0)
                        
                    if Us_Ds[edepvoxels-1]!=Us_Ds[edepvoxels]:
                        
                        
                        #print("Transition from up to down")
                        #print("fragment_id: ", fragment_id)
                        #print("group_id: ", group_id)

                        #get particle container and fill
                        entry_particles = self.out_larcv.get_data( "particle", "corrected")
                        entry_particles.append(particle)

                        # add voxelset to container
                        vsa.insert( track_as_voxelset )
    
                        # get the cluster3d entry container, by contributing VoxelSetArray and the Voxel3Dmeta
                        entry_clust3d = self.out_larcv.get_data( "cluster3d", "pcluster" )
                        entry_clust3d.set( vsa, vox3dmeta )
                        
                        fragment_id+=1
                        
                        fragment_id_comp.append(track_as_voxelset.id())
                        fragment_id_comp.append(particle.id())

                        group_id_comp.append(particle.group_id())

                        # fill voxelset
                        track_as_voxelset = larcv.VoxelSet()
                        track_as_voxelset.id(int(fragment_id))

                        #Create particle
                        particle = larcv.Particle(larcv.ShapeType_t.kShapeTrack)
                        particle.id(int(fragment_id))
                        particle.group_id(int(group_id))

                        #print("index: ", idx)
                        #print("Particle group (true) id: ", particle.group_id())
                        #print()
                        #print("Fragment ID in voxelset (or fragment): ", track_as_voxelset.id())
                        #print("Fragment ID in particle: ", particle.id())
                        #print()
                    voxel = larcv.Voxel( voxelid )
                    #print("voxel: ", voxel)
                    if voxelid!=18446744073709551615: #Some failure mode?
                        track_as_voxelset.add( voxel )
                #for i in range(0, len(fragment_id_comp)-1, 2):
                    #if (fragment_id_comp[i]!=fragment_id_comp[i+1]):
                        #print()
                        #print("idx: ", idx)
                        #print("mismatching fragment ids", fragment_id_comp[i], fragment_id[i+1])
                #for i in range(0, len(group_id_comp)-1, 2):
                    #if (group_id_comp[i]!=group_id_comp[i+1]):
                        #print()
                        #print("idx: ", idx)
                        #print("group_ids: ", group_id_comp)
                        #print("mismatching group ids: ", group_id_comp[i], group_id_comp[i+1])

                # add voxelset to container
                vsa.insert( track_as_voxelset )

                # get the cluster3d entry container, by contributing VoxelSetArray and the Voxel3Dmeta
                entry_clust3d = self.out_larcv.get_data( "cluster3d", "pcluster" )
                entry_clust3d.set( vsa, vox3dmeta )

                #get particle container and fill
                entry_particles = self.out_larcv.get_data( "particle", "corrected")
                entry_particles.append(particle)
                #print()
                #print("entry_particles size: ", entry_particles.size())
                #print("entry_clust3d size: ", entry_clust3d.size())
                #print("index: ", idx)
                #print("Particle group (true) id: ", particle.group_id())
                #print("Fragment ID in voxelset (or fragment): ", track_as_voxelset.id())
                #print("Fragment ID in particle: ", particle.id())
                #print()


        #print("Entry: ", entry)
        #print()
                #if entry==0 and idx == Mx2Hits.trk_index[entry][-1]:
            #print()
         #   print("entry_particles size: ", entry_particles.size())
          #  print("entry_clust3d size: ", entry_clust3d.size())
            #print()
                    #print("Tracks: ", (Mx2Hits.trk_index[entry][-1]+1))
                    #print("Clusters: ", None)
                #if entry==0:
                    #print("Us_Ds: ", Us_Ds)

    def write_and_close(self):
        self.out_larcv.finalize()

if __name__ == "__main__":
    directory = "/n/holystore01/LABS/iaifi_lab/Users/jmicallef/data_2x2/minerva/"
    training_directory = "/n/holyscratch01/iaifi_lab/Users/nsantiago/inputfiles"
    validation_directory = "/n/holyscratch01/iaifi_lab/Users/nsantiago/validationfiles"

    # List all .root files in the specified directory
    input_files = glob.glob(os.path.join(directory, "*.root"))

    # Randomly select 20% of the files for validation
    validation_files = random.sample(input_files, int(0.2 * len(input_files)))

    for input_file in input_files:
        base_name = os.path.basename(input_file)
        print(base_name)

        if base_name not in (
            "MiniRun5_1E19_RHC.minerva.0000289.dst.root",#Dysfunctional files 1-4
            "MiniRun5_1E19_RHC.minerva.0000172.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000705.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000240.dst.root",
            "MiniRun5_1E19_RHC.minerva.0000139.dst.root",
            "MiniRun5_1E19_RHC.minerva.0000156.dst.root",
            "MiniRun5_1E19_RHC.minerva.0000088.dst.root",#Test Files 5-end
            "MiniRun5_1E19_RHC.minerva.0000208.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000314.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000934.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000092.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000209.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000316.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000096.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000215.dst.root", 
            "MiniRun5_1E19_RHC.minerva.0000318.dst.root"):
            
            if input_file in validation_files:
                validation_file = os.path.join(validation_directory, f"out_{base_name}")
                Mx2Hits = Mx2Data(input_file, validation_file)
                Mx2Hits.process_entries()
                Mx2Hits.write_and_close()
            else:
                training_file = os.path.join(training_directory, f"out_{base_name}")
                Mx2Hits = Mx2Data(input_file, training_file)
                Mx2Hits.process_entries()
                Mx2Hits.write_and_close()
    # Print the number of files moved to validation directory
    final_validation_files = len(glob.glob(os.path.join(validation_directory, "*.root")))
    print(f"Number of files in validation directory: {final_validation_files}")

    # Print the number of files processed in training directory
    final_training_files = len(glob.glob(os.path.join(training_directory, "*.root")))
    print(f"Number of files in training directory: {final_training_files}")

