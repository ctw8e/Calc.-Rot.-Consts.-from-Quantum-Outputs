# -*- coding: utf-8 -*-
"""
Author: Channing West

Changelog: 5/17/2019

This program takes atom positions from Gaussian input files and calculates 
the rotational constants of the molecule before the molecule is optimized. 
This will be useful for identifying candidate structures for optimization
using Gaussian

"""

import numpy as np
import os
import math
from tkinter.filedialog import askdirectory
import contextlib
from contextlib import closing
import pandas as pd

class RotConsts_gjf:
    
    _file_folder = askdirectory()#'C:\\Users\\chann\\OneDrive\\Graduate School\\Pate Group\\Peach Brandy\\Gamma Decalactone\\Gaussian Input'
#    _file_name = 'decalactoneletsgo_c1.gjf'
    
    def __init__(self, file_folder = None, file_name = None):
        if file_folder == None:
            self.file_folder = RotConsts_gjf._file_folder
#        if file_name == None:
        self.file_name = file_name       

    def COM(self):
                
        atomic_mass = {'H_mass': 1.00782503207, 'C_mass': 12, \
                       'N_mass': 14.0030740048, 'O_mass': 15.99491461956, \
                       'Ne_mass': 19.9924401754,'Si_mass': 27.9769265325, \
                       'S_mass': 31.97207100,'Cl_mass': 34.96885268, \
                       'Ar_mass': 39.9623831225, 'Br_mass': 78.9183371}
        
        atom_list = []
        coordinate_list = []
        atom_mass_list = []

        os.chdir(self.file_folder)
        
        f = open(self.file_name)
        with closing(f) as file:
            for i, line in enumerate(file):
                if i > 6:
                    splitline = line.split()
                    if len(splitline) == 0:
                        pass
                    else:
                        atom_list.append(splitline[0])
                        for j in range(1,len(splitline)):
                            coordinate_list.append(float(splitline[j]))
                            
        coordinate_array = np.reshape(np.array(coordinate_list),(len(atom_list),3))
        atom_array = np.array(atom_list)

        
        for atom in range(0, len(atom_array)):
            atom_mass = atomic_mass[atom_array[atom] + '_mass']
            atom_mass_list.append(atom_mass)
            
        molecular_mass = sum(atom_mass_list)
        
        m_r_x_list = []
        m_r_y_list =[]
        m_r_z_list = []
        
        for atom in range(0, len(atom_array)):
            m_r_x_list.append(atom_mass_list[atom] * coordinate_array[atom,0])
            m_r_y_list.append(atom_mass_list[atom] * coordinate_array[atom,1])
            m_r_z_list.append(atom_mass_list[atom] * coordinate_array[atom,2])
            
        sum_m_r_x, sum_m_r_y,sum_m_r_z = sum(m_r_x_list), sum(m_r_y_list), sum(m_r_z_list)
                
        COM_x, COM_y, COM_z = sum_m_r_x/molecular_mass, sum_m_r_y/molecular_mass, sum_m_r_z/molecular_mass
            
        COM = [COM_x, COM_y, COM_z]
                
        COM_coords = np.zeros(np.shape(coordinate_array))
        
        for row in range(0, len(coordinate_array)):
            COM_coords[row,0] = coordinate_array[row,0] - COM_x        
            COM_coords[row,1] = coordinate_array[row,1] - COM_y        
            COM_coords[row,2] = coordinate_array[row,2] - COM_z  

        return atom_mass_list, COM, COM_coords
    
    def Rot_consts(self):
        
        H_8pi2 = 505379.0094
        
        atom_mass_list, COM, COM_coords = self.COM()
        
        I_xx_list = []
        I_yy_list = []
        I_zz_list = []
        I_xy_list = []
        I_xz_list = []
        I_yz_list = []
        
        for atom in range(0, len(atom_mass_list)):          
            I_xx_list.append(atom_mass_list[atom] * (COM_coords[atom,1]**2 + COM_coords[atom,2]**2))
            I_yy_list.append(atom_mass_list[atom] * (COM_coords[atom,0]**2 + COM_coords[atom,2]**2))
            I_zz_list.append(atom_mass_list[atom] * (COM_coords[atom,0]**2 + COM_coords[atom,1]**2))
            I_xy_list.append(atom_mass_list[atom] * COM_coords[atom,0] * COM_coords[atom,1])
            I_xz_list.append(atom_mass_list[atom] * COM_coords[atom,0] * COM_coords[atom,2])
            I_yz_list.append(atom_mass_list[atom] * COM_coords[atom,1] * COM_coords[atom,2])
                    
        I_xx, I_yy, I_zz = sum(I_xx_list), sum(I_yy_list), sum(I_zz_list)
        I_xy, I_xz, I_yz = (-1)*sum(I_xy_list), (-1)*sum(I_xz_list), (-1)*sum(I_yz_list)
                
        matrix = np.zeros((3,3))
        
        matrix[0,0] = I_xx
        matrix[1,1] = I_yy
        matrix[2,2] = I_zz
        matrix[0,1] = I_xy
        matrix[1,0] = I_xy
        matrix[2,0] = I_xz
        matrix[0,2] = I_xz
        matrix[2,1] = I_yz
        matrix[1,2] = I_yz

        eigenvals_ = np.linalg.eigh(matrix)
        eigenvals = eigenvals_[0]
        I_x, I_y, I_z = eigenvals[0], eigenvals[1], eigenvals[2]
        A, B, C = (H_8pi2 / I_x), (H_8pi2 / I_y), (H_8pi2 / I_z)

        return A, B, C
    
    def Rot_consts_folder(self):
        
        name_rot_consts = []
#        file_names = []
        
        for file in os.listdir(self.file_folder):
            if file.endswith('.gjf'):
                self.file_name = file
#                file_names.append(file)
                file = file.replace('.gjf','').split('_c')
                name_rot_consts.append((int(file[1]), self.Rot_consts()))      #self.file_name.replace('.gjf','')
                
        filename_tuple = lambda name_rot_consts: name_rot_consts[0]
        name_rot_consts.sort(key=filename_tuple, reverse=False)
        rot_consts = []
        for element in range(0,len(name_rot_consts)):
            conformer = name_rot_consts[element]
            rot_consts.append(conformer[1])
        rot_consts = np.array(rot_consts)
        column_headers = np.array(('A (MHz)','B (MHz)','C (MHz)'))
        index = pd.Index(range(1,len(rot_consts)+1))

        rot_consts = pd.DataFrame(rot_consts, index=index, columns=column_headers)

        print(rot_consts)

        
                
instance = RotConsts_gjf()
instance.Rot_consts_folder()
#instance.COM()