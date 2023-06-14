# 
# This file is part of the nn_3D-anomaly-detection distribution (https://github.com/snipdome/nn_3D-anomaly-detection).
# Copyright (c) 2022-2023 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import re, os
 
shown_warning_extractSliceNumber = False

# Function to extract all the numbers from the given string
def extractSliceNumber(stringa):
    stringa = os.path.basename(stringa)
    numeri =  re.findall(r'[0-9]+', stringa)
    if len(numeri) != 1 and not shown_warning_extractSliceNumber:
        print('WARNING: function extractSliceNumber found more than a number in filename of input files. Taking last number as filename value')
        shown_warning_extractSliceNumber = True
    numeri = numeri[-1]
    return int(numeri)