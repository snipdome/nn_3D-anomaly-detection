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


import importlib, sys

def call_external_code(module: str, fun: str, **kwargs ):
    pkg = importlib.import_module(module)
    cls = getattr(pkg, fun)
    try:
        res = cls(**kwargs)
    except TypeError as e:
        kwargs.pop('gpu_list') # gpu_list tells where the argument were
        res = cls(**kwargs)
    if isinstance(res, dict):
        return res
    else:
        return {'res':res}