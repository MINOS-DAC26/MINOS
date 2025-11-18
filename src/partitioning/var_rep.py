# var_rep.py

import numpy as np
import json

class Var:
    def __init__(self, name, dimensions, lower_bound, upper_bound, type=int, index=None):
        self.name = name
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.index = np.array(index).astype('int').tolist() if index is not None else "None"
        if type == int or type == float:
            self.type = type
        else:
            raise ValueError("Type must be either 'int' or 'float'")


    def __repr__(self):
        return f"Variable(name={self.name}, dimensions={self.dimensions}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, type={self.type}, index={self.index})"


class ModelVars:
    def __init__(self, matrix_load_vars=False):
        self.vars = {}
        self.starts = {}
        self.matrix_load_vars = matrix_load_vars


    def add_var(self, var):
        if not isinstance(var, Var):
            raise TypeError("Only Var instances can be added")
        self.vars[var.name] = var
        

    def get_dim(self, var_name):
        return self.vars[var_name].dimensions
    

    def add_start(self, var_name, start):
        if var_name not in self.vars.keys():
            raise ValueError(f"Variable {var_name} not found in vars")
        self.starts[var_name] = start


    def get_var_by_name(self, name):
        for var in self.vars:
            if var.name == name:
                return var
        return None


    def vars_to_json(self):
        return {var_name: {'dimensions': var.dimensions, 'lower_bound': var.lower_bound, 'upper_bound': var.upper_bound, 'type': var.type.__name__, 'index': (var.index if var.index is not None else "None")} for var_name, var in self.vars.items()}


    def to_json(self, file_name):
        vars_json = self.vars_to_json()
        vars_json.update({'matrix_load_vars': self.matrix_load_vars})
        with open(file_name, 'w') as f:
            json.dump(vars_json, f)


    def from_json(self, file_name):
        with open(file_name, 'r') as f:
            vars_json = json.load(f)
        self.matrix_load_vars = vars_json['matrix_load_vars']
        vars_json.pop('matrix_load_vars')
        for var_name, var in vars_json.items():
            type = int if var['type'] == 'int' else float
            index = None if var['index']=="None" else var['index']
            self.add_var(Var(var_name, var['dimensions'], var['lower_bound'], var['upper_bound'], type, index))


    def __repr__(self):
        return f"ModelVars(vars={self.vars}, starts={self.starts})"
    

    def all_vars_have_starts(self):
        for var_name in self.vars:
            if var_name not in self.starts.keys():
                return False
        return True
    
    
    def write_var_starts(self, fheader, heuristic):
        file_name = fheader + f"_{heuristic}.mst"
        with open(file_name, 'w') as f:
            f.write(f"# Variable starts from {heuristic}\n")

            for var_name, start in self.starts.items():
                var = self.vars[var_name]
                dims = var.dimensions
                sub_index = var.index

                # Case 1: Scalar (0-dimensional)
                if dims == () or dims == 0 or dims == (0,) or dims == []:
                    if isinstance(start, (int, float, np.integer, np.floating)):
                        value = start
                    elif isinstance(start, np.ndarray) and start.size == 1:
                        value = start.item()
                    else:
                        continue
                    f.write(f"{var_name} {var.type(value)}\n")
                    continue

                # Case 2: Multi-dimensional
                for index, value in np.ndenumerate(start):
                    if sub_index == "None":
                        idx_tuple = index
                    else:
                        idx_tuple = tuple(sub_index[index[0]])
                    idx_str = ",".join(str(i) for i in idx_tuple)
                    f.write(f"{var_name}[{idx_str}] {var.type(value)}\n")

        return file_name
