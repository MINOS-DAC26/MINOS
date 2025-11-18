# problem_rep.py

import json

class Problem:
    def __init__(self, **params):
        self.params = params  


    def to_json(self):
        return json.dumps(self.params)


    @classmethod
    def from_json(cls, json_str):
        params = json.loads(json_str)
        return cls(**params)


    def get_solution_key(self):
        key_parts = [f"{k}={v}" for k, v in self.params.items()]
        return "_".join(key_parts)
    

    def get_model_key(self, **overrides):
        excluded_keywords = {"id", "com", "coe", "wts", "wcs", "po", "t", "cs"}
        params = self.params.copy()
        params.update(overrides)
        
        key_parts = [
            f"{k}={v}" for k, v in params.items()
            if not any(excluded_word == k.lower() for excluded_word in excluded_keywords)
            and not (k.lower() == "sc" and v in (0, 1.0))
        ]
        return "_".join(key_parts)
    

    @classmethod
    def from_solution_key(cls, key):
        params = {}
        for part in key.split("_"):
            if "=" in part:
                k, v = part.split("=", 1)
                if v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v) if "." in v and v.replace(".", "", 1).isdigit() else v
                    except ValueError:
                        pass
                params[k] = v
        return cls(**params)
    
    
    def update_values(self, **updates):
        new_params = self.params.copy()  
        new_params.update(updates) 
        return Problem(**new_params)