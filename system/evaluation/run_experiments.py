
from typing import List, Optional, Dict, Any, Union, Callable, Tuple
import subprocess
import json
import sys
import logging

def validate_experiment_dictionary(experiment_dict: Dict[str, any]):
    """
    Validate the schema of an experiment configuration dictionary.

    experiment_dict: The dictionary containing experiment configuration.
    """
    required_keys = ["exp_name", "run_name", "pipe_path"]
    
    logging.info("Validating experiment configuration dictionary.")

    for key in required_keys:
        if key not in experiment_dict:
            logging.error(f"Key '{key}' is missing in the dictionary.")
            raise ValueError(f"Key '{key}' is missing in the dictionary.")
        elif not experiment_dict[key]:
            logging.error(f"Value for key '{key}' is empty.")
            raise ValueError(f"Value for key '{key}' is empty.")

try:
    subprocess.run(["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"], check=True)
except subprocess.CalledProcessError as e:
    error_output = str(e.output)
    # Check if the error message indicates that the connection is in use
    if "Connection in use" in error_output:
        print("Port 5000 is already in use.")
    else:
        print("Error starting MLflow server:", e)
else:
    print("MLflow server started successfully")
    
if __name__ == "__main__":
    
    config_filename = sys.argv[1]
    
    with open (config_filename, "r") as fp:
        # load expirement set
        exp_dict = json.load(fp)
        validate_experiment_dictionary(exp_dict)

        experiments =  exp_dict["experiments"]
        # load specific experiment instance
        for experiment in experiments:

            exp_name = experiment ["exp_name"]
            
            for eval_filename in experiment ["eval_data"]:

                for run in experiment ["runs"]:

                    run_name = run ["run_name"]

                    pipeline_path = run ["pipeline_path"]

                    subprocess.run(["python3", "evaluate.py", exp_name, eval_filename, pipeline_path, run_name])
