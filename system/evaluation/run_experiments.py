
import subprocess
import json
import sys

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
        experiments =  json.load(fp)["experiments"]
        # load specific experiment instance
        for experiment in experiments:

            exp_name = experiment ["exp_name"]
            
            for eval_filename in experiment ["eval_data"]:

                for run in experiment ["runs"]:

                    run_name = run ["run_name"]

                    pipeline_path = run ["pipeline_path"]

                    subprocess.run(["python3", "evaluate.py", exp_name, eval_filename, pipeline_path, run_name])