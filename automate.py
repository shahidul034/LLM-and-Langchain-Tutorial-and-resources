import argparse
import paramiko
import sys
import subprocess
import re
import time
from tqdm import tqdm

def get_empty_gpu(ssh, conda_init_path):
    try:
        # Determine Python executable path from Conda
        python_path = "/home/mshahidul/miniconda3/bin/python"
        
        # Verify Python executable exists
        stdin, stdout, stderr = ssh.exec_command(f"test -f {python_path} && echo 'exists'")
        if stdout.read().decode().strip() != 'exists':
            return None, f"Error: Python executable {python_path} does not exist on the server"
        
        # GPU selection script to run on the remote server
        gpu_script = """
import subprocess
import sys

def gpu_info():
    dat = []
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    out2 = out.split("\\n")[8:]
    splitter = out2[3]
    out3 = "\\n".join(out2).split(splitter)[:-1]
    
    for idx, x in enumerate(out3):
        gpu = ((x.split("|"))[1].replace("Off", ""))
        t = ((x.split("|"))[6].replace("Off", "").replace("MiB", "").split("/"))
        ans = (int(t[0]) / int(t[1])) * 100
        dat.append({
            "gpu_id": idx,
            "gpu": ((gpu[:-5])[6:]).strip(),
            "usage": f"{ans:.2f}"
        })
    return dat

gpu_info_ans = gpu_info()
flag = 0
for gpu in gpu_info_ans:
    if float(gpu['usage']) < 10:
        empty_gpu = gpu['gpu_id']
        flag = 1
        print(f"Using GPU {empty_gpu} for processing.")
        sys.exit(0)
if flag == 0:
    print("No empty GPU found, please check your GPU usage.")
    sys.exit(1)
"""
        # Write the script to a temporary file on the remote server
        temp_script_path = "/tmp/gpu_selection.py"
        sftp = ssh.open_sftp()
        with sftp.file(temp_script_path, 'w') as f:
            f.write(gpu_script)
        sftp.close()
        
        # Execute the GPU selection script using Conda's Python
        stdin, stdout, stderr = ssh.exec_command(f"{python_path} {temp_script_path}")
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        
        # Clean up the temporary file
        ssh.exec_command(f"rm {temp_script_path}")
        
        # Check the output
        if "No empty GPU found" in output:
            return None, error
        elif "Using GPU" in output:
            gpu_id = output.split("Using GPU ")[1].split(" ")[0]
            return gpu_id, None
        else:
            return None, f"Unexpected output from GPU selection: {output}\n{error}"
            
    except Exception as e:
        return None, f"Error in GPU selection: {str(e)}"

def run_script(server, cuda_gpu, script_path, virtual_env):
    try:
        # Server details
        servers = {
            "omega": "ss@172.16.34.1",
            "beta": "ss@172.16.34.22",
            "alpha": "ss@172.16.34.21",
            "gamma": "ss@172.16.34.29"
        }
        
        if server not in servers:
            return f"Error: Invalid server selected: {server}"
        
        print("Initializing SSH client...")
        # Initialize SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Extract username and host
        username, host = servers[server].split('@')
        
        print(f"Connecting to server {server} ({host})...")
        # Connect to server
        ssh.connect(hostname=host, username=username, timeout=10)
        
        # Determine environment activation command
        default_conda_init_path = "/home/mshahidul/miniconda3/etc/profile.d/conda.sh"
        if virtual_env:
            print("Checking Conda virtual environment...")
            # Extract Conda init path and environment name
            if "source" in virtual_env:
                # E.g., "source /path/to/conda.sh && conda activate my_env"
                parts = virtual_env.strip().split('&&')
                conda_init_path = parts[0].strip().split('source')[-1].strip()
                env_name = parts[1].strip().split('conda activate')[-1].strip()
                env_command = virtual_env
            else:
                # E.g., "conda activate my_env"
                conda_init_path = default_conda_init_path
                env_name = virtual_env.strip().split('conda activate')[-1].strip()
                env_command = f"source {conda_init_path} && {virtual_env}"
        else:
            print("Checking default Conda initialization file...")
            # Use default Conda environment
            conda_init_path = default_conda_init_path
            env_name = "unsloth"
            env_command = f"source {conda_init_path} && conda activate {env_name}"
        
        # Check if Conda init file exists
        stdin, stdout, stderr = ssh.exec_command(f"test -f {conda_init_path} && echo 'exists'")
        if stdout.read().decode().strip() != 'exists':
            ssh.close()
            return f"Error: Conda initialization file {conda_init_path} does not exist on the server"
        
        # Check if Conda environment exists
        stdin, stdout, stderr = ssh.exec_command(f"source {conda_init_path} && conda env list")
        env_list = stdout.read().decode().strip()
        if env_name not in env_list:
            ssh.close()
            return f"Error: Conda environment '{env_name}' does not exist on the server"
        
        # Determine CUDA GPU if not provided
        if cuda_gpu is None:
            print("No CUDA GPU specified, selecting an empty GPU...")
            selected_gpu, error = get_empty_gpu(ssh, conda_init_path)
            if error:
                ssh.close()
                return error
            if selected_gpu is None:
                ssh.close()
                return "No empty GPU found, please check your GPU usage."
            cuda_gpu = selected_gpu
            print(f"Selected GPU: {cuda_gpu}")
        
        print("Executing script...")
        # Construct command
        command = f"""
        {env_command} && 
        export CUDA_VISIBLE_DEVICES={cuda_gpu} && 
        python {script_path}
        """
        
        # Execute command
        stdin, stdout, stderr = ssh.exec_command(command)
        
        print("Collecting output...")
        # Simulate progress bar since we can't track SSH command progress directly
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        # Close SSH connection
        ssh.close()
        
        # Print results
        result = "Output:\n" + output
        if error:
            result += "\nErrors:\n" + error
            
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a Python script on a remote server with specified or auto-selected CUDA GPU and Conda environment")
    parser.add_argument("--server", required=True, choices=["omega", "beta", "alpha", "gamma"], help="Server to run the script on")
    parser.add_argument("--cuda_gpu", default=None, type=str, help="CUDA GPU number(s) to use (e.g., '0' or '0,1'); if not provided, an empty GPU will be selected")
    parser.add_argument("--script", required=True, help="Path to Python script with arguments (e.g., '/path/to/script.py --arg')")
    parser.add_argument("--virtual_env", default=None, help="Conda environment activation command (e.g., 'conda activate my_env' or 'source /path/to/conda.sh && conda activate my_env')")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the script and print the result
    result = run_script(args.server, args.cuda_gpu, args.script, args.virtual_env)
    print(result)

if __name__ == "__main__":
    main()