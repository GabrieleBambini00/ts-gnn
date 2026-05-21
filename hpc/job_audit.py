import paramiko
import sys

def check_hpc():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect("slogin.hpc.unibocconi.it", username="3393519", password="barracuda", timeout=15)
        
        print("--- REMOTE FILE STRUCTURE ---")
        stdin, stdout, stderr = ssh.exec_command("find ~/ts-gnn/ -maxdepth 3")
        print(stdout.read().decode())
        
        print("\n--- CHECKING src/tsgnn CONTENT ---")
        stdin, stdout, stderr = ssh.exec_command("ls -F ~/ts-gnn/src/tsgnn/")
        print(stdout.read().decode())
        
        print("\n--- CONTENT OF run_pipeline.py on remote ---")
        stdin, stdout, stderr = ssh.exec_command("head -n 30 ~/ts-gnn/scripts/run_pipeline.py")
        print(stdout.read().decode())
        
        ssh.close()
    except Exception as e:
        print(f"Error connecting to HPC: {e}")

if __name__ == "__main__":
    check_hpc()
