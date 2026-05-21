import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("slogin.hpc.unibocconi.it", username="3393519", password="barracuda")

stdin, out, err = ssh.exec_command("squeue -u 3393519")
print(out.read().decode())

ssh.close()
