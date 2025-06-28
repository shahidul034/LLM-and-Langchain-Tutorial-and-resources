# Md Shahidul Salim
# Phd Candidate, Umass Lowell
import os
def gpu_info():
    dat=[]
    import subprocess
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    out2 = out.split("\n")[8:]
    splitter = out2[3]
    out3 = "\n".join(out2).split(splitter)[:-1]

    # Check GPU usage
    for idx,x in enumerate(out3):
        gpu = ((x.split("|"))[1].replace("Off", ""))
        t = ((x.split("|"))[6].replace("Off", "").replace("MiB", "").split("/"))
        ans = (int(t[0]) / int(t[1])) * 100
            # print(f"{gpu[:-5]}--->   GPU is idle ({ans:.2f}%)")
        dat.append({
                "gpu_id": idx,
                "gpu": ((gpu[:-5])[6:]).strip(),
                "usage": f"{ans:.2f}"
            })
    return dat
gpu_info_ans = gpu_info()
flag=0
for gpu in gpu_info_ans:
    if float(gpu['usage'])<10:
        empty_gpu = gpu['gpu_id']
        flag=1
        break
if flag==0:
    print("No empty GPU found, please check your GPU usage.")
    exit(1)
else:
    print(f"Using GPU {empty_gpu} for processing.")

os.environ["CUDA_VISIBLE_DEVICES"] = str(empty_gpu)
