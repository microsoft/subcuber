import subprocess
import os
import re

def execute(command):
  # print(f"Executing {command}")
  (s, o) = subprocess.getstatusoutput(command)
  return s,o

def run_and_get_gflops(command):
  s,o = execute(command)
  if s == 0:
    return float(re.findall(r'GFLOPS (\d+)', o)[0])
  return -1

def find_best_split_k(binary, m,n,k):
  max_flops = 0
  max_split_k = -1
  for split_k in range(1,5):
    if m > 4*1024 and split_k != 1 and "basic_gemm" in binary: continue
    if "strassen" in os.path.basename(binary):
      if "threadblock" not in os.path.basename(binary) and split_k > 1: continue
      if split_k > 1 and m > 4*1024: continue

    if k == 1024: k = 1152
    if "strassen" in os.path.basename(binary):
      flops = run_and_get_gflops(f"{binary} --m={m} --n={n} --k={k} --iterations=100 --check=0 --split_k_slices={split_k} --streams=7")
    else:
      flops = run_and_get_gflops(f"{binary}_splitk {m} {n} {k} {split_k}")
    print("Flops for last command", flops)
    if flops > max_flops:
      max_split_k = split_k
      max_flops = flops

  print(f"Max FLOPs for {binary} {m} {n} {k}: {max_flops} at split_k {max_split_k}")
  return max_flops

def evaluate():
  execute("make ARCH=AMPERE -j --always-make")
  d = os.getcwd()
  os.chdir(os.path.join(d, "../00_basic_gemm"))
  basic_gemm_dir = os.getcwd()
  execute("make ARCH=AMPERE -j --always-make")
  os.chdir(d)
  results = [("M", "N", "K", "Gemm128", "Gemm256", "StrassenTB", "StrassenGlobal256", "WinogradPresum4")]
  
  try:
    for multiple_of_128 in range(2,129,2):#range(2, 76, 2): #range(2,20):
      m = n = k = multiple_of_128 * 128
      #m = n = multiple_of_128*128
      #k = 4096
      #m = n = 4096
      #k = multiple_of_128*128
      if m % 256 != 0: continue
      basic_gemm_128 = find_best_split_k(os.path.join(d, basic_gemm_dir+"/basic_gemm_128"),m,n,k)
      basic_gemm_256 = find_best_split_k(os.path.join(d, basic_gemm_dir+"/basic_gemm_256"),m,n,k)
      basic = max(basic_gemm_128, basic_gemm_256) 
      strassen_threadblock = find_best_split_k(os.path.join(d,"strassen_winograd_threadblock_splitk"),m,n,k)
      strassen_global_256 = find_best_split_k(os.path.join(d,"strassen_global_256"),m,n,k)
      #strassen_winograd_min_lds_presum0_global_256 = find_best_split_k(os.path.join(d,"strassen_winograd_min_lds_presum0_global_256"),m,n,k)
      #strassen_winograd_min_lds_presum1_global_256 = find_best_split_k(os.path.join(d,"strassen_winograd_min_lds_presum1_global_256"),m,n,k)
      #strassen_winograd_min_lds_presum2_global_256 = find_best_split_k(os.path.join(d,"strassen_winograd_min_lds_presum2_global_256"),m,n,k)
      strassen_winograd_min_lds_presum4_global_256 = find_best_split_k(os.path.join(d,"strassen_winograd_min_lds_presum4_global_256"),m,n,k)
      flops = (basic_gemm_128, basic_gemm_256, strassen_threadblock, strassen_global_256, strassen_winograd_min_lds_presum4_global_256)
#               strassen_winograd_min_lds_presum0_global_256, strassen_winograd_min_lds_presum1_global_256,\
               #strassen_winograd_min_lds_presum2_global_256,
              # strassen_winograd_min_lds_presum4_global_256)
      
      results += [("FLOPS",   m,   n,  k)+flops]
      results += [("Speedup", "", "", "", ) + tuple("%.3f"%(fl/basic) for fl in flops)]
  except BaseException as ex:
    print(ex)

  for row in results:
    print("& ".join([str(r) for r in row]))
    
if __name__=="__main__":
  evaluate()
