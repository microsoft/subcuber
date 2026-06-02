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
    return float(re.findall(r'GFLOPS (\d+)', o)[0]), float(re.findall(r'Time elapsed ([\d.]+)', o)[0])
  return -1

def find_best_split_k(binary, m,n,k):
  max_flops = 0
  max_split_k = -1
  min_time = 1<<30
  for split_k in range(1,5):
    if m > 4*1024 and split_k != 1 and "basic_gemm" in binary: continue
    if "strassen" in os.path.basename(binary):
      if "threadblock" not in os.path.basename(binary) and split_k > 1: continue
      if split_k > 1 and m > 4*1024: continue

    if k == 1024: k = 1152
    if "strassen" in os.path.basename(binary):
      flops,runtime = run_and_get_gflops(f"{binary} --m={m} --n={n} --k={k} --iterations=100 --check=0 --split_k_slices={split_k} --streams=7")
    else:
      flops,runtime = run_and_get_gflops(f"{binary}_splitk {m} {n} {k} {split_k}")
    print("Flops for last command", flops, runtime)
    if flops > max_flops:
      max_split_k = split_k
      max_flops = flops
      min_time = runtime

  print(f"Max FLOPs for {binary} {m} {n} {k}: {max_flops}, mintime {min_time} at split_k {max_split_k}")
  return max_flops, min_time

def evaluate():
  # execute("make ARCH=AMPERE -j --always-make")
  d = os.getcwd()
  os.chdir(os.path.join(d, "../00_basic_gemm"))
  basic_gemm_dir = os.getcwd()
  # execute("make ARCH=AMPERE -j --always-make")
  os.chdir(d)
  results = [("M", "N", "K", "7/8MNK", "NoISum+NoOSum", "ISum+NoOSum", "Everything", "Type-3")]
  
  try:
  # if True:
    for multiple_of_128 in range(2,129,2):#range(2, 76, 2): #range(2,20):
      m = n = k = multiple_of_128 * 128
      #m = n = multiple_of_128*128
      #k = 4096
      #m = n = 4096
      #k = multiple_of_128*128
      if m % 512 != 0: continue
      # basic_gemm_128_flops,basic_gemm_128_time = find_best_split_k(os.path.join(d, basic_gemm_dir+"/basic_gemm_128"),m,n,k)
      # basic_gemm_256 = find_best_split_k(os.path.join(d, basic_gemm_dir+"/basic_gemm_256"),m,n,k)
      # basic = max(basic_gemm_256, basic_gemm_128)
      # basic_gemm_7by8_flops,basic_gemm_7by8_time = find_best_split_k(os.path.join(d,basic_gemm_dir+"/basic_gemm_128"),7*m//2,n//2,k//2)
      # _,strassen_winograd_no_isum_no_osum = find_best_split_k(os.path.join(d,"strassen_winograd_min_lds_presum4_global_256_noinputsum_nooutputsum"),m,n,k)
      # _,strassen_winograd_isum_noosum = find_best_split_k(os.path.join(d,"strassen_winograd_min_lds_presum4_global_256_inputsum_nooutputsum"),m,n,k)
      _,strassen_winograd = find_best_split_k(os.path.join(d,"strassen_winograd_min_lds_presum4_global_256"),m,n,k)
      times = (strassen_winograd,)
      # times = (basic_gemm_128_time, basic_gemm_7by8_time, strassen_winograd_no_isum_no_osum, strassen_winograd_isum_noosum, strassen_winograd)
#               strassen_winograd_min_lds_presum0_global_256, strassen_winograd_min_lds_presum1_global_256,\
               #strassen_winograd_min_lds_presum2_global_256,
              # strassen_winograd_min_lds_presum4_global_256)
      
      results += [("FLOPS",   m,   n,  k)+times]
  except BaseException as ex:
    print(ex)

  for row in results:
    print("& ".join([str(r) for r in row]))
    
if __name__=="__main__":
  evaluate()
