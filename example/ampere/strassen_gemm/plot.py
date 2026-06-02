import sys
import csv
from common import *

csv_file = sys.argv[1]
pdf_file = sys.argv[2]

filtered_data = []

with open(csv_file, 'r') as f:
    csv_reader = csv.reader(f,delimiter='&')
    for i, row in enumerate(csv_reader):
        row_new = []
        for e in row:
            if e != "":
                row_new.append(e.strip())
        row = row_new
        if row == [] or row == None:
            continue
        filtered_data += [row]

print(filtered_data)

def parse_p_n(s):
    return s[0]

import matplotlib.pyplot as plt
import numpy as np

ind = []
width = 0.1
m = 0
n = 1
k = 2
strassen_tb = 3
strassen_global_128 = 4
strassen_global_256 = 5
strassen_global_128_parallel_subgemm = 6
strassen_global_256_parallel_subgemm = 7
baseline_128 = 8
baseline_256 = 9

# fig = plt.subplots(figsize =(10, 7))
strassen_tb_flops = []
strassen_global_128_flops = []
strassen_global_256_flops = []
strassen_global_128_parallel_subgemm_flops = []
strassen_global_256_parallel_subgemm_flops = []
baseline_128_flops = []
baseline_256_flops = []
x_ticks = []
i = 0
for row in filtered_data:
    m = int(row[0])
    if m< 768: continue
    if m % 256 != 0: continue
    # if m > 2048 and m % 512 != 0:
    #   continue
    # if m > 4096 and m % 1024 != 0:
    #   continue
    if m > 6*1024 and m % 512 != 0:
      continue
    if m > 10*1024 and m % 512 != 0:
      continue
    x_ticks += [m]
    ind += [i]
    strassen_tb_flops         += [float(row[strassen_tb])/1e3]
    strassen_global_128_flops += [float(row[strassen_global_128])/1e3]
    strassen_global_256_flops += [float(row[strassen_global_256])/1e3]
    strassen_global_128_parallel_subgemm_flops += [float(row[strassen_global_128_parallel_subgemm])/1e3]
    strassen_global_256_parallel_subgemm_flops += [float(row[strassen_global_256_parallel_subgemm])/1e3]
    baseline_128_flops        += [float(row[baseline_128])/1e3]
    baseline_256_flops        += [float(row[baseline_256])/1e3]
    i += 1
  
max_strassen_flops = np.maximum(strassen_tb_flops, 
                                np.maximum(strassen_global_128_flops, 
                                           np.maximum(strassen_global_256_flops,
                                           np.maximum(strassen_global_128_parallel_subgemm_flops,
                                                      strassen_global_256_parallel_subgemm_flops))))
max_baseline_flops = np.maximum(baseline_128_flops, baseline_256_flops)

fig, ax2 = plt.subplots(1,1,sharex=True)
ind = np.array(ind)
p1 = ax2.plot(ind, max_strassen_flops, color=colors[0], marker='o')
p2 = ax2.plot(ind, max_baseline_flops, color=colors[1], marker='d')

# for i, f in enumerate(pytorchflops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for i, f in enumerate(cogentflops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for i, f in enumerate(fastkronwosharedflops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

for i, f in enumerate(zip(max_strassen_flops, (max_strassen_flops/max_baseline_flops))):
    ax2.text(i, f[0], "%.1f"%round((f[1] - 1)*100, 1), color = 'black', fontsize='large', ha='center')

# for bar1, d in zip(p3, fastkrontimes):
#     ax2.text(bar1.get_x()+bar1.get_width()/2, (bar1.get_height())/2, "%.2f ms"%d, color = 'black', ha = 'center', va = 'center', rotation=90, fontsize='large')

# for bar1, speedup in zip(p3, fastkronspeedup):
#     ax2.text(bar1.get_x()+bar1.get_width()/2+0.04, bar1.get_height()+0.05, r"%.2f$\times$"%(1/speedup), color = 'black', ha = 'center', va = 'center', rotation=0, fontsize='large')

plt.ylabel('TFLOPS')

plt.xlabel('M=N=K', fontsize='large')
# plt.title('Contribution by the teams')
# plt.yticks([0,2,4,6,8,10,12,14,16,18])
plt.xticks(ind, x_ticks,rotation=90)
plt.legend((p1[0], p2[0]), ('Strassen', 'Baseline'),
            loc='upper left', fontsize='large', bbox_to_anchor=(0.1, 1.1),
            ncol=2,columnspacing=1,handlelength=1.7)

FIGURES_DIR = "./"
    
plt.rcParams["font.family"] = "libertine"
#FIGURES_DIR = "./"
fig = plt.gcf()
fig.subplots_adjust(bottom=0.1)
fig.set_size_inches(30, 10)

# ax.set_xticks([])
fig.savefig(FIGURES_DIR+pdf_file.replace('pdf','png'),bbox_inches='tight',pad_inches=0)
plt.show()