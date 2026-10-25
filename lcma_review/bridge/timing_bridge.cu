#include "lcma_original_entry_bridge.cu"
// Host-only timing ABI. Kernel code, math, scheduler arguments and defaults unchanged.
extern "C" int oracle_timing_init(void* context,void* stream){
 auto* p=static_cast<SubContext*>(context);
 return int(p->op.initialize(p->args,p->swizzles,p->workspace,static_cast<cudaStream_t>(stream)));
}
extern "C" int oracle_timing_batch(void* context,void* stream,int count){
 auto* p=static_cast<SubContext*>(context); auto s=static_cast<cudaStream_t>(stream);
 cudaStream_t streams[7]={s,s,s,s,s,s,s};
 for(int i=0;i<count;++i){auto status=p->op.run(streams,1);if(status!=cutlass::Status::kSuccess)return int(status);}
 return 0;
}
