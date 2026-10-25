// Bind the original named SubCuber aliases; do not repair/modify algorithm templates.
#if defined(ORACLE_LCMA_PINGPONG)
#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cuh"
using Native=HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::StrassenGemmKernel;
#elif defined(ORACLE_LCMA_COOPERATIVE)
#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion.cuh"
using Native=HopperF16InterleavedPresumCooperativeMaxFusion_4x256_4x256_OptNo::StrassenGemmKernel;
#elif defined(ORACLE_CLASSICAL_PINGPONG)
#include "cuda/kernels/hopper/cubic/hopper_f16_cutlass_128x128_pingpong.cuh"
using Native=HopperF16Cutlass128x128Pingpong::CutlassGemm;
#else
#include "cuda/kernels/hopper/cubic/hopper_f16_cutlass_128x256_cooperative.cuh"
using Native=HopperF16Cutlass128x256Cooperative::CutlassGemm;
#endif
struct SubContext {
 Native op;Native::Arguments args;size_t bytes=0;void* workspace=nullptr;
 int swizzles[7]={2,2,1,1,1,1,1};
 SubContext(Native::Arguments a):args(a){}
 ~SubContext(){if(workspace)cudaFree(workspace);}
};
static thread_local int sub_error=0;
extern "C" void* oracle_sub_prepare(int m,int n,int k,int sms,void* a,void* b,void* c,void* d){
 using K=Native::GemmKernel;
 auto sa=cutlass::make_cute_packed_stride(typename K::StrideA{},{m,k,1});
 auto sb=cutlass::make_cute_packed_stride(typename K::StrideB{},{n,k,1});
 auto sc=cutlass::make_cute_packed_stride(typename K::StrideC{},{m,n,1});
 auto sd=cutlass::make_cute_packed_stride(typename K::StrideD{},{m,n,1});
 cutlass::KernelHardwareInfo hw{};hw.device_id=0;hw.sm_count=sms;
 typename K::ProblemShape shape{m,n,k};
 Native::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,shape,
   {static_cast<cutlass::half_t*>(a),sa,static_cast<cutlass::half_t*>(b),sb},
   {{1.f,0.f},static_cast<cutlass::half_t*>(c),sc,static_cast<cutlass::half_t*>(d),sd},hw};
 using Raster=decltype(args.scheduler.raster_order);args.scheduler.raster_order=Raster::AlongN;
 auto p=new SubContext(args);
 sub_error=int(p->op.can_implement(args));if(sub_error){delete p;return nullptr;}
 p->bytes=Native::get_workspace_size(args);
 if(p->bytes>size_t(2147483648)){sub_error=-20;delete p;return nullptr;}
 if(p->bytes && cudaMalloc(&p->workspace,p->bytes)!=cudaSuccess){sub_error=-21;delete p;return nullptr;}
 return p;
}
extern "C" int oracle_sub_error(){return sub_error;}
extern "C" size_t oracle_sub_workspace(void* p){return static_cast<SubContext*>(p)->bytes;}
extern "C" int oracle_sub_run(void* context,void* stream){
 auto* p=static_cast<SubContext*>(context);auto s=static_cast<cudaStream_t>(stream);
#if defined(ORACLE_LCMA_PINGPONG) || defined(ORACLE_LCMA_COOPERATIVE)
 auto status=p->op.initialize(p->args,p->swizzles,p->workspace,s);
 if(status!=cutlass::Status::kSuccess)return int(status);
 // Upstream run(num_streams=1) fills streams[0..6]; provide its required storage.
 cudaStream_t streams[7]={s,s,s,s,s,s,s};
 return int(p->op.run(streams,1));
#else
 auto status=p->op.initialize(p->args,p->workspace,s);
 if(status!=cutlass::Status::kSuccess)return int(status);
 return int(p->op.run(s));
#endif
}
extern "C" void oracle_sub_destroy(void* p){delete static_cast<SubContext*>(p);}
