// Host bridges only. Kernel code and original GTest launch helper are not edited.
#include "subcuber_adapter.cu"
#if defined(ORACLE_LCMA_PINGPONG)
using OriginalOuter=HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo;
#include "original_pingpong_methods.inc"
#else
using OriginalOuter=HopperF16InterleavedPresumCooperativeMaxFusion_4x256_4x256_OptNo;
#endif
#include "tests/base_test.cuh"
extern "C" int oracle_original_gtest_call(int m,int n,int k,void* a,void* b,void* c,void* d){
 // Match sync_device() completion before the original helper owns its CUDA streams.
 auto err=cudaDeviceSynchronize();if(err!=cudaSuccess)return int(err);
 std::array<cudaStream_t,49> streams{};
 for(auto& s:streams){err=cudaStreamCreate(&s);if(err!=cudaSuccess){for(auto t:streams)if(t)cudaStreamDestroy(t);return int(err);}}
 using Ref=cutlass::TensorRef<cutlass::half_t,cutlass::layout::RowMajor>;
 strassen_tests::TestCase test{cutlass::gemm::GemmCoord(m,n,k),1,1,"same_input_equivalence"};
 auto rc=strassen_tests::run_cutlass_gemm<OriginalOuter>(test,
  Ref(static_cast<cutlass::half_t*>(a),k),Ref(static_cast<cutlass::half_t*>(b),n),
  Ref(static_cast<cutlass::half_t*>(c),n),Ref(static_cast<cutlass::half_t*>(d),n),streams);
 for(auto s:streams)cudaStreamDestroy(s);
 return int(rc);
}
extern "C" int oracle_default_raster(void* context){
 auto* p=static_cast<SubContext*>(context);
 decltype(p->args.scheduler) defaults{};
 p->args.scheduler.raster_order=defaults.raster_order;
 auto status=Native::can_implement(p->args);if(status!=cutlass::Status::kSuccess)return int(status);
 size_t expected=Native::get_workspace_size(p->args);
 if(expected!=p->bytes)return -100; // No assumption that raster leaves workspace size unchanged.
 return 0;
}
