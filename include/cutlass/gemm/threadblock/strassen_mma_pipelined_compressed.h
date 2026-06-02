/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  typename StrassenShape_,
  /// Iterates over tiles of A operand in global memory 
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of A operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of B operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_,
  typename StrassenMiGroup_,
  /// Transformation applied to A operand
  typename TransformA_,
  ///
  /// Transformation applied to B operand
  typename TransformB_,
  /// Used for partial specialization
  typename Enable
>
class StrassenMmaPipeline<Shape_, StrassenShape_,IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, ElementC_, LayoutC_, Policy_, MmaStrassen::Type::Compressed, StrassenMiGroup_, TransformA_, TransformB_, Enable> : 
  public StrassenMmaBase<Shape_, StrassenShape_, Policy_, 2, MmaStrassen::Type::Compressed, StrassenMiGroup_> {
public:
  static const auto StrassenKind = MmaStrassen::Type::Compressed;
  ///< Base class
  using Base = StrassenMmaBase<Shape_, StrassenShape_, Policy_, 2, StrassenKind, StrassenMiGroup_>;

  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using StrassenShape = StrassenShape_;
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // staticaly assert kStages for StrassenMmaPipeline is two (Double-buffered pipeline)
  static_assert((Base::kStages==2), "StrassenMmaPipeline requires kStages set to value 2");

protected:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A0_;
  SmemIteratorA smem_iterator_A1_;
  SmemIteratorA smem_iterator_A2_;
  SmemIteratorA smem_iterator_A3_;

  SmemIteratorA smem_iterator_A03_;
  SmemIteratorA smem_iterator_A23_;
  SmemIteratorA smem_iterator_A01_;
  SmemIteratorA smem_iterator_A20_;
  SmemIteratorA smem_iterator_A13_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B0_;
  SmemIteratorB smem_iterator_B3_;
  SmemIteratorB smem_iterator_B03_;
  SmemIteratorB smem_iterator_B13_;
  SmemIteratorB smem_iterator_B20_;
  SmemIteratorB smem_iterator_B01_;
  SmemIteratorB smem_iterator_B23_;

  ///< transformation applied to A fragment
  TransformA transform_A_;

  ///< transformation applied to B fragment
  TransformB transform_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx;
  int smem_write_stage_idx_M2M4;

public:
  typename Base::SharedStorage& shared_storage;
  /// Construct from tensor references
  CUTLASS_DEVICE
  StrassenMmaPipeline(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx,                                       ///< ID of each thread within a warp
    TransformA transform_A = TransformA(),              ///< transformation applied to A fragment
    TransformB transform_B = TransformB()               ///< transformation applied to B fragment
  ):
    shared_storage(shared_storage),
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A0_(shared_storage.operand_A0_ref(), thread_idx),
    // smem_iterator_A1_(shared_storage.operand_A1_ref(), thread_idx),
    // smem_iterator_A2_(shared_storage.operand_A2_ref(), thread_idx),
    smem_iterator_A3_(shared_storage.operand_A3_ref(), thread_idx),
    smem_iterator_A03_(shared_storage.operand_A03_ref(), thread_idx),
    smem_iterator_A23_(shared_storage.operand_A23_ref(), thread_idx),
    smem_iterator_A01_(shared_storage.operand_A01_ref(), thread_idx),
    smem_iterator_A20_(shared_storage.operand_A20_ref(), thread_idx),
    smem_iterator_A13_(shared_storage.operand_A13_ref(), thread_idx),

    smem_iterator_B0_(shared_storage.operand_B0_ref(), thread_idx),
    smem_iterator_B3_(shared_storage.operand_B3_ref(), thread_idx),
    smem_iterator_B03_(shared_storage.operand_B03_ref(), thread_idx),
    smem_iterator_B13_(shared_storage.operand_B13_ref(), thread_idx),
    smem_iterator_B20_(shared_storage.operand_B20_ref(), thread_idx),
    smem_iterator_B01_(shared_storage.operand_B01_ref(), thread_idx),
    smem_iterator_B23_(shared_storage.operand_B23_ref(), thread_idx),
  
    transform_A_(transform_A),
    transform_B_(transform_B),
    smem_write_stage_idx(0),
    smem_write_stage_idx_M2M4(0)
  {

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;
    // if (threadIdx.x == 0 && blockIdx.x * blockIdx.y == 0)
    //   printf("Base::WarpCount::kM %d\n", Base::WarpCount::kM);
    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A0_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    // this->warp_tile_iterator_A1_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    // this->warp_tile_iterator_A2_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_A3_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_A03_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_A23_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_A01_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_A20_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_A13_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});

    this->warp_tile_iterator_B0_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B3_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B03_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B13_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B20_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B01_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B23_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }


  /// Advance shared memory write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_write_stageM2M4() {
    ++this->smem_iterator_A0_;
    ++this->smem_iterator_A01_;
    ++this->smem_iterator_B3_;
    ++this->smem_iterator_B13_;


    if (smem_write_stage_idx_M2M4 == 1) {
      const int NumStages = 2;
      this->smem_iterator_A0_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A01_.add_tile_offset({0, -NumStages});
      this->smem_iterator_B3_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B13_.add_tile_offset({-NumStages, 0});
    }

    smem_write_stage_idx_M2M4 ^= 1;
  }

  CUTLASS_DEVICE
  void advance_smem_stagesM2M4() {
    ++this->smem_iterator_A0_;
    ++this->smem_iterator_A01_;
    ++this->smem_iterator_B3_;
    ++this->smem_iterator_B13_;

    if (smem_write_stage_idx_M2M4 == 1) {
      const int NumStages = 2;
      this->smem_iterator_A0_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A01_.add_tile_offset({0, -NumStages});
      this->smem_iterator_B3_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B13_.add_tile_offset({-NumStages, 0});
    } else {
      const int NumStages = 2;
      this->warp_tile_iterator_A0_.add_tile_offset(
        {0, - NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_A01_.add_tile_offset(
        {0, - NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});

      this->warp_tile_iterator_B3_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B13_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    }

    smem_write_stage_idx_M2M4 ^= 1;
  }

  CUTLASS_DEVICE
  void advance_smem_write_stage()
  {
    ++this->smem_iterator_A0_;
    ++this->smem_iterator_A3_;
    ++this->smem_iterator_A03_;
    ++this->smem_iterator_A23_;
    ++this->smem_iterator_A01_;
    ++this->smem_iterator_A20_;
    ++this->smem_iterator_A13_;

    ++this->smem_iterator_B0_;
    ++this->smem_iterator_B3_;
    ++this->smem_iterator_B03_;
    ++this->smem_iterator_B13_;
    ++this->smem_iterator_B20_;
    ++this->smem_iterator_B01_;
    ++this->smem_iterator_B23_;
    const int NumStages = 2;//Base::kStages;
      
    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      this->smem_iterator_A0_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A3_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A03_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A23_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A01_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A20_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A13_.add_tile_offset({0, -NumStages});

      this->smem_iterator_B0_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B3_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B03_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B13_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B20_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B01_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B23_.add_tile_offset({-NumStages, 0});
    }

    smem_write_stage_idx ^= 1;
  }

  /// Advance shared memory read- and write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_stages()
  {
    ++this->smem_iterator_A0_;
    // ++this->smem_iterator_A1_;
    // ++this->smem_iterator_A2_;
    ++this->smem_iterator_A3_;
    ++this->smem_iterator_A03_;
    ++this->smem_iterator_A23_;
    ++this->smem_iterator_A01_;
    ++this->smem_iterator_A20_;
    ++this->smem_iterator_A13_;

    ++this->smem_iterator_B0_;
    ++this->smem_iterator_B3_;
    ++this->smem_iterator_B03_;
    ++this->smem_iterator_B13_;
    ++this->smem_iterator_B20_;
    ++this->smem_iterator_B01_;
    ++this->smem_iterator_B23_;

    const int NumStages = 2;//Base::kStages;
      
    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      // wrap write stage
      this->smem_iterator_A0_.add_tile_offset({0, -NumStages});
      // this->smem_iterator_A1_.add_tile_offset({0, -Base::kStages});
      // this->smem_iterator_A2_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_A3_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A03_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A23_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A01_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A20_.add_tile_offset({0, -NumStages});
      this->smem_iterator_A13_.add_tile_offset({0, -NumStages});

      this->smem_iterator_B0_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B3_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B03_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B13_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B20_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B01_.add_tile_offset({-NumStages, 0});
      this->smem_iterator_B23_.add_tile_offset({-NumStages, 0});
    }
    else
    {
      // wrap read stage
      this->warp_tile_iterator_A0_.add_tile_offset(
        {0, - NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      // this->warp_tile_iterator_A1_.add_tile_offset(
      //   {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      // this->warp_tile_iterator_A2_.add_tile_offset(
      //   {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_A3_.add_tile_offset(
        {0, -NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});

      this->warp_tile_iterator_A03_.add_tile_offset(
        {0, -NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_A23_.add_tile_offset(
        {0, -NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_A01_.add_tile_offset(
        {0, -NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_A20_.add_tile_offset(
        {0, -NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_A13_.add_tile_offset(
        {0, -NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations});

      this->warp_tile_iterator_B0_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B3_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B03_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B13_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B20_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B01_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B23_.add_tile_offset(
        {-NumStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    }

    smem_write_stage_idx ^= 1;
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void sum(Fragment& frag_op1, Fragment& frag_op2, Fragment& frag_out) {
    #pragma unroll
    for (int i = 0; i < frag_op1.size(); i++) {
      frag_out[i] = frag_op1[i] + frag_op2[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void subtract(Fragment& frag_op1, Fragment& frag_op2, Fragment& frag_out) {
    #pragma unroll
    for (int i = 0; i < frag_op1.size(); i++) {
      frag_out[i] = frag_op1[i] - frag_op2[i];
    }
  }

  template<typename Fragment, typename Element>
  CUTLASS_DEVICE
  void storeToMSh(Fragment& m, Element* data) {
    for (int i = 0; i < m.size(); i++) {
      data[i * blockDim.x + threadIdx.x] = m[i];
    }
  }

  template<typename Fragment, typename Element>
  CUTLASS_DEVICE
  void loadFromMSh(Fragment& m, Element* data) {
    for (int i = 0; i < m.size(); i++) {
      m[i] = data[i * blockDim.x + threadIdx.x];
    }
  }


  template<typename Element>
  CUTLASS_DEVICE
  void zeroMsh(Element* data, size_t sz) {
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
      data[i] = 0;
    }
  }

  CUTLASS_DEVICE
  void strassen_A(FragmentA& tb_frag_A0, FragmentA& tb_frag_A1, FragmentA& tb_frag_A2, FragmentA& tb_frag_A3,
                  FragmentA& tb_frag_A03, FragmentA& tb_frag_A23, FragmentA& tb_frag_A01, FragmentA& tb_frag_A20, FragmentA& tb_frag_A13) {
    sum(tb_frag_A0, tb_frag_A3, tb_frag_A03);
    sum(tb_frag_A2, tb_frag_A3, tb_frag_A23);
    // if (threadIdx.x == 0) {
    //   printf("401: %f %f %f\n", tb_frag_A2[0], tb_frag_A3[0], tb_frag_A23[0]);
    // }
    //sum(tb_frag_A0, 0, tb_frag_A0)
    //sum(tb_frag_A3, 0, tb_frag_A3)
    sum(tb_frag_A0, tb_frag_A1, tb_frag_A01);
    subtract(tb_frag_A2, tb_frag_A0, tb_frag_A20);
    subtract(tb_frag_A1, tb_frag_A3, tb_frag_A13);
  }

  CUTLASS_DEVICE
  void strassen_B(FragmentB& tb_frag_B0, FragmentB& tb_frag_B1, FragmentB& tb_frag_B2, FragmentB& tb_frag_B3,
                  FragmentB& tb_frag_B03, FragmentB& tb_frag_B13, FragmentB& tb_frag_B20, FragmentB& tb_frag_B01, FragmentB& tb_frag_B23) {
    sum(tb_frag_B0, tb_frag_B3, tb_frag_B03);
    //sum(tb_frag_B0, 0, tb_frag_B0)
    //sum(tb_frag_B3, 0, tb_frag_B3)
    subtract(tb_frag_B1, tb_frag_B3, tb_frag_B13);
    subtract(tb_frag_B2, tb_frag_B0, tb_frag_B20);
    sum(tb_frag_B0, tb_frag_B1, tb_frag_B01);
    sum(tb_frag_B2, tb_frag_B3, tb_frag_B23);
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A0,      ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A1,
    IteratorA &iterator_A2,      ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A3,
    IteratorB &iterator_B0,      ///< [in|out] iterator over B operand in global memory
    IteratorB &iterator_B1,
    IteratorB &iterator_B2,
    IteratorB &iterator_B3,
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // The last kblock is loaded in the prolog

    // Load A fragment from global A
    FragmentA tb_frag_A0;
    tb_frag_A0.clear();
    iterator_A0.load(tb_frag_A0);
    // if (threadIdx.x == 0) {
    //   printf("258: %f\n", tb_frag_A[0]);
    // if (false) {
      // for (int i = 0; i < tb_frag_A0.size(); i++) {
      //   printf("256: %d: %d %f\n", threadIdx.x, i, tb_frag_A0[i]);
      // }
    // }
    // iterator_A.store(tb_frag_A);
    ++iterator_A0;
    
    FragmentA tb_frag_A1;
    tb_frag_A1.clear();
    iterator_A1.load(tb_frag_A1);
    ++iterator_A1;

    FragmentA tb_frag_A2;
    tb_frag_A2.clear();
    iterator_A2.load(tb_frag_A2);
    ++iterator_A2;

    FragmentA tb_frag_A3;
    tb_frag_A3.clear();
    iterator_A3.load(tb_frag_A3);
    ++iterator_A3;
    // if (threadIdx.x == 0)
    // for (int i = 0; i < tb_frag_A2.size(); i++) {
    //   printf("256: %f + %f\n", tb_frag_A2[i], tb_frag_A3[i]);
    // }

    FragmentA tb_frag_A03, tb_frag_A23, tb_frag_A01, tb_frag_A20, tb_frag_A13;

    strassen_A(tb_frag_A0, tb_frag_A1, tb_frag_A2, tb_frag_A3,
               tb_frag_A03, tb_frag_A23, tb_frag_A01, tb_frag_A20, tb_frag_A13);

    // Load B fragment from global B
    FragmentB tb_frag_B0;
    tb_frag_B0.clear();
    iterator_B0.load(tb_frag_B0);
    // for (int i = 0; i < tb_frag_B0.size(); i++) {
    //   printf("256: %d: %d %f\n", threadIdx.x, i, tb_frag_B0[i]);
    // }
    ++iterator_B0;

    FragmentB tb_frag_B1;
    tb_frag_B1.clear();
    iterator_B1.load(tb_frag_B1);
    // for (int i = 0; i < tb_frag_B1.size(); i++) {
    //   printf("256: %d: %d %f\n", threadIdx.x, i, tb_frag_B1[i]);
    // }
    ++iterator_B1;

    FragmentB tb_frag_B2;
    tb_frag_B2.clear();
    iterator_B2.load(tb_frag_B2);
    // for (int i = 0; i < tb_frag_B1.size(); i++) {
    //   printf("256: %d: %d %f\n", threadIdx.x, i, tb_frag_B1[i]);
    // }
    ++iterator_B2;

    FragmentB tb_frag_B3;
    tb_frag_B3.clear();
    iterator_B3.load(tb_frag_B3);
    ++iterator_B3;
    
    FragmentB tb_frag_B03, tb_frag_B13, tb_frag_B20, tb_frag_B01, tb_frag_B23;

    strassen_B(tb_frag_B0, tb_frag_B1, tb_frag_B2, tb_frag_B3,
               tb_frag_B03, tb_frag_B13, tb_frag_B20, tb_frag_B01, tb_frag_B23);
    // if (threadIdx.x == 0) {
    //   printf("256: %f %f %f\n", tb_frag_B2[0], tb_frag_B3[0], tb_frag_B23[0]);
    // }
    // // Store A and B fragments to shared
    this->smem_iterator_A0_.store(transform_A_(tb_frag_A0));
    // this->smem_iterator_A1_.store(transform_A_(tb_frag_A1));
    // this->smem_iterator_A2_.store(transform_A_(tb_frag_A2));
    this->smem_iterator_A3_.store(transform_A_(tb_frag_A3));

    this->smem_iterator_A03_.store(transform_A_(tb_frag_A03));
    this->smem_iterator_A23_.store(transform_A_(tb_frag_A23));
    this->smem_iterator_A01_.store(transform_A_(tb_frag_A01));
    this->smem_iterator_A20_.store(transform_A_(tb_frag_A20));
    this->smem_iterator_A13_.store(transform_A_(tb_frag_A13));

    this->smem_iterator_B0_.store(transform_B_(tb_frag_B0));
    this->smem_iterator_B3_.store(transform_B_(tb_frag_B3));
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //   for (int i = 0; i < tb_frag_B3.size(); i++) {
    //     printf("523: %d %f\n", threadIdx.x, tb_frag_B3[i]);
    // }
    this->smem_iterator_B03_.store(transform_B_(tb_frag_B03));
    this->smem_iterator_B13_.store(transform_B_(tb_frag_B13));
    this->smem_iterator_B20_.store(transform_B_(tb_frag_B20));
    this->smem_iterator_B01_.store(transform_B_(tb_frag_B01));
    this->smem_iterator_B23_.store(transform_B_(tb_frag_B23));

    // // Advance write stage
    advance_smem_write_stage();
  }

  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    __syncthreads();
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
    // int part,
    int gemm_k_iterations,        ///< number of threadblock mainloop iterations
    FragmentC &m0, FragmentC &m1, 
    FragmentC &m2, FragmentC &m3, 
    FragmentC &m4, FragmentC &m5, 
    FragmentC &m6,
    IteratorA &iterator_A0,        ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A1,        ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A2,        ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A3,        ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B0,
    IteratorB &iterator_B1,
    IteratorB &iterator_B2,
    IteratorB &iterator_B3
    )        ///< [in|out] iterator over B operand in global memory
  {

    // The last kblock is loaded in the prolog

    // Load A fragment from global A
    FragmentA tb_frag_A0;
    FragmentA tb_frag_A1;
    FragmentA tb_frag_A2;
    FragmentA tb_frag_A3;
    FragmentA tb_frag_A03, tb_frag_A23, tb_frag_A01, tb_frag_A20, tb_frag_A13;

    // Load B fragment from global B
    FragmentB tb_frag_B0;
    FragmentB tb_frag_B1;
    FragmentB tb_frag_B2;
    FragmentB tb_frag_B3;
    FragmentB tb_frag_B03, tb_frag_B13, tb_frag_B20, tb_frag_B01, tb_frag_B23;

    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    // Pair of fragments used to overlap shared memory loads and math instructions
    const int NumStages = 1;
    WarpFragmentA warp_frag_A0[NumStages], warp_frag_A3[NumStages];
    WarpFragmentA warp_frag_A03[NumStages], warp_frag_A23[NumStages], warp_frag_A01[NumStages], warp_frag_A20[NumStages], warp_frag_A13[NumStages];

    WarpFragmentB warp_frag_B0[NumStages], warp_frag_B3[NumStages];
    WarpFragmentB warp_frag_B03[NumStages], warp_frag_B13[NumStages], warp_frag_B20[NumStages], warp_frag_B01[NumStages], warp_frag_B23[NumStages];

    iterator_A0.load(tb_frag_A0);
    ++iterator_A0;
    iterator_A1.load(tb_frag_A1);
    ++iterator_A1;
    
    iterator_B1.load(tb_frag_B1);
    ++iterator_B1;
    iterator_B3.load(tb_frag_B3);
    ++iterator_B3;

    sum(tb_frag_A0, tb_frag_A1, tb_frag_A01);
    subtract(tb_frag_B1, tb_frag_B3, tb_frag_B13);

    this->smem_iterator_A0_.store(transform_A_(tb_frag_A0));
    this->smem_iterator_A01_.store(transform_A_(tb_frag_A01));

    this->smem_iterator_B3_.store(transform_B_(tb_frag_B3));
    this->smem_iterator_B13_.store(transform_B_(tb_frag_B13));
    advance_smem_write_stageM2M4();

    gmem_wait();

    // this->warp_tile_iterator_A0_.set_kgroup_index(0);
    // this->warp_tile_iterator_A01_.set_kgroup_index(0);
    
    // this->warp_tile_iterator_B3_.set_kgroup_index(0);
    // this->warp_tile_iterator_B13_.set_kgroup_index(0);

    // this->warp_tile_iterator_A0_.load(warp_frag_A0[0]);
    // this->warp_tile_iterator_A01_.load(warp_frag_A01[0]);

    // this->warp_tile_iterator_B3_.load(warp_frag_B3[0]);
    // this->warp_tile_iterator_B13_.load(warp_frag_B13[0]);
    
    // ++this->warp_tile_iterator_A0_;
    // ++this->warp_tile_iterator_A01_;
    
    // ++this->warp_tile_iterator_B3_;
    // ++this->warp_tile_iterator_B13_;

    // Avoid reading out of bounds if this was the last loop iteration
    iterator_A0.clear_mask(gemm_k_iterations <= 1);
    iterator_A1.clear_mask(gemm_k_iterations <= 1);
    iterator_A2.clear_mask(gemm_k_iterations <= 1);
    iterator_A3.clear_mask(gemm_k_iterations <= 1);

    iterator_B0.clear_mask(gemm_k_iterations <= 1);
    iterator_B1.clear_mask(gemm_k_iterations <= 1);
    iterator_B2.clear_mask(gemm_k_iterations <= 1);
    iterator_B3.clear_mask(gemm_k_iterations <= 1);

    //
    // Mainloop
    //
    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
      // as the case may be.
      // Load fragment from global A

      tb_frag_A0.clear(); tb_frag_A1.clear();
      tb_frag_B1.clear(); tb_frag_B3.clear();

      iterator_A0.load(tb_frag_A0); iterator_A1.load(tb_frag_A1);
      iterator_B1.load(tb_frag_B1); iterator_B3.load(tb_frag_B3);

      ++iterator_A0; ++iterator_A1;
      ++iterator_B1; ++iterator_B3;

      iterator_A0.clear_mask(gemm_k_iterations <= 2);
      iterator_A1.clear_mask(gemm_k_iterations <= 2);

      iterator_B1.clear_mask(gemm_k_iterations <= 2);
      iterator_B3.clear_mask(gemm_k_iterations <= 2);

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          

        }
        this->warp_tile_iterator_A0_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_A01_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_B3_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B13_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A0_.load(warp_frag_A0[(warp_mma_k) % NumStages]);
        this->warp_tile_iterator_A01_.load(warp_frag_A01[(warp_mma_k) % NumStages]);

        this->warp_tile_iterator_B3_.load(warp_frag_B3[(warp_mma_k) % NumStages]);
        this->warp_tile_iterator_B13_.load(warp_frag_B13[(warp_mma_k) % NumStages]);
        
        ++this->warp_tile_iterator_A0_;
        ++this->warp_tile_iterator_A01_;
        
        ++this->warp_tile_iterator_B3_;
        ++this->warp_tile_iterator_B13_;

        warp_mma(m2,
          warp_frag_A0[(warp_mma_k) % NumStages],
          warp_frag_B13[(warp_mma_k) % NumStages],
          m2);
        warp_mma(m4,
          warp_frag_A01[(warp_mma_k) % NumStages],
          warp_frag_B3[(warp_mma_k) % NumStages],
          m4);
      }

      sum(tb_frag_A0, tb_frag_A1, tb_frag_A01);
      subtract(tb_frag_B1, tb_frag_B3, tb_frag_B13);

      this->smem_iterator_A0_.store(transform_A_(tb_frag_A0));
      this->smem_iterator_A01_.store(transform_A_(tb_frag_A01));

      this->smem_iterator_B3_.store(transform_B_(tb_frag_B3));
      this->smem_iterator_B13_.store(transform_B_(tb_frag_B13));

      gmem_wait();
      // Advance smem read and write stages
      advance_smem_stagesM2M4();
      /*
      iterator_A3.load(tb_frag_A3);
      ++iterator_A3;

      iterator_B2.load(tb_frag_B2);
      ++iterator_B2;

      iterator_B0.load(tb_frag_B0);
      ++iterator_B0;

      subtract(tb_frag_B2, tb_frag_B0, tb_frag_B20);
      subtract(tb_frag_A1, tb_frag_A3, tb_frag_A13);
      sum(tb_frag_B2, tb_frag_B3, tb_frag_B23);

      gmem_wait();

      this->smem_iterator_B20_.store(transform_B_(tb_frag_B20));
      this->smem_iterator_A3_.store(transform_A_(tb_frag_A3));
      this->smem_iterator_A13_.store(transform_A_(tb_frag_A13));
      this->smem_iterator_B23_.store(transform_B_(tb_frag_B23));

      gmem_wait();

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
        this->warp_tile_iterator_A13_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B23_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A13_.load(warp_frag_A13[(warp_mma_k) % NumStages]);
        this->warp_tile_iterator_B23_.load(warp_frag_B23[(warp_mma_k) % NumStages]);

        ++this->warp_tile_iterator_A13_;
        ++this->warp_tile_iterator_B23_;

        warp_mma(m6,
          warp_frag_A13[warp_mma_k % NumStages],
          warp_frag_B23[warp_mma_k % NumStages],
          m6);

        this->warp_tile_iterator_A3_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B20_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A3_.load(warp_frag_A3[(warp_mma_k) % NumStages]);
        this->warp_tile_iterator_B20_.load(warp_frag_B20[(warp_mma_k) % NumStages]);

        ++this->warp_tile_iterator_A3_;
        ++this->warp_tile_iterator_B20_;

        warp_mma(m3,
          warp_frag_A3[warp_mma_k % NumStages],
          warp_frag_B20[warp_mma_k % NumStages],
          m3);
      }

      iterator_A2.load(tb_frag_A2);
      ++iterator_A2;

      subtract(tb_frag_A2, tb_frag_A0, tb_frag_A20);
      sum(tb_frag_B0, tb_frag_B1, tb_frag_B01);
      sum(tb_frag_A2, tb_frag_A3, tb_frag_A23);

      sum(tb_frag_A0, tb_frag_A3, tb_frag_A03);
      sum(tb_frag_B0, tb_frag_B3, tb_frag_B03);

      gmem_wait();

      this->smem_iterator_B0_.store(transform_B_(tb_frag_B0));
      this->smem_iterator_A03_.store(transform_A_(tb_frag_A03));
      this->smem_iterator_B03_.store(transform_B_(tb_frag_B03));
      this->smem_iterator_A23_.store(transform_A_(tb_frag_A23));
      this->smem_iterator_A20_.store(transform_A_(tb_frag_A20));
      this->smem_iterator_B01_.store(transform_B_(tb_frag_B01));
      gmem_wait();
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
        this->warp_tile_iterator_A03_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B03_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A03_.load(warp_frag_A03[0]);
        this->warp_tile_iterator_B03_.load(warp_frag_B03[(warp_mma_k) % NumStages]);

        ++this->warp_tile_iterator_A03_;
        ++this->warp_tile_iterator_B03_;

        warp_mma(m0,
          warp_frag_A03[warp_mma_k % NumStages],
          warp_frag_B03[warp_mma_k % NumStages],
          m0);
     
        this->warp_tile_iterator_A20_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B01_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A20_.load(warp_frag_A20[(warp_mma_k) % NumStages]);
        this->warp_tile_iterator_B01_.load(warp_frag_B01[(warp_mma_k) % NumStages]);

        ++this->warp_tile_iterator_A20_;
        ++this->warp_tile_iterator_B01_;

        warp_mma(m5,
          warp_frag_A20[warp_mma_k % NumStages],
          warp_frag_B01[warp_mma_k % NumStages],
          m5);
        this->warp_tile_iterator_A23_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B0_.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A23_.load(warp_frag_A23[0]);
        this->warp_tile_iterator_B0_.load(warp_frag_B0[(warp_mma_k) % NumStages]);

        ++this->warp_tile_iterator_A23_;
        ++this->warp_tile_iterator_B0_;

        warp_mma(m1,
          warp_frag_A23[warp_mma_k % NumStages],
          warp_frag_B0[warp_mma_k % NumStages],
          m1);
      }
      gmem_wait();*/
    }
  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    assert(false);
    // First, increment remaining warp tiles to catch it up with the write stage.
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A0_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_A1_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B0_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B1_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A0_;
      ++this->warp_tile_iterator_A1_;
      ++this->warp_tile_iterator_B0_;
      ++this->warp_tile_iterator_B1_;
    }

    // If we bumped the read iterators to the end of the circular buffer, wrap them around to
    // align them with the write iterators
    if (smem_write_stage_idx == 0)
    {
      this->warp_tile_iterator_A0_.add_tile_offset(
        {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_A1_.add_tile_offset(
        {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B0_.add_tile_offset(
        {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_B1_.add_tile_offset(
        {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &m0, FragmentC &m1, 
    FragmentC &m2, FragmentC &m3, 
    FragmentC &m4, FragmentC &m5, 
    FragmentC &m6,
    IteratorA iterator_A0,                             ///< iterator over A operand in global memory
    IteratorA iterator_A1,
    IteratorA iterator_A2,
    IteratorA iterator_A3,
    IteratorB iterator_B0,                             ///< iterator over B operand in global memory
    IteratorB iterator_B1,
    IteratorB iterator_B2,
    IteratorB iterator_B3)
  {    
    // Prologue
    // prologue(iterator_A0, iterator_A1, iterator_A2, iterator_A3, 
    //          iterator_B0, iterator_B1, iterator_B2, iterator_B3,
    //          gemm_k_iterations);

    // // Wait until we have at least one completed global fetch stage
    // gmem_wait();

    // // // Perform accumulation in the 'd' output operand

    // // // // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations,
               m0, m1, m2, m3, m4, m5, m6, 
               iterator_A0, iterator_A1, iterator_A2, iterator_A3,
               iterator_B0, iterator_B1, iterator_B2, iterator_B3);
  }
};