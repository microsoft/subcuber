#pragma once

#include "cutlass/tensor_ref.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include <bit>
#include <type_traits>
#include <utility>

namespace StrassenType {
enum ENUM {
  Normal  = 0,
  Strassen = 1,
  StrassenWinograd = 2,
  Compressed = 3,

  NormalGlobalLevel1,
  CompressedGlobalLevel1,
  MatrixGlobalLevel1,
  NormalGlobalPreSumLevel1,
};
}

namespace MmaStrassen {

CUTLASS_HOST_DEVICE
int ABS(int i) {
  return i >=0 ? i : -i; 
}

CUTLASS_HOST_DEVICE
int SIGN(int i) {
  return i >=0 ? 1 : -1; 
}

enum Type {
  Normal                  = 0,
  Strassen                = 1,
  StrassenWinograd        = 2,

  Compressed              = 1,
  GlobalLevel1_M0         = 2,
  GlobalLevel1_M1         = 3,
  GlobalLevel1_M2         = 4,
  GlobalLevel1_M3         = 5,
  GlobalLevel1_M4         = 6,
  GlobalLevel1_M5         = 7,
  GlobalLevel1_M6         = 8,
  CompressedGlobalLevel1  = 9,
  MatrixGlobalLevel1_M0   = 10,
  MatrixGlobalLevel1_M1   = 11,
  MatrixGlobalLevel1_M2   = 12,
  MatrixGlobalLevel1_M3   = 13,
  MatrixGlobalLevel1_M4   = 14,
  MatrixGlobalLevel1_M5   = 15,
  MatrixGlobalLevel1_M6   = 16,

  GlobalPreSumLevel1_M0   = 17,
  GlobalPreSumLevel1_M1   = 18,
  GlobalPreSumLevel1_M2   = 19,
  GlobalPreSumLevel1_M3   = 20,
  GlobalPreSumLevel1_M4   = 21,
  GlobalPreSumLevel1_M5   = 22,
  GlobalPreSumLevel1_M6   = 23
};

CUTLASS_HOST CUTLASS_DEVICE
constexpr uint GetBit(uint64_t x, uint bit) {
  return (x >> bit) & 0x1;
}

CUTLASS_HOST CUTLASS_DEVICE
constexpr uint SetBit(uint64_t x, uint bit) {
  return x | (0x1 << bit);
}

CUTLASS_HOST CUTLASS_DEVICE
static constexpr int count_ones(uint x) {
  int c = 0;
  for (int i = 0; i < 31; i++) {
    if (((x >> i) & 0x1) == 1) c += 1;
  }
  return c;
}

template<int Inputs_>
class InputAccesses {
public:
  static const int Inputs = Inputs_;
private:
  bool access[Inputs] = {false};
  int indices[Inputs] = {0};
  int accessTrues = 0;

public:
  CUTLASS_HOST CUTLASS_DEVICE
  constexpr InputAccesses () {
    for (int i = 0; i < Inputs; i++)
      access[i] = false;
  }
  
  CUTLASS_HOST CUTLASS_DEVICE
  constexpr void addAccess(int i) {
    if (access[i] == false) {
      indices[i] = accessTrues;
      accessTrues++;
    }
    access[i] = true;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  constexpr void setIndex(int i, int idx) {
    indices[i] = idx;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  constexpr bool hasAccess(int i, int j = 0) {
    return (i + j) < Inputs && access[i + j];
  }

  CUTLASS_HOST CUTLASS_DEVICE
  constexpr uint numAccess() {
    uint n = 0;
    for (int i = 0; i < Inputs; i++)
      if (hasAccess(i)) n++;
    return n;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  constexpr uint index(int i) {return indices[i];}

  CUTLASS_HOST CUTLASS_DEVICE
  constexpr uint get_first_access_idx() {
    for (int i = 0; i < Inputs; i++)
      if (hasAccess(i)) return i;
    return 0;
  }
};

enum GlobalPresumType {
  PresumNo = 0,
  PresumYes = 1,
  PresumCompute = 2,
  PresumAvailable = 3,
  PresumGlobalKernel = 4
};

enum APresums {
  A0 = 0, APresum_M0 = 0,
  A1 = 1, APresum_M1 = 1,
  A2 = 2,
  A3 = 3, APresum_M6 = 3,
  A02 = 4, APresum_M3 = 4, APresumStart = 4,
  S1 = 5, APresum_M4 = 5,
  S2 = 6, APresum_M2 = 6,
  A1S2 = 7, APresum_M5 = 7,
  ANumPresums = 8
};

enum BPresums {
  B0 = 0, BPresum_M0 = 0,
  B1 = 1,
  B2 = 2, BPresum_M1 = 2,
  B3 = 3, BPresum_M5 = 3,
  B31 = 4, BPresum_M3 = 4, BPresumStart = 4,
  B10 = 5, BPresum_M4 = 5,
  S3 = 6, BPresum_M2 = 6,
  S3B2 = 7, BPresum_M6 = 7,
  BNumPresums = 8
};

template<GlobalPresumType kS2   = GlobalPresumType::PresumNo,
         GlobalPresumType kA02  = GlobalPresumType::PresumNo,
         GlobalPresumType kS1   = GlobalPresumType::PresumNo,
         GlobalPresumType kA1S2 = GlobalPresumType::PresumNo,
         GlobalPresumType kB31  = GlobalPresumType::PresumNo,
         GlobalPresumType kB10  = GlobalPresumType::PresumNo,
         GlobalPresumType kS3   = GlobalPresumType::PresumNo,
         GlobalPresumType kS3B2 = GlobalPresumType::PresumNo>
class AllPresums {
public:
  static const GlobalPresumType S2   = kS2;
  static const GlobalPresumType A02  = kA02;
  static const GlobalPresumType S1   = kS1;
  static const GlobalPresumType A1S2 = kA1S2;
  static const GlobalPresumType B10  = kB10;
  static const GlobalPresumType B31  = kB31;
  static const GlobalPresumType S3   = kS3;
  static const GlobalPresumType S3B2 = kS3B2;

  static constexpr int numAPresumYes() {
    int n = 0;
    if (S2   >= GlobalPresumType::PresumYes) n++;
    if (A02  >= GlobalPresumType::PresumYes) n++;
    if (S1   >= GlobalPresumType::PresumYes) n++;
    if (A1S2 >= GlobalPresumType::PresumYes) n++;
    return n;
  }

  static constexpr int numBPresumYes() {
    int n = 0;
    if (S3   >= GlobalPresumType::PresumYes) n++;
    if (B31  >= GlobalPresumType::PresumYes) n++;
    if (B10  >= GlobalPresumType::PresumYes) n++;
    if (S3B2 >= GlobalPresumType::PresumYes) n++;

    return n;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int indexAPresum(APresums presum) {
    int n = 0;
    if (A02  >= GlobalPresumType::PresumYes) {
      if (presum == APresums::A02) return n;
      n++;
    }
    if (S1   >= GlobalPresumType::PresumYes) {
      if (presum == APresums::S1) return n;
      n++;
    }
    if (S2   >= GlobalPresumType::PresumYes) {
      if (presum == APresums::S2) return n;
      n++;
    }
    if (A1S2 >= GlobalPresumType::PresumYes) {
      if (presum == APresums::A1S2) return n;
      n++;
    }
    return -1;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int indexBPresum(BPresums presum) {
    int n = 0;
    if (B31  >= GlobalPresumType::PresumYes) {
      if (presum == BPresums::B31) return n;
      n++;
    }
    if (B10  >= GlobalPresumType::PresumYes) {
      if (presum == BPresums::B10) return n;
      n++;
    }
    if (S3   >= GlobalPresumType::PresumYes) {
      if (presum == BPresums::S3) return n;
      n++;
    }
    if (S3B2 >= GlobalPresumType::PresumYes) {
      if (presum == BPresums::S3B2) return n;
      n++;
    }

    return -1;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr InputAccesses<APresums::ANumPresums> APresumComputeLoads(GlobalPresumType compute_type = PresumCompute) {
    InputAccesses<APresums::ANumPresums> accesses;

    if (S1 == compute_type) {
      accesses.addAccess(APresums::A2);
      accesses.addAccess(APresums::A3);
    }

    if (S2 == compute_type) {
      if (S1 == GlobalPresumType::PresumAvailable) {
        accesses.addAccess(APresums::S1);
      } else {
        accesses.addAccess(APresums::A2);
        accesses.addAccess(APresums::A3);
      }
      accesses.addAccess(APresums::A0);
    }

    if (A02 == compute_type) {
      accesses.addAccess(APresums::A0);
      accesses.addAccess(APresums::A2);
    }

    if (A1S2 == compute_type) {
      if (S2 == GlobalPresumType::PresumAvailable)
        accesses.addAccess(APresums::S2);
      else {
        if (S1 == GlobalPresumType::PresumAvailable) {
          accesses.addAccess(APresums::S1);
        } else {
          accesses.addAccess(APresums::A2);
          accesses.addAccess(APresums::A3);
        }
        accesses.addAccess(APresums::A0);
      }

      accesses.addAccess(APresums::A1);
    }

    return accesses;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr InputAccesses<BPresums::BNumPresums> BPresumComputeLoads(GlobalPresumType compute_type = PresumCompute) {
    InputAccesses<BPresums::BNumPresums> accesses;

    if (B10 == compute_type) {
      accesses.addAccess(BPresums::B1);
      accesses.addAccess(BPresums::B0);
    }

    if (S3 == compute_type) {
      if (B31 == GlobalPresumType::PresumAvailable) {
        accesses.addAccess(BPresums::B31);
      } else {
        accesses.addAccess(BPresums::B1);
        accesses.addAccess(BPresums::B3);
      }
      accesses.addAccess(BPresums::B0);
    }

    if (B31 == compute_type) {
      accesses.addAccess(BPresums::B3);
      accesses.addAccess(BPresums::B1);
    }

    if (S3B2 == compute_type) {
      if (S3 == GlobalPresumType::PresumAvailable)
        accesses.addAccess(BPresums::S3);
      else {
        if (B31 == GlobalPresumType::PresumAvailable) {
          accesses.addAccess(BPresums::B31);
        } else {
          accesses.addAccess(BPresums::B1);
          accesses.addAccess(BPresums::B3);
        }
        accesses.addAccess(BPresums::B0);
      }

      accesses.addAccess(BPresums::B2);
    }

    return accesses;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool computeAnyAPresum(GlobalPresumType compute_type = PresumCompute) {
    if (S2   == compute_type) return true;
    if (S1   == compute_type) return true;
    if (A02  == compute_type) return true;
    if (A1S2 == compute_type) return true;
    
    return false;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool computeAnyBPresum(GlobalPresumType compute_type = PresumCompute) {
    if (B10   == compute_type) return true;
    if (B31   == compute_type) return true;
    if (S3  == compute_type) return true;
    if (S3B2 == compute_type) return true;
    
    return false;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool computeAllAPresums(GlobalPresumType compute_type = PresumCompute) {
    return (S2  == compute_type) && (S1   == compute_type) &&
           (A02 == compute_type) && (A1S2 == compute_type);
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool computeAllBPresums(GlobalPresumType compute_type = PresumCompute) {
    return (B10 == compute_type) && (B31  == compute_type) &&
           (S3  == compute_type) && (S3B2 == compute_type);
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool doesComputeA(APresums presum, GlobalPresumType compute_type = PresumCompute) {
    if (S2   == compute_type && presum == APresums::S2)   return true;
    if (S1   == compute_type && presum == APresums::S1)   return true;
    if (A02  == compute_type && presum == APresums::A02)  return true;
    if (A1S2 == compute_type && presum == APresums::A1S2) return true;
    
    return false;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool doesComputeB(BPresums presum, GlobalPresumType compute_type = PresumCompute) {
    if (B10  == compute_type && presum == BPresums::B10)  return true;
    if (B31  == compute_type && presum == BPresums::B31)  return true;
    if (S3   == compute_type && presum == BPresums::S3)   return true;
    if (S3B2 == compute_type && presum == BPresums::S3B2) return true;
    
    return false;
  }
};

enum TaskOutputM {
  OutputMNone = 0,
  WriteShared = 1,
  ReadShared = 2,
  WriteGlobal = 3,
  ReadGlobal = 4,
  UpdateGlobal = 5,
  IncrementGlobal = 6,
  ContinueAccums = 7,
  KeepAccums = 8,
  WriteInterim = 9,
  UpdateInterim = 10,
};

enum TaskOutputC {
  OutputCNone = 0,
  UpdateC = 1,
  WriteC = 2
};

enum ReadWriteM {
  RWNone              = 0UL,
  RWShared            = 1UL,
  RWGlobalContig      = 2UL,
  RWGlobalTensorOpMap = 3UL,
  RWAccumInShared     = 4UL,
  RWSharedAddMiddle   = 5UL,
  RWSharedAddEnd      = 6UL,
  RWGlobalContigAndShared = 7UL,
  RWAccumInSharedInEpilogue = 8UL,
  RWSharedInEpilogue  = 9UL,
  RWUseSign           = 10UL,
  RWCombineMMALoop    = 11UL,
  RWBits              = 4UL,
  RWBitsAugmented     = 6UL,
  RWOutputCBits       = 3UL,
};

using ReadWriteOp = ReadWriteM;

enum MemLayout {
  LayoutNone = 0,
  LayoutInterim = 1,
  LayoutInterim1D = 2,
  LayoutFinal = 3
};

enum MemType {
  MemTypeNone = 0,
  MemRegisters = 1,
  MemGlobal = 2,
  MemGlobalAsync = 3,
  MemShared = 4
};

template<int kMi, int kVal>
class RW {
public:
  static const int Mi = kMi;
  static const int Val = kVal;
};

struct PostsumOp {
private:
  int op; 
  int sign;
  MemLayout mem_layout;
  MemType mem_type;

public:
  CUTLASS_HOST_DEVICE
  constexpr PostsumOp() : op(-1), sign(0), mem_layout(LayoutNone), mem_type(MemTypeNone) {}

  CUTLASS_HOST_DEVICE
  constexpr PostsumOp(int op, int sign, MemType mem_type, MemLayout mem_layout) :
    op(op), sign(sign), mem_type(mem_type), mem_layout(mem_layout) {}

  // CUTLASS_HOST_DEVICE
  // constexpr PostsumOp() : op(-1), sign(0), mem_layout(LayoutNone), mem_type(MemTypeNone) {}

  // CUTLASS_HOST_DEVICE
  // constexpr PostsumOp(int op, int sign, MemType mem_type, MemLayout mem_layout) :
  //   op(op), sign(sign), mem_type(mem_type), mem_layout(mem_layout) {}

  CUTLASS_HOST_DEVICE
  PostsumOp(const PostsumOp& other):
    PostsumOp(other.op, other.sign, other.mem_type, other.mem_layout) {}

  template<typename Expr>
  CUTLASS_HOST_DEVICE
  PostsumOp(const Expr expr):
    PostsumOp(expr.Op, expr.Sign, (MemType)expr.MemType, (MemLayout)expr.MemLayout) {}

  CUTLASS_HOST_DEVICE
  constexpr int get_sign() const {
    return sign;
  }

  CUTLASS_HOST_DEVICE
  constexpr int get_op() const {
    return op;
  }

  CUTLASS_HOST_DEVICE
  constexpr bool is_layout_interim() const {
    return mem_layout == LayoutInterim1D || mem_layout == LayoutInterim;
  }

  CUTLASS_HOST_DEVICE
  constexpr bool is_layout_final() const {
    return mem_layout == LayoutFinal;
  }

  CUTLASS_HOST_DEVICE
  constexpr MemLayout get_mem_layout() const {
    return mem_layout;
  }

  CUTLASS_HOST_DEVICE
  constexpr bool is_mem_global() const {
    return mem_type == MemGlobal;
  }

  CUTLASS_HOST_DEVICE
  constexpr bool is_mem_shared() const {
    return mem_type == MemShared;// || mem_type == MemGlobalAsync;
  }

  CUTLASS_HOST_DEVICE
  constexpr bool valid() const {
    return op >= 0 && sign != 0 &&
           mem_type != MemTypeNone &&
           mem_layout != LayoutNone;
  }
};

template<int kOp, MemType kMemType = MemTypeNone, MemLayout kLayout = LayoutNone>
class Plus {
public:
  static const int Op = kOp;
  static const int Sign = 1;
  static const int MemLayout = kLayout;
  static const int MemType = kMemType;
  static const int RWOp = int(kLayout);

  CUTLASS_HOST_DEVICE
  static PostsumOp Adder() {
    return PostsumOp(Plus());
  }
};

template<int kOp, MemType kMemType = MemTypeNone, MemLayout kLayout = LayoutNone>
class Neg {
public:
  static const int Op = kOp;
  static const int Sign = -1;
  static const int MemLayout = kLayout;
  static const int MemType = kMemType;
  static const int RWOp = int(kLayout);

  CUTLASS_HOST_DEVICE
  static PostsumOp Adder() {
    return PostsumOp(Neg());
  }
};

template<int kOp, typename... SubExprs>
struct FindExprOpType {
  using type = Plus<-1, MemTypeNone, LayoutNone>;
};

template<int kOp, typename SubExpr, typename... Rest>
struct FindExprOpType<kOp, SubExpr, Rest...> {
  using type = cute::conditional_t<
    (SubExpr::Op == kOp),
    SubExpr,
    typename FindExprOpType<kOp, Rest...>::type>;
};

template<int kSrcIndex, typename... SubExprs>
struct FindExprOpTypeByIndex {
  using type = Plus<-1, MemTypeNone, LayoutNone>;
};

template<int kSrcIndex, typename SubExpr, typename... Rest>
struct FindExprOpTypeByIndex<kSrcIndex, SubExpr, Rest...> {
  using type = cute::conditional_t<
    (kSrcIndex == 0),
    SubExpr,
    typename FindExprOpTypeByIndex<kSrcIndex - 1, Rest...>::type>;
};

template<typename... SubExprs>
class Expr {
public:
  template<int kOp>
  using AdderType = typename FindExprOpType<kOp, SubExprs...>::type;

  template<int kSrcIndex>
  using AdderTypeByIndex = typename FindExprOpTypeByIndex<kSrcIndex, SubExprs...>::type;

  CUTLASS_HOST_DEVICE
  static PostsumOp Adder(int op) {
    PostsumOp val;
    bool found = (... || (SubExprs::Op == op ? (val = SubExprs::Adder(), true)
                                             : false));
    return val;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp AdderByIndex(int src_index) {
    PostsumOp val;
    int index = 0;
    bool found = (... || (index++ == src_index ? (val = SubExprs::Adder(), true)
                                                : false));
    (void)found;
    return val;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp GlobalAsyncAdder() {
    PostsumOp val;
    bool _ = (... || (SubExprs::MemType == MemType::MemGlobalAsync ? (val = SubExprs::Adder(), true)
                                                                   : false)); 
    return val;
  }

  CUTLASS_HOST_DEVICE
  static constexpr TaskOutputM RWOp(int op) {
    int val = 0;
    bool found = (... || (SubExprs::Op == op ? (val = SubExprs::RWOp, true) : false));
    return TaskOutputM(val);
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool NeedSharedMem() {
    return (... || (SubExprs::MemType == MemGlobalAsync ? true : false));
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasGlobalLoad() {
    return (... || (SubExprs::MemType == MemType::MemGlobal));
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool IsEmpty() {
    return sizeof...(SubExprs) == 0;
  }
};

template<int myVal = 0, typename... RWs>
class RWMTypes {
public:
  static const int MyVal = myVal; 
  CUTLASS_HOST_DEVICE
  static constexpr TaskOutputM GetRWForMi(int Mi) {
    TaskOutputM rw;
    bool found = (... || (RWs::Mi == Mi ? (rw = RWs::Val, true) : false));
    return (found == false) ? OutputMNone : rw;
  }
};

template<int kDestC, MemLayout kGlobalLayout, MemLayout kSharedLayout, class MExpr = Expr<>, class CExpr = Expr<>>
class CUW {
public:
  static const int DestC = kDestC;
  using GlobalDestOpType = Plus<kDestC, MemGlobal, kGlobalLayout>;
  using SharedDestOpType = Plus<kDestC, MemShared, kSharedLayout>;
  using CExprType = CExpr;

  CUTLASS_HOST_DEVICE
  static PostsumOp Adder(int ci) {
    return CExpr::Adder(ci);
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp GlobalDest() {
    return PostsumOp(kDestC, 1, MemGlobal, kGlobalLayout);
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp SharedDest() {
    return PostsumOp(kDestC, 1, MemShared, kSharedLayout);
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasSharedDest() {
    return kSharedLayout != LayoutNone;
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasGlobalDest() {
    return kGlobalLayout != LayoutNone;
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasGlobalAsyncLd() {
    return CExpr::NeedSharedMem();
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp PostsumSrcGlobalAsync() {
    return CExpr::GlobalAsyncAdder();
  }

  CUTLASS_HOST_DEVICE
  static constexpr int HasCSrc() {
    return CExpr::IsEmpty();
  }

  CUTLASS_HOST_DEVICE
  static constexpr int HasGlobalLoad() {
    return CExpr::HasGlobalLoad();
  }

  CUTLASS_HOST_DEVICE
  static constexpr int SignM(int mi) {
    return MExpr::Adder(mi).get_sign();
  }
};

template<typename... CUWs>
class RWCTypes {
public:
  CUTLASS_HOST_DEVICE
  static constexpr int SignForCoWithMi(int co, int mi) {
    int sign = 0;
    bool found = (... || ((CUWs::DestC == co && CUWs::SignM(mi) != 0) ? (sign = CUWs::SignM(mi), true) : false));
    return sign;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp GlobalOutputOp(int Co) {
    PostsumOp layout;
    bool found = (... || (CUWs::DestC == Co ? (layout = CUWs::GlobalOutputOp(), true) : false));
    return layout;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp SharedOutputOp(int Co) {
    PostsumOp layout;
    bool found = (... || (CUWs::DestC == Co ? (layout = CUWs::SharedOutputOp(), true) : false));
    return layout;
  }

  CUTLASS_HOST_DEVICE
  static constexpr PostsumOp PostsumSrcs(int Co, int Ci) {
    PostsumOp adder;
    bool found = (... || ((CUWs::DestC == Co && CUWs::Adder(Ci).valid()) ? (adder = CUWs::Adder(Ci), true) : false));
    return adder;
  }

  CUTLASS_HOST_DEVICE
  static bool HasGlobalSrcLoad() {
    return (... || CUWs::HasGlobalLoad());
  }

  CUTLASS_HOST_DEVICE
  static bool HasPostsumSrc(int Co) {
    bool found = (... || ((CUWs::DestC == Co) ? (CUWs::HasCSrc(), true) : false));
    return found;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp PostsumGlobalDest(int DestC) {
    PostsumOp dst;
    bool found = (... || ((CUWs::DestC == DestC) ? (dst = CUWs::GlobalDest(), true) : false));
    return dst;
  }

  CUTLASS_HOST_DEVICE
  static constexpr PostsumOp PostsumGlobalDestByOutputIndex(int output_index) {
    PostsumOp dst;
    int index = 0;
    bool found = (... || (CUWs::HasGlobalDest() //TODO This works for Hopper F16 and F32 but not Ampere F16
      ? (index == output_index ? (dst = CUWs::GlobalDest(), true) : (index += 1, false))
      : false));
    (void)found;
    return dst;
  }

  CUTLASS_HOST_DEVICE
  static int MiSignByOutputIndex(int output_index, int mi) {
    int sign = 0;
    int index = 0;
    bool found = (... || (CUWs::HasGlobalDest()
      ? (index == output_index && CUWs::SignM(mi) != 0 ? (sign = CUWs::SignM(mi), true) : (index += 1, false))
      : false));
    (void)found;
    return sign;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp PostsumSharedDest(int DestC) {
    PostsumOp dst;
    bool found = (... || ((CUWs::DestC == DestC) ? (dst = CUWs::SharedDest(), true) : false));
    return dst;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp PostsumSharedDestByOutputIndex(int output_index) {
    PostsumOp dst;
    int index = 0;
    bool found = (... || (CUWs::HasSharedDest()
      ? (index == output_index ? (dst = CUWs::SharedDest(), true) : (index += 1, false))
      : false));
    (void)found;
    return dst;
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp PostsumSrcByOutputIndex(int output_index, int src_index) {
    PostsumOp src;
    int index = 0;
    bool found = (... || (CUWs::HasGlobalDest() //TODO: Is this comparison needed?
      ? (index == output_index
          ? ((src = CUWs::CExprType::AdderByIndex(src_index)).valid(), true)
          : (index += 1, false))
      : false));
    (void)found;
    return src;
  }

  CUTLASS_HOST_DEVICE
  static constexpr int CFusedSharedBuffs() {
    int n = 0;
    bool _ = (... || (CUWs::HasSharedDest() ? (n += 1, true) : false));
    return n;
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasGlobalAsyncLd() {
    return (... || (CUWs::HasGlobalAsyncLd() ? true : false));
  }

  CUTLASS_HOST_DEVICE
  static PostsumOp PostsumSrcGlobalAsync() {
    //TODO: Only one GlobalAsync is allowed
    PostsumOp adder;
    bool _ = (... || (CUWs::HasGlobalAsyncLd() ? (adder = CUWs::PostsumSrcGlobalAsync(), true) : false));
    return adder;
  }

  CUTLASS_DEVICE
  static constexpr int NumGlobalStores() {
    int n = 0;
    bool _ = (... || (CUWs::HasGlobalDest() ? (n += 1, true) : false));
    return n;
  }

  CUTLASS_DEVICE
  static bool HasGlobalMemoryLoadForSingleM() {
    return false;  
  }

  CUTLASS_DEVICE
  static constexpr int NumCUWs() {
    return sizeof...(CUWs);
  }
  // CUTLASS_HOST_DEVICE
  // template<typename Fn>
  // static constexpr void ProcessAllCUWs(Fn fn) {
  //   (fn())
  // }
};

template<int kStrassenLevel = 1,
         int kLevel1Idx_ = 0,
         typename ThreadBlockShape_ = cutlass::gemm::GemmShape<128, 128, 32>,
         typename AllPresums_ = AllPresums<>>
class StrassenPresum {
public:
  static const int Level = kStrassenLevel;
  static constexpr int Level1Idx = kLevel1Idx_;
  using ThreadBlockShape = ThreadBlockShape_;

  using AllPresums = AllPresums_;
};


template<int kStrassenLevel = 1,
         int kLevel1Idx_ = 0,
         typename ThreadBlockShape_ = cutlass::gemm::GemmShape<128, 128, 32>,
         typename WarpShape_ = cutlass::gemm::GemmShape<64, 32, 32>,
         uint StageCount_ = 5,
         typename RWMTypes_ = RWMTypes<>, typename RWCTypes_ = RWCTypes<>,
         typename AllPresums_ = AllPresums<>,
         int kFusedOrContinueMMA = 0, //TODO: 0 for fused and 1 for continue
         int... kMis>
class StrassenLevel1MiGroup {
public:
  static const int Level = kStrassenLevel;
  static constexpr int Level1Idx = kLevel1Idx_;
  using ThreadBlockShape = ThreadBlockShape_;
  using WarpShape = WarpShape_;
#ifdef CUTLASS_API_v3
  using StageCountType = cute::Int<StageCount_>;
#elif defined(CUTLASS_API_v2)
  static const uint StageCountType = StageCount_;
#else
  #error("CUTLASS_API_v3 or CUTLASS_API_v2 not defined")
#endif
  using AllPresums = AllPresums_;
  using APresums = MmaStrassen::APresums;
  using BPresums = MmaStrassen::BPresums;
  using RWMTypes = RWMTypes_;
  using RWCTypes = RWCTypes_;

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasM0() {return hasMi(0);}

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasM1() {return hasMi(1);}

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasM2() {return hasMi(2);}

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasM3() {return hasMi(3);}

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasM4() {return hasMi(4);}

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasM5() {return hasMi(5);}

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasM6() {return hasMi(6);}

  CUTLASS_HOST_DEVICE
  static constexpr int FusedOrContinueMMA() {return kFusedOrContinueMMA;}

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasAllM() {
    for (int i = 0; i < 7; i++)
      if (!hasMi(i)) return false;

    return true;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasAnyM() {
    for (int i = 0; i < 7; i++)
      if (hasMi(i)) return true;

    return false;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasMi(int mi) {//TODO: Change this to hasFusedMi
    if (kFusedOrContinueMMA == 1) {
      int first = -1;
      bool _ = (... || (first = kMis, true));
      return first == mi;
    }

    bool found = false;
    found = (... || (kMis == mi));
    return found;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int NumContinueMis() {
    if (kFusedOrContinueMMA == 0) return 1;
    return sizeof...(kMis);
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr uint getMi(int i = 0) {
    int index = 0;
    int mi = 0;
    bool found = ((index++ == i ? (mi = kMis, true) : false) || ...);
    (void)found;
    return static_cast<uint>(mi);
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool isEmpty() {
    return !hasAnyM();
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int numMs() {
    if (kFusedOrContinueMMA == 0)
      return sizeof...(kMis);
    else
      return 1;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr bool hasManyMs() {
    return numMs() >= 2;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int TileMDivisor() {
    if constexpr (numMs() == 1 or numMs() == 2) {
      return numMs();
    } else if constexpr (numMs() == 4 or numMs() == 7) {
      return 2;
    } else {
      return 1;
    }
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int TileNDivisor() {
    if constexpr (numMs() == 1 or numMs() == 2) {
      return 1;
    } else if constexpr (numMs() == 4 or numMs() == 7) {
      return 2;
    } else {
      return 1;
    }
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int TileOffsetMDivisor() {
    if constexpr (numMs() == 1 or numMs() == 2) {
      return numMs();
    } else if constexpr (numMs() == 4) {
      return 2;
    }
    return 1;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int TileOffsetNDivisor() {
    if constexpr (numMs() == 1 or numMs() == 2) {
      return 1;
    } else if constexpr (numMs() == 4) {
      return 2;
    }
    return 1;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int GridMMultiplier() {
    if constexpr (numMs() == 1 or numMs() == 2) {
      return numMs();
    } else if constexpr (numMs() == 4 or numMs() == 7) {
      return 2;
    } else {
      return 1;
    }
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr int GridNMultiplier() {
    if constexpr (numMs() == 1 or numMs() == 2) {
      return 1;
    } else if constexpr (numMs() == 4 or numMs() == 7) {
      return 2;
    } else {
      return 1;
    }
  }

  static constexpr int CFusedSharedBuffs() {
    return RWCTypes::CFusedSharedBuffs();
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasGlobalAsyncLd() {
    return RWCTypes::HasGlobalAsyncLd();
  }

  //TODO: Make function AInputLoad?
  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr InputAccesses<APresums::ANumPresums> APresumLoads() {
    InputAccesses<APresums::ANumPresums> accesses;
           if (hasM0()) {
      accesses.addAccess(0);
    }
    if (hasM1()) {
      accesses.addAccess(1);
    }
    if (hasM2()){
      if (AllPresums::S2 == GlobalPresumType::PresumNo) {
        if (AllPresums::S1 == GlobalPresumType::PresumAvailable) {
          accesses.addAccess(APresums::S1);
        } else {
          accesses.addAccess(2);
          accesses.addAccess(3);
        }
        accesses.addAccess(0);
      } else {
        if (AllPresums::S2 == GlobalPresumType::PresumAvailable)
          accesses.addAccess(APresums::S2);
      }
    } 
    if (hasM3()) {
      if (AllPresums::A02 == GlobalPresumType::PresumNo) {
        accesses.addAccess(2);
        accesses.addAccess(0);
      } else {
        if (AllPresums::S2 == GlobalPresumType::PresumAvailable)
          accesses.addAccess(APresums::A02);
      }
    }
    if (hasM4()) {
      if (AllPresums::S1 == GlobalPresumType::PresumNo) {
        accesses.addAccess(2);
        accesses.addAccess(3);
      } else {
        if (AllPresums::S1 == GlobalPresumType::PresumAvailable)
          accesses.addAccess(APresums::S1);
      }
    }
    if (hasM5()) {
      if (AllPresums::A1S2 == GlobalPresumType::PresumNo) {
        accesses.addAccess(1);

        if (AllPresums::S2 == GlobalPresumType::PresumAvailable) {
          accesses.addAccess(APresums::S2);
        } else {
          accesses.addAccess(0);
          if (AllPresums::S1 == GlobalPresumType::PresumAvailable) {
            accesses.addAccess(APresums::S1);
          } else {
            accesses.addAccess(2);
            accesses.addAccess(3);
          } 
        }
      } else {
        if (AllPresums::A1S2 == GlobalPresumType::PresumAvailable)
          accesses.addAccess(APresums::A1S2);
      }
    }
    if (hasM6()) {
      accesses.addAccess(3);
    }

    return accesses;
  }

  //TODO: change to AInputsInSharedMem
  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr InputAccesses<APresums::ANumPresums> APresumStores(bool shareIndices=false) {
    InputAccesses<APresums::ANumPresums> accesses;

    if (hasM0()) {
      accesses.addAccess(APresums::A0);
    } 
    if (hasM1()) {
      accesses.addAccess(APresums::A1);
    }
    if (hasM6()) {
      accesses.addAccess(APresums::A3);
    }
    if (hasM2()) {
      accesses.addAccess(APresums::S2);
    } 
    if (hasM3()) {
      accesses.addAccess(APresums::A02);
    } 
    if (hasM4()) {
      accesses.addAccess(APresums::S1);
    }
    if (hasM5()) {
      accesses.addAccess(APresums::A1S2);
    }

    if (shareIndices) {
      //For A0, A1, A3, and computed presums set the same shared memory index as PresumLoads
      auto loads = APresumLoads();
      int indicesUpdated = 0;
      int indicesAvailable[accesses.Inputs] = {0};
      int indicesAvailableNum = 0;
      
      for (int i = 0; i < accesses.Inputs; i++) {
        if (loads.hasAccess(i)) {
          if (accesses.hasAccess(i)) {
            accesses.setIndex(i, loads.index(i));
            indicesUpdated++;
          } else {
            indicesAvailable[indicesAvailableNum++] = loads.index(i);
          }
        }
      }

      for (int i = 0; i < accesses.Inputs; i++) {
        if (accesses.hasAccess(i) && not loads.hasAccess(i)) {
          if (indicesAvailableNum > 0) {
            accesses.setIndex(i, indicesAvailable[--indicesAvailableNum]);
            indicesUpdated++;
          } else {
            accesses.setIndex(i, indicesUpdated++);
          }
        }
      }
    }

    return accesses;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr uint APresumSharedBuffs() {
    return APresumStores().numAccess();
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr InputAccesses<BPresums::BNumPresums> BPresumLoads() {
    InputAccesses<BPresums::BNumPresums> accesses;
           if (hasM0()) {
      accesses.addAccess(0);
    }
    if (hasM1()) {
      accesses.addAccess(2);
    }
    if (hasM2()) {
      if (AllPresums::S3 == GlobalPresumType::PresumNo) {
        accesses.addAccess(0);
        if (AllPresums::B31 == GlobalPresumType::PresumNo) {
          accesses.addAccess(3);
          accesses.addAccess(1);
        } else if (AllPresums::B31 == GlobalPresumType::PresumAvailable) {
          accesses.addAccess(BPresums::B31);
        }
      } else if (AllPresums::S3 == GlobalPresumType::PresumAvailable) {
        accesses.addAccess(BPresums::S3);
      }
    }
    if (hasM3()) {
      if (AllPresums::B31 == GlobalPresumType::PresumNo) {
        accesses.addAccess(3);
        accesses.addAccess(1);
      } else if (AllPresums::B31 == GlobalPresumType::PresumAvailable) {
        accesses.addAccess(BPresums::B31);
      }
    }
    if (hasM4()) {
      if (AllPresums::B10 == GlobalPresumType::PresumNo) {
        accesses.addAccess(1);
        accesses.addAccess(0);
      } else if (AllPresums::B10 == GlobalPresumType::PresumAvailable) {
        accesses.addAccess(BPresums::B10);
      }
    }
    if (hasM5()) {
      accesses.addAccess(3);
    } 
    if (hasM6()) {
      if (AllPresums::S3B2 == GlobalPresumType::PresumNo) {
        accesses.addAccess(2);
        if (AllPresums::S3 == GlobalPresumType::PresumNo) {
          accesses.addAccess(0);
          if (AllPresums::B31 == GlobalPresumType::PresumNo) {
            accesses.addAccess(3);
            accesses.addAccess(1);
          } else if (AllPresums::B31 == GlobalPresumType::PresumAvailable) {
            accesses.addAccess(BPresums::B31);
          }
        } else if (AllPresums::S3 == GlobalPresumType::PresumAvailable) {
          accesses.addAccess(BPresums::S3);
        }
      } else if (AllPresums::S3B2 == GlobalPresumType::PresumAvailable) {
        accesses.addAccess(BPresums::S3B2);
      }
    }

    return accesses;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr InputAccesses<BPresums::BNumPresums> BPresumStores(bool shareIndices = false) {
    InputAccesses<BPresums::BNumPresums> accesses;
    if (hasM0()) {
      accesses.addAccess(BPresums::B0);
    } 
    if (hasM1()) {
      accesses.addAccess(BPresums::B2);
    }
    if (hasM5()) {
      accesses.addAccess(BPresums::B3);
    } 
    if (hasM2()) {
      accesses.addAccess(BPresums::S3);
    } 
    if (hasM3()) {
      accesses.addAccess(BPresums::B31);
    }
    if (hasM4()) {
      accesses.addAccess(BPresums::B10);
    }
    if (hasM6()) {
      accesses.addAccess(BPresums::S3B2);
    }

    if (shareIndices) {
      //For B0, B2, B3, and already computed presums set the same shared memory index as PresumLoads
      auto loads = BPresumLoads();
      int indicesUpdated = 0;
      int indicesAvailable[accesses.Inputs] = {0};
      int indicesAvailableNum = 0;

      for (int i = 0; i < accesses.Inputs; i++) {
        if (loads.hasAccess(i)) {
          if (accesses.hasAccess(i)) {
            accesses.setIndex(i, loads.index(i));
            indicesUpdated++;
          } else {
            indicesAvailable[indicesAvailableNum++] = loads.index(i);
          }
        }
      }

      for (int i = 0; i < accesses.Inputs; i++) {
        if (accesses.hasAccess(i) && not loads.hasAccess(i)) {
          if (indicesAvailableNum > 0) {
            accesses.setIndex(i, indicesAvailable[--indicesAvailableNum]);
            indicesUpdated++;
          } else {
            accesses.setIndex(i, indicesUpdated++);
          }
        }
      }
    }

    return accesses;
  }

  CUTLASS_HOST CUTLASS_DEVICE
  static constexpr uint BPresumSharedBuffs() {
    return BPresumStores().numAccess();
  }
};

template<int level, int kLevel1Idx, typename ThreadBlockShape, typename WarpShape, int StageCount, typename RWMTypes = RWMTypes<>, typename RWCTypes = RWCTypes<>, typename AllPresums = AllPresums<>>
using StrassenLevel1M0Group = StrassenLevel1MiGroup<level, kLevel1Idx, ThreadBlockShape, WarpShape, StageCount, RWMTypes, RWCTypes, AllPresums, 0, 0>;

template<int level, int kLevel1Idx, typename ThreadBlockShape, typename WarpShape, int StageCount, typename RWMTypes = RWMTypes<>, typename RWCTypes = RWCTypes<>, typename AllPresums = AllPresums<>>
using StrassenLevel1M1Group = StrassenLevel1MiGroup<level, kLevel1Idx, ThreadBlockShape, WarpShape, StageCount, RWMTypes, RWCTypes, AllPresums, 0, 1>;

template<int level, int kLevel1Idx, typename ThreadBlockShape, typename WarpShape, int StageCount, typename RWMTypes = RWMTypes<>, typename RWCTypes = RWCTypes<>, typename AllPresums = AllPresums<>>
using StrassenLevel1M2Group = StrassenLevel1MiGroup<level, kLevel1Idx, ThreadBlockShape, WarpShape, StageCount, RWMTypes, RWCTypes, AllPresums, 0, 2>;

template<int level, int kLevel1Idx, typename ThreadBlockShape, typename WarpShape, int StageCount, typename RWMTypes = RWMTypes<>, typename RWCTypes = RWCTypes<>, typename AllPresums = AllPresums<>>
using StrassenLevel1M3Group = StrassenLevel1MiGroup<level, kLevel1Idx, ThreadBlockShape, WarpShape, StageCount, RWMTypes, RWCTypes, AllPresums, 0, 3>;

template<int level, int kLevel1Idx, typename ThreadBlockShape, typename WarpShape, int StageCount,  typename RWMTypes = RWMTypes<>, typename RWCTypes = RWCTypes<>, typename AllPresums = AllPresums<>>
using StrassenLevel1M4Group = StrassenLevel1MiGroup<level, kLevel1Idx, ThreadBlockShape, WarpShape, StageCount, RWMTypes, RWCTypes, AllPresums, 0, 4>;

template<int level, int kLevel1Idx, typename ThreadBlockShape, typename WarpShape, int StageCount, typename RWMTypes = RWMTypes<>, typename RWCTypes = RWCTypes<>, typename AllPresums = AllPresums<>>
using StrassenLevel1M5Group = StrassenLevel1MiGroup<level, kLevel1Idx, ThreadBlockShape, WarpShape, StageCount, RWMTypes, RWCTypes, AllPresums, 0, 5>;

template<int level, int kLevel1Idx, typename ThreadBlockShape, typename WarpShape, int StageCount, typename RWMTypes = RWMTypes<>, typename RWCTypes = RWCTypes<>, typename AllPresums = AllPresums<>>
using StrassenLevel1M6Group = StrassenLevel1MiGroup<level, kLevel1Idx, ThreadBlockShape, WarpShape, StageCount, RWMTypes, RWCTypes, AllPresums, 0, 6>;


template<typename kPresum = StrassenPresum<1>,
         typename kGroup0 = StrassenLevel1MiGroup<1>,
         typename kGroup1 = StrassenLevel1MiGroup<1>,
         typename kGroup2 = StrassenLevel1MiGroup<1>,
         typename kGroup3 = StrassenLevel1MiGroup<1>,
         typename kGroup4 = StrassenLevel1MiGroup<1>,
         typename kGroup5 = StrassenLevel1MiGroup<1>,
         typename kGroup6 = StrassenLevel1MiGroup<1>>
class StrassenLevel1Groups {
public:
  using PresumGroup = kPresum;
  using Group0 = kGroup0;
  using Group1 = kGroup1;
  using Group2 = kGroup2;
  using Group3 = kGroup3;
  using Group4 = kGroup4;
  using Group5 = kGroup5;
  using Group6 = kGroup6;
};

// template<typename ThreadBlockShape, typename WarpShape>
// using StrassenLevel1SingleMiGroup = StrassenLevel1Groups<StrassenLevel1MiGroup<ThreadBlockShape, WarpShape, true>,
//                                                          StrassenLevel1MiGroup<ThreadBlockShape, WarpShape, false, true>,
//                                                          StrassenLevel1MiGroup<ThreadBlockShape, WarpShape, false, false, true>,
//                                                          StrassenLevel1MiGroup<ThreadBlockShape, WarpShape, false, false, false, true>,
//                                                          StrassenLevel1MiGroup<ThreadBlockShape, WarpShape, false, false, false, false, true>,
//                                                          StrassenLevel1MiGroup<ThreadBlockShape, WarpShape, false, false, false, false, false, true>,
//                                                          StrassenLevel1MiGroup<ThreadBlockShape, WarpShape, false, false, false, false, false, false, true>>;

template<int level = 1, int level_1_idx = 0,
         typename ThreadBlockShape1 = cutlass::gemm::GemmShape<1,1,1>, typename WarpShape1 = cutlass::gemm::GemmShape<1,1,1>, uint StageCount1 = 3,
         typename RWMTypes1 = RWMTypes<>, typename RWCTypes1 = RWCTypes<>, typename AllPresums1 = AllPresums<>,
         typename ThreadBlockShape2 = cutlass::gemm::GemmShape<1,1,1>, typename WarpShape2 = cutlass::gemm::GemmShape<1,1,1>, uint StageCount2 = 3,
         typename RWMTypes2 = RWMTypes<>, typename RWCTypes2 = RWCTypes<>, typename AllPresums2 = AllPresums<>,
         typename ThreadBlockShape3 = cutlass::gemm::GemmShape<1,1,1>, typename WarpShape3 = cutlass::gemm::GemmShape<1,1,1>, uint StageCount3 = 3,
         typename RWMTypes3 = RWMTypes<>, typename RWCTypes3 = RWCTypes<>, typename AllPresums3 = AllPresums<>,
         typename ThreadBlockShape4 = cutlass::gemm::GemmShape<1,1,1>, typename WarpShape4 = cutlass::gemm::GemmShape<1,1,1>, uint StageCount4 = 3,
         typename RWMTypes4 = RWMTypes<>, typename RWCTypes4 = RWCTypes<>, typename AllPresums4 = AllPresums<>,
         typename ThreadBlockShape5 = cutlass::gemm::GemmShape<1,1,1>, typename WarpShape5 = cutlass::gemm::GemmShape<1,1,1>, uint StageCount5 = 3,
         typename RWMTypes5 = RWMTypes<>, typename RWCTypes5 = RWCTypes<>, typename AllPresums5 = AllPresums<>,
         typename ThreadBlockShape6 = cutlass::gemm::GemmShape<1,1,1>, typename WarpShape6 = cutlass::gemm::GemmShape<1,1,1>, uint StageCount6 = 3,
         typename RWMTypes6 = RWMTypes<>, typename RWCTypes6 = RWCTypes<>, typename AllPresums6 = AllPresums<>,
         typename ThreadBlockShape7 = cutlass::gemm::GemmShape<1,1,1>, typename WarpShape7 = cutlass::gemm::GemmShape<1,1,1>, uint StageCount7 = 3,
         typename RWMTypes7 = RWMTypes<>, typename RWCTypes7 = RWCTypes<>, typename AllPresums7 = AllPresums<>>
using StrassenLevel1SingleMiGroup = StrassenLevel1Groups<StrassenLevel1M0Group<level, level_1_idx, ThreadBlockShape1, WarpShape1, StageCount1, RWMTypes1, RWCTypes1, AllPresums1>,
                                                         StrassenLevel1M1Group<level, level_1_idx, ThreadBlockShape2, WarpShape2, StageCount2, RWMTypes2, RWCTypes2, AllPresums2>,
                                                         StrassenLevel1M2Group<level, level_1_idx, ThreadBlockShape3, WarpShape3, StageCount3, RWMTypes3, RWCTypes3, AllPresums3>,
                                                         StrassenLevel1M3Group<level, level_1_idx, ThreadBlockShape4, WarpShape4, StageCount4, RWMTypes4, RWCTypes4, AllPresums4>,
                                                         StrassenLevel1M4Group<level, level_1_idx, ThreadBlockShape5, WarpShape5, StageCount5, RWMTypes5, RWCTypes5, AllPresums5>,
                                                         StrassenLevel1M5Group<level, level_1_idx, ThreadBlockShape6, WarpShape6, StageCount6, RWMTypes6, RWCTypes6, AllPresums6>,
                                                         StrassenLevel1M6Group<level, level_1_idx, ThreadBlockShape7, WarpShape7, StageCount7, RWMTypes7, RWCTypes7, AllPresums7>>;


template<int kMi, typename RWMTypes, typename RWCTypes>
class StrassenMi {
public:
  static const int Mi = kMi;

  CUTLASS_HOST_DEVICE
  static constexpr ReadWriteM GetRWForMi(int Mi) {
    return RWMTypes::GetRWForMi(Mi);
  }
};

// template<size_t N, int... Elements>
// class BitSet {
//   BitSet() {}

//   public:
//     CUTLASS_HOST_DEVICE
//     static constexpr bool hasElement(int i) {
//       return ((Elements == i) || ...);  
//     }
// };

template<size_t N, int... Mis>
class FusedMiGroup{
//TODO: Do checking for Fused Groups:
//  1. Tile Sizes are same
//  2. Register sizes are same.
//  3. Number of Mis are same.
public:
  CUTLASS_HOST_DEVICE
  FusedMiGroup() {}

  CUTLASS_HOST_DEVICE
  static constexpr bool HasGroup(int i) { //TODO: Change to has MiGroup
    return ((Mis == i) || ...);
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasAnyGroup() {
    for (int i = 0; i < N; i++)
      if (HasGroup(i)) return true;
    return false;
  }

  template<typename GemmKernel0, typename GemmKernel1, typename GemmKernel2,
           typename GemmKernel3, typename GemmKernel4, typename GemmKernel5,
           typename GemmKernel6>
  CUTLASS_HOST_DEVICE
  static constexpr bool IsAnyGroupValid() {
    bool valid0 = (HasGroup(0) && GemmKernel0::StrassenMiGroup::hasAnyM());
    bool valid1 = (HasGroup(1) && GemmKernel1::StrassenMiGroup::hasAnyM());
    bool valid2 = (HasGroup(2) && GemmKernel2::StrassenMiGroup::hasAnyM());
    bool valid3 = (HasGroup(3) && GemmKernel3::StrassenMiGroup::hasAnyM());
    bool valid4 = (HasGroup(4) && GemmKernel4::StrassenMiGroup::hasAnyM());
    bool valid5 = (HasGroup(5) && GemmKernel5::StrassenMiGroup::hasAnyM());
    bool valid6 = (HasGroup(6) && GemmKernel6::StrassenMiGroup::hasAnyM());
    return valid0 || valid1 || valid2 || valid3 || valid4 || valid5 || valid6;
  }

  template<typename ParallelMiKernels,
           typename GemmKernel0, typename GemmKernel1, typename GemmKernel2,
           typename GemmKernel3, typename GemmKernel4, typename GemmKernel5,
           typename GemmKernel6>
  CUTLASS_DEVICE
  static bool run(const ParallelMiKernels& parallel_kernels, int* SharedStorage,
                  int start_block_z, int end_block_z) {
    char accumStore[ParallelMiKernels::AccumSize()];
    if (!(start_block_z <= blockIdx.z and blockIdx.z < end_block_z)) return false;
    dim3 base_block = dim3{0, 0, (uint)start_block_z};

    if (HasGroup(0)) {
      typename GemmKernel0::SharedStorage* shared_storage_0 =
          reinterpret_cast<typename GemmKernel0::SharedStorage *>(SharedStorage);
      GemmKernel0 op;
      // typename GemmKernel0::Params params0(common_params, params0);
      op(parallel_kernels.params.params0(), *shared_storage_0, accumStore, base_block);
      // __syncthreads();
    }
    if (HasGroup(1)) {
      typename GemmKernel1::SharedStorage *shared_storage_1 =
          reinterpret_cast<typename GemmKernel1::SharedStorage *>(SharedStorage);
      GemmKernel1 op;
      //FIXME: using params0 to initialize params1 leads to some stack size
      // typename GemmKernel1::Params params1(parallel_kernels.params0);
      // auto params1 = parallel_kernels.params1;
      op(parallel_kernels.params.params1(), *shared_storage_1, accumStore, base_block);
      // __syncthreads();
    }
    if (HasGroup(2)) {
      typename GemmKernel2::SharedStorage *shared_storage_2 =
          reinterpret_cast<typename GemmKernel2::SharedStorage *>(SharedStorage);
      GemmKernel2 op;
      // typename GemmKernel2::Params params2(parallel_kernels.params0);
      // auto params2 = parallel_kernels.params2;
      op(parallel_kernels.params.params2(), *shared_storage_2, accumStore, base_block);
      // __syncthreads();
    }
    if (HasGroup(3)) {
      typename GemmKernel3::SharedStorage *shared_storage_3 =
          reinterpret_cast<typename GemmKernel3::SharedStorage *>(SharedStorage);
      GemmKernel3 op;
      // typename GemmKernel3::Params params3(parallel_kernels.params0);
      // auto params3 = parallel_kernels.params3;
      op(parallel_kernels.params.params3(), *shared_storage_3, accumStore, base_block);
      // __syncthreads();
    }
    if (HasGroup(4)) {
      typename GemmKernel4::SharedStorage *shared_storage_4 =
          reinterpret_cast<typename GemmKernel4::SharedStorage *>(SharedStorage);
      GemmKernel4 op;
      // typename GemmKernel4::Params params4(parallel_kernels.params0);
      // auto params4 = parallel_kernels.params4;
      op(parallel_kernels.params.params4(), *shared_storage_4, accumStore, base_block);
      // __syncthreads();
    }
    if (HasGroup(5)) {
      typename GemmKernel5::SharedStorage *shared_storage_5 =
          reinterpret_cast<typename GemmKernel5::SharedStorage *>(SharedStorage);
      GemmKernel5 op;
      // typename GemmKernel5::Params params5(parallel_kernels.params0);
      // auto params5 = parallel_kernels.params5;
      op(parallel_kernels.params.params5(), *shared_storage_5, accumStore, base_block);
      // __syncthreads();
    }
    if (HasGroup(6)) {
      typename GemmKernel6::SharedStorage *shared_storage_6 =
          reinterpret_cast<typename GemmKernel6::SharedStorage *>(SharedStorage);
      GemmKernel6 op;
      // typename GemmKernel6::Params params6(parallel_kernels.params0);
      // auto params6 = parallel_kernels.params6;
      op(parallel_kernels.params.params6(), *shared_storage_6, accumStore, base_block);
      // __syncthreads();
    }

    //Return false so that fold expression in ParallelMiGroups::run 
    //do not shortcircuit. 
    return false;
  }
};

template<bool kCommonParams = false, typename... kFusedMiGroups>
class ParallelMiGroups {
public:
  static const bool CommonParams = kCommonParams;

  CUTLASS_HOST_DEVICE
  ParallelMiGroups() {}

  template<typename GemmKernel0, typename GemmKernel1, typename GemmKernel2,
           typename GemmKernel3, typename GemmKernel4, typename GemmKernel5,
           typename GemmKernel6>
  CUTLASS_HOST_DEVICE
  static constexpr bool IsAnyGroupValid() {
    bool r = (kFusedMiGroups::template
                  IsAnyGroupValid<GemmKernel0, GemmKernel1, GemmKernel2,
                                  GemmKernel3, GemmKernel4, GemmKernel5,
                                  GemmKernel6>() || ...);
    return r;
  }

  CUTLASS_HOST_DEVICE
  static constexpr uint NumKernels() {
    return sizeof...(kFusedMiGroups);
  }

  CUTLASS_HOST_DEVICE
  static constexpr bool HasGroup(int i) {
    bool found = ((kFusedMiGroups::HasGroup(i) ? true : false) || ...);
    return found;
  }

  template<typename ParallelMiKernels,
           typename GemmKernel0, typename GemmKernel1, typename GemmKernel2,
           typename GemmKernel3, typename GemmKernel4, typename GemmKernel5,
           typename GemmKernel6>
  CUTLASS_DEVICE
  static void run(ParallelMiKernels const& parallel_kernels, int* SharedStorage, dim3 orig_grid) {
    int n = -1;
#if CUTLASS_API_v3
    orig_grid.z = 1;
#endif
    bool _ = ((n += 1,
              kFusedMiGroups::template
              run<ParallelMiKernels,
                  GemmKernel0, GemmKernel1, GemmKernel2,
                  GemmKernel3, GemmKernel4, GemmKernel5,
                  GemmKernel6>(parallel_kernels, SharedStorage,
                               n*orig_grid.z, (n+1)*orig_grid.z))
              || ...);
    CUTLASS_UNUSED(orig_grid);
  }
};

// Helper to conditionally store a kernel param by value if the corresponding
// Mi group is active in the ParallelGroup, or as an empty struct if inactive.
// This reduces the kernel parameter size (and thus register spills) by
// excluding unused params from the ParallelMiKernels struct.
template<bool Active, typename Params>
struct FilteredParam {
    Params value_;
    CUTLASS_HOST_DEVICE
    FilteredParam() = default;
    CUTLASS_HOST_DEVICE
    FilteredParam(Params const& p) : value_(p) {}
    CUTLASS_HOST_DEVICE
    Params const& get() const { return value_; }
};

template<typename Params>
struct FilteredParam<false, Params> {
    CUTLASS_HOST_DEVICE
    FilteredParam() = default;
    CUTLASS_HOST_DEVICE
    FilteredParam(Params const&) {} // ignore the param
    CUTLASS_HOST_DEVICE
    Params const& get() const {
        // Dead code path - only reached in compile-time-eliminated branches
        return *reinterpret_cast<Params const*>(this);
    }
};

template<bool kCommonParams,
         typename kParallelGroup,
         typename GemmKernel0,
         typename GemmKernel1,
         typename GemmKernel2,
         typename GemmKernel3,
         typename GemmKernel4,
         typename GemmKernel5,
         typename GemmKernel6
         >
class ParallelGroupParams;

template<
         typename kParallelGroup,
         typename GemmKernel0,
         typename GemmKernel1,
         typename GemmKernel2,
         typename GemmKernel3,
         typename GemmKernel4,
         typename GemmKernel5,
         typename GemmKernel6
         >
class ParallelGroupParams<false, kParallelGroup, GemmKernel0, GemmKernel1, GemmKernel2,
                                 GemmKernel3, GemmKernel4, GemmKernel5, GemmKernel6> {
public:
  using Params0 = typename GemmKernel0::Params;
  using Params1 = typename GemmKernel1::Params;
  using Params2 = typename GemmKernel2::Params;
  using Params3 = typename GemmKernel3::Params;
  using Params4 = typename GemmKernel4::Params;
  using Params5 = typename GemmKernel5::Params;
  using Params6 = typename GemmKernel6::Params;

private:
  FilteredParam<kParallelGroup::HasGroup(0), Params0> params0_;
  FilteredParam<kParallelGroup::HasGroup(1), Params1> params1_;
  FilteredParam<kParallelGroup::HasGroup(2), Params2> params2_;
  FilteredParam<kParallelGroup::HasGroup(3), Params3> params3_;
  FilteredParam<kParallelGroup::HasGroup(4), Params4> params4_;
  FilteredParam<kParallelGroup::HasGroup(5), Params5> params5_;
  FilteredParam<kParallelGroup::HasGroup(6), Params6> params6_;

public:
  ParallelGroupParams(Params0 const & params0, Params1 const & params1, Params2 const & params2,
                      Params3 const & params3, Params4 const & params4, Params5 const & params5,
                      Params6 const & params6) :
                      params0_(params0), params1_(params1), params2_(params2),
                      params3_(params3), params4_(params4), params5_(params5),
                      params6_(params6)
  {}

  CUTLASS_HOST_DEVICE
  Params0 const& params0() const {
    return params0_.get();
  }

  CUTLASS_HOST_DEVICE
  Params1 const& params1() const {
    return params1_.get();
  }

  CUTLASS_HOST_DEVICE
  Params2 const& params2() const {
    return params2_.get();
  }

  CUTLASS_HOST_DEVICE
  Params3 const& params3() const {
    return params3_.get();
  }

  CUTLASS_HOST_DEVICE
  Params4 const& params4() const {
    return params4_.get();
  }

  CUTLASS_HOST_DEVICE
  Params5 const& params5() const {
    return params5_.get();
  }

  CUTLASS_HOST_DEVICE
  Params6 const& params6() const {
    return params6_.get();
  }
};

template<typename kParallelGroup,
         typename GemmKernel0,
         typename GemmKernel1,
         typename GemmKernel2,
         typename GemmKernel3,
         typename GemmKernel4,
         typename GemmKernel5,
         typename GemmKernel6
         >
class ParallelGroupParams<true, kParallelGroup, GemmKernel0, GemmKernel1, GemmKernel2,
                                GemmKernel3, GemmKernel4, GemmKernel5, GemmKernel6> {
public:
  using Params0 = typename GemmKernel0::Params;
  using Params1 = typename GemmKernel1::Params;
  using Params2 = typename GemmKernel2::Params;
  using Params3 = typename GemmKernel3::Params;
  using Params4 = typename GemmKernel4::Params;
  using Params5 = typename GemmKernel5::Params;
  using Params6 = typename GemmKernel6::Params;

private:
  Params0 params0_;

public:
  ParallelGroupParams(const Params0& params0, const Params1& params1, const Params2& params2,
                      const Params3& params3, const Params4& params4, const Params5& params5,
                      const Params6& params6) :
                      params0_(params0)
  {}

  CUTLASS_HOST_DEVICE
  Params0 params0() const {
    return params0_;
  }

  CUTLASS_HOST_DEVICE
  Params1 params1() const {
    return Params1(params0());
  }

  CUTLASS_HOST_DEVICE
  Params2 params2() const {
    return Params2(params0());
  }

  CUTLASS_HOST_DEVICE
  Params3 params3() const {
    return Params3(params0());
  }

  CUTLASS_HOST_DEVICE
  Params4 params4() const {
    return Params4(params0());
  }

  CUTLASS_HOST_DEVICE
  Params5 params5() const {
    return Params5(params0());
  }

  CUTLASS_HOST_DEVICE
  Params6 params6() const {
    return Params6(params0());
  }
};

template<typename ParallelGroup,
         typename GemmKernel0,
         typename GemmKernel1,
         typename GemmKernel2,
         typename GemmKernel3,
         typename GemmKernel4,
         typename GemmKernel5,
         typename GemmKernel6  
        >
class ParallelMiKernels {
public:
  using GemmKernel0_ = GemmKernel0;
  using Params0 = typename GemmKernel0::Params;
  using Params1 = typename GemmKernel1::Params;
  using Params2 = typename GemmKernel2::Params;
  using Params3 = typename GemmKernel3::Params;
  using Params4 = typename GemmKernel4::Params;
  using Params5 = typename GemmKernel5::Params;
  using Params6 = typename GemmKernel6::Params;

  ParallelGroupParams<ParallelGroup::CommonParams, ParallelGroup, GemmKernel0, GemmKernel1, GemmKernel2,
                      GemmKernel3, GemmKernel4, GemmKernel5, GemmKernel6> params;

  ParallelMiKernels(const Params0& params0, const Params1& params1, const Params2& params2,
                    const Params3& params3, const Params4& params4, const Params5& params5,
                    const Params6& params6) :
                    params(params0, params1, params2, params3, params4, params5, params6)
  {}

  cutlass::gemm::GemmCoord grid_single_tiled_shape() {
    //TODO: convert this to loop?
    //TODO: when grid_tiled_shape is different then?
    if (ParallelGroup::HasGroup(0))
      return params.params0().grid_tiled_shape;
    if (ParallelGroup::HasGroup(1))
      return params.params1().grid_tiled_shape;
    if (ParallelGroup::HasGroup(2))
      return params.params2().grid_tiled_shape;
    if (ParallelGroup::HasGroup(3))
      return params.params3().grid_tiled_shape;
    if (ParallelGroup::HasGroup(4))
      return params.params4().grid_tiled_shape;
    if (ParallelGroup::HasGroup(5))
      return params.params5().grid_tiled_shape;
    if (ParallelGroup::HasGroup(6))
      return params.params6().grid_tiled_shape;
    return params.params0().grid_tiled_shape;
  }

  static constexpr uint ThreadCount() {
    if (ParallelGroup::HasGroup(0))
      return GemmKernel0::kThreadCount;
    if (ParallelGroup::HasGroup(1))
      return GemmKernel1::kThreadCount;
    if (ParallelGroup::HasGroup(2))
      return GemmKernel2::kThreadCount;
    if (ParallelGroup::HasGroup(3))
      return GemmKernel3::kThreadCount;
    if (ParallelGroup::HasGroup(4))
      return GemmKernel4::kThreadCount;
    if (ParallelGroup::HasGroup(5))
      return GemmKernel5::kThreadCount;
    if (ParallelGroup::HasGroup(6))
      return GemmKernel6::kThreadCount;
    return 256;
  }

  static constexpr bool HasAKernel() {
    return ParallelGroup::template
              IsAnyGroupValid<GemmKernel0, GemmKernel1, GemmKernel2,
                              GemmKernel3, GemmKernel4, GemmKernel5,
                              GemmKernel6>() && ParallelGroup::NumKernels() > 0;
  }

  static constexpr size_t SharedStorageSize() {
    size_t max_size = 0;
    if (ParallelGroup::HasGroup(0))
      max_size = std::max(max_size, sizeof(typename GemmKernel0::SharedStorage));
    if (ParallelGroup::HasGroup(1))
      max_size = std::max(max_size, sizeof(typename GemmKernel1::SharedStorage));
    if (ParallelGroup::HasGroup(2))
      max_size = std::max(max_size, sizeof(typename GemmKernel2::SharedStorage));
    if (ParallelGroup::HasGroup(3))
      max_size = std::max(max_size, sizeof(typename GemmKernel3::SharedStorage));
    if (ParallelGroup::HasGroup(4))
      max_size = std::max(max_size, sizeof(typename GemmKernel4::SharedStorage));
    if (ParallelGroup::HasGroup(5))
      max_size = std::max(max_size, sizeof(typename GemmKernel5::SharedStorage));
    if (ParallelGroup::HasGroup(6))
      max_size = std::max(max_size, sizeof(typename GemmKernel6::SharedStorage));
    return std::max(max_size, 1024UL);
  }

  CUTLASS_HOST_DEVICE
  static constexpr size_t AccumSize() {
    if (ParallelGroup::HasGroup(0))
      return sizeof(typename GemmKernel0::Mma::FragmentC);
    if (ParallelGroup::HasGroup(1))
      return sizeof(typename GemmKernel1::Mma::FragmentC);
    if (ParallelGroup::HasGroup(2))
      return sizeof(typename GemmKernel2::Mma::FragmentC);
    if (ParallelGroup::HasGroup(3))
      return sizeof(typename GemmKernel3::Mma::FragmentC);
    if (ParallelGroup::HasGroup(4))
      return sizeof(typename GemmKernel4::Mma::FragmentC);
    if (ParallelGroup::HasGroup(5))
      return sizeof(typename GemmKernel5::Mma::FragmentC);
    if (ParallelGroup::HasGroup(6))
      return sizeof(typename GemmKernel6::Mma::FragmentC);
    return sizeof(typename GemmKernel0::Mma::FragmentC);
  }

  CUTLASS_HOST_DEVICE
  static constexpr uint NumKernels() {
    return ParallelGroup::NumKernels();
  }

  CUTLASS_DEVICE
  void operator()(int* SharedStorage, dim3 orig_grid) const {
    ParallelGroup::template
              run<ParallelMiKernels,
                  GemmKernel0, GemmKernel1, GemmKernel2,
                  GemmKernel3, GemmKernel4, GemmKernel5,
                  GemmKernel6> (*this, SharedStorage, orig_grid);
  }
};

template<typename kParallelGroups0 = ParallelMiGroups<>,
         typename kParallelGroups1 = ParallelMiGroups<>,
         typename kParallelGroups2 = ParallelMiGroups<>,
         typename kParallelGroups3 = ParallelMiGroups<>,
         typename kParallelGroups4 = ParallelMiGroups<>,
         typename kParallelGroups5 = ParallelMiGroups<>,
         typename kParallelGroups6 = ParallelMiGroups<>>
class ScheduleStrassenGroups {
public:
  using ParallelGroups0 = kParallelGroups0;
  using ParallelGroups1 = kParallelGroups1;
  using ParallelGroups2 = kParallelGroups2;
  using ParallelGroups3 = kParallelGroups3;
  using ParallelGroups4 = kParallelGroups4;
  using ParallelGroups5 = kParallelGroups5;
  using ParallelGroups6 = kParallelGroups6;

  CUTLASS_HOST_DEVICE
  ScheduleStrassenGroups() {}
};


constexpr uint64_t SetReadWriteM(const ReadWriteM val, const MmaStrassen::Type mi,
                                 bool neg, bool global_sync = false) {
  uint bits = (mi - GlobalPreSumLevel1_M0)*ReadWriteM::RWBitsAugmented;
  uint64_t augmented = ((neg ? 1UL : 0UL) << 1) | (global_sync ? 1UL : 0UL);
  return ((augmented << ReadWriteM::RWBits) | val) << bits;
}

CUTLASS_DEVICE
constexpr uint GetRWBit(uint64_t x, const MmaStrassen::Type mi) {
  uint64_t rwwithmul = x >> ((mi - GlobalPreSumLevel1_M0)*ReadWriteM::RWBitsAugmented);
  return rwwithmul & ((1U << ReadWriteM::RWBits)-1);
}

CUTLASS_DEVICE
constexpr bool GetRWMulSign(uint64_t x, const MmaStrassen::Type mi) {
  uint64_t rwwithmul = x >> ((mi - GlobalPreSumLevel1_M0)*ReadWriteM::RWBitsAugmented);
  return ((rwwithmul >> (ReadWriteM::RWBits+1)) & 0x1) == 0x1;
}

CUTLASS_DEVICE
constexpr bool GetRWGlobalSync(uint64_t x, const MmaStrassen::Type mi) {
  uint64_t rwwithmul = x >> ((mi - GlobalPreSumLevel1_M0)*ReadWriteM::RWBitsAugmented);
  return ((rwwithmul >> (ReadWriteM::RWBits)) & 0x1) == 0x1;
}

constexpr uint64_t SetRWOutputC(uint64_t c) {
  return (c + 1UL) << ((GlobalPreSumLevel1_M6 - GlobalPreSumLevel1_M0 + 1)*ReadWriteM::RWBitsAugmented);
}

CUTLASS_DEVICE
constexpr uint HasRWOutputC(uint64_t rw) {
  uint64_t c = rw >> ((GlobalPreSumLevel1_M6 - GlobalPreSumLevel1_M0 + 1)*ReadWriteM::RWBitsAugmented);
  return (c & ((1UL << RWOutputCBits) - 1)) > 0;
}

CUTLASS_DEVICE
constexpr uint GetRWOutputC(uint64_t rw) {
  uint64_t c = rw >> ((GlobalPreSumLevel1_M6 - GlobalPreSumLevel1_M0 + 1)*ReadWriteM::RWBitsAugmented);
  return (c & ((1UL << RWOutputCBits) - 1)) - 1;
}

// template<uint8_t M0_0, uint8_t M0_1, uint8_t M1_0, uint8_t M1_1, uint8_t M2_0, uint8_t M2_1, uint8_t M3_0, uint8_t M3_1,
//          uint8_t M4_0, uint8_t M4_1, uint8_t M5_0, uint8_t M5_1, uint8_t M6_0, uint8_t M6_1>
// struct ScheduleM {
//   uint8_t m0_0 : 4 = M0_0;
//   uint8_t m0_1 : 4 = M0_1;
//   uint8_t m1_0 : 4 = M1_0;
//   uint8_t m1_1 : 4 = M1_1;
//   uint8_t m2_0 : 4 = M2_0;
//   uint8_t m2_1 : 4 = M2_1;
//   uint8_t m3_0 : 4 = M3_0;
//   uint8_t m3_1 : 4 = M3_1;
//   uint8_t m4_0 : 4 = M4_0;
//   uint8_t m4_1 : 4 = M4_1;
//   uint8_t m5_0 : 4 = M5_0;
//   uint8_t m5_1 : 4 = M5_1;
//   uint8_t m6_0 : 4 = M6_0;
//   uint8_t m6_1 : 4 = M6_1;


//   // constexpr ScheduleM(uint8_t m0_0, uint8_t m0_1, uint8_t m1_0, uint8_t m1_1, uint8_t m2_0, uint8_t m2_1, uint8_t m3_0, uint8_t m3_1,
//   //           uint8_t m4_0, uint8_t m4_1, uint8_t m5_0, uint8_t m5_1, uint8_t m6_0, uint8_t m6_1) :
//   //   sched{m0_0, m0_1, m1_0, m1_1, m2_0, m2_1, m3_0, m3_1, m4_0, m4_1, m5_0, m5_1, m6_0, m6_1} {}
//   constexpr uint64_t bytes() const {
//     return std::bit_cast<uint64_t>(*this);
//   }

//   static ScheduleM fromBytes(uint64_t bytes) {
//     return *reinterpret_cast<ScheduleM*>(&bytes);
//   }
// };

// CUTLASS_HOST_DEVICE
// constexpr ScheduleM Schedule230F15FF46F() {
//   // uint MBits = 3;
//   // uint64_t F = (1L << MBits) - 1;
//   // return 2L | (3L << MBits) |              (0L << (2*MBits)) |
//   //        ((F | 1L << MBits | 5L << (2*MBits) | F << (3*MBits)) << ((2+4)*MBits)) |
//   //        ((F | 4L << MBits | 6L << (2*MBits)) << ((2+4+4)*MBits));
//   return ScheduleM(2,0,3,0,0,0,1,0,4,0,5,1,6,1);//ScheduleM{2,0,3,0,1,0,4,0,3,1,4,1};
// }

CUTLASS_HOST_DEVICE
constexpr uint GetSchedule1(uint64_t x, uint i) {
  uint bits = 4;
  return (x >> (i*2*bits)) & ((1<<bits) - 1);
}

CUTLASS_HOST_DEVICE
constexpr uint GetSchedule2(uint64_t x, uint i) {
  uint bits = 4;
  uint v = (x >> (i*2*bits));
  v = v >> bits;
  return v & ((1<<bits) - 1);
}

CUTLASS_HOST_DEVICE
constexpr uint SetBits(uint64_t v, uint i, uint bits) {
  return (v << (i*bits));
}

CUTLASS_HOST_DEVICE
constexpr uint64_t SetSchedule(uint m, uint64_t sched1, uint64_t sched2) {
  uint bits = 4;
  return (sched1 | (sched2 << bits)) << (m*2*bits);
}

CUTLASS_HOST_DEVICE
constexpr uint64_t Schedule230F15FF46F() {
  // uint MBits = 3;
  // uint64_t F = (1L << MBits) - 1;
  // return 2L | (3L << MBits) |              (0L << (2*MBits)) |
  //        ((F | 1L << MBits | 5L << (2*MBits) | F << (3*MBits)) << ((2+4)*MBits)) |
  //        ((F | 4L << MBits | 6L << (2*MBits)) << ((2+4+4)*MBits));
  return SetSchedule(0,0,0) | SetSchedule(1,1,0) | SetSchedule(2,2,0) | SetSchedule(3,3,0) |
         SetSchedule(4,4,0) | SetSchedule(5,5,0) | SetSchedule(6,6,0);
}

template<int Gamma1_, int Gamma2_, int Delta1_, int Delta2_>
class Consts {
  static const int Gamma1 = Gamma1_;
  static const int Gamma2 = Gamma2_;
  static const int Delta1 = Delta1_;
  static const int Delta2 = Delta2_;
};
}