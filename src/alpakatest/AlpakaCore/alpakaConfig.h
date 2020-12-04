#ifndef alpakaConfig_h_
#define alpakaConfig_h_

#include <alpaka/alpaka.hpp>

namespace alpaka_common {
  using Dim1 = alpaka::dim::DimInt<1u>;
  using Dim2 = alpaka::dim::DimInt<2u>;
  using Dim3 = alpaka::dim::DimInt<3u>;
  using Idx = uint32_t;
  using Extent = uint32_t;
  using DevHost = alpaka::dev::DevCpu;
  using PltfHost = alpaka::pltf::Pltf<DevHost>;
  using WorkDiv1 = alpaka::workdiv::WorkDivMembers<Dim1, Idx>;
  using WorkDiv2 = alpaka::workdiv::WorkDivMembers<Dim2, Idx>;
  using WorkDiv3 = alpaka::workdiv::WorkDivMembers<Dim3, Idx>;
  using Vec1 = alpaka::vec::Vec<Dim1, Idx>;
  using Vec2 = alpaka::vec::Vec<Dim2, Idx>;
  using Vec3 = alpaka::vec::Vec<Dim3, Idx>;
}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
namespace alpaka_cuda_async {
  using namespace alpaka_common;
  using Acc1 = alpaka::acc::AccGpuCudaRt<Dim1, Extent>;
  using Acc2 = alpaka::acc::AccGpuCudaRt<Dim2, Extent>;
  using Acc3 = alpaka::acc::AccGpuCudaRt<Dim3, Extent>;
  using Acc = Acc3;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCudaRtNonBlocking;
}  // namespace alpaka_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cuda
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda_async
#endif  // ALPAKA_ACC_GPU_CUDA_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
namespace alpaka_serial_sync {
  using namespace alpaka_common;
  using Acc1 = alpaka::acc::AccCpuSerial<Dim1, Extent>;
  using Acc2 = alpaka::acc::AccCpuSerial<Dim2, Extent>;
  using Acc3 = alpaka::acc::AccCpuSerial<Dim3, Extent>;
  using Acc = Acc3;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCpuBlocking;
}  // namespace alpaka_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
namespace alpaka_tbb_async {
  using namespace alpaka_common;
  using Acc1 = alpaka::acc::AccCpuTbbBlocks<Dim1, Extent>;
  using Acc2 = alpaka::acc::AccCpuTbbBlocks<Dim2, Extent>;
  using Acc3 = alpaka::acc::AccCpuTbbBlocks<Dim3, Extent>;
  using Acc = Acc3;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCpuNonBlocking;
}  // namespace alpaka_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_tbb_async
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
namespace alpaka_omp2_async {
  using namespace alpaka_common;
  using Acc1 = alpaka::acc::AccCpuOmp2Blocks<Dim1, Extent>;
  using Acc2 = alpaka::acc::AccCpuOmp2Blocks<Dim2, Extent>;
  using Acc3 = alpaka::acc::AccCpuOmp2Blocks<Dim3, Extent>;
  using Acc = Acc3;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCpuNonBlocking;
}  // namespace alpaka_omp2_async

#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp2_async
#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ASYNC_BACKEND

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
#define ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
namespace alpaka_omp4_async {
  using namespace alpaka_common;
  using Acc1 = alpaka::acc::AccCpuOmp4<Dim1, Extent>;
  using Acc2 = alpaka::acc::AccCpuOmp4<Dim2, Extent>;
  using Acc3 = alpaka::acc::AccCpuOmp4<Dim3, Extent>;
  using Acc = Acc3;
  using DevAcc = alpaka::dev::Dev<Acc>;
  using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
  using Queue = alpaka::queue::QueueCpuNonBlocking;
}  // namespace alpaka_omp4_async

#endif  // ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND
#define ALPAKA_ARCHITECTURE_NAMESPACE alpaka_cpu
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_omp4_async
#endif  // ALPAKA_ACC_CPU_BT_OMP4_ASYNC_BACKEND

// trick to force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification inside DEFINE_FWK_MODULE
#define DEFINE_FWK_ALPAKA_MODULE2(name) DEFINE_FWK_MODULE(name)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(name) DEFINE_FWK_EVENTSETUP_MODULE(name)
#define DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(name) \
  DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)

#endif  // alpakaConfig_h_
