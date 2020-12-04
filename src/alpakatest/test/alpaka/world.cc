#include <iostream>

#include "AlpakaCore/alpakaConfig.h"

namespace {
  struct Print {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc) const {
      uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const elemDimension(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
      printf("Alpaka kernel thread index %u, number of elements %u\n", blockThreadIdx, elemDimension);
    }
  };
}  // namespace

int main() {
  std::cout << "World" << std::endl;

  using namespace ALPAKA_ACCELERATOR_NAMESPACE;
  const DevAcc device(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
  Queue queue(device);

  Vec1 elementsPerThread(Vec1::all(1));
  Vec1 threadsPerBlock(Vec1::all(4));
  Vec1 blocksPerGrid(Vec1::all(1));
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || \
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
  // on the GPU, run with 32 threads in parallel per block, each looking at a single element
  // on the CPU, run serially with a single thread per block, over 32 elements
  std::swap(threadsPerBlock, elementsPerThread);
#endif
  const WorkDiv1 workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<Acc1>(workDiv, Print()));
  alpaka::wait::wait(queue);
  return 0;
}
