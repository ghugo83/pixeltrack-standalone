#ifndef ALPAKAWORKDIVHELPER_H
#define ALPAKAWORKDIVHELPER_H

#include "AlpakaCore/alpakaConfig.h"

using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    /*
     * Creates the accelerator-dependent workdiv.
     */
    template <typename TDim>
    WorkDiv<TDim> make_workdiv(const Vec<TDim>& blocksPerGrid, const Vec<TDim>& threadsPerBlockOrElementsPerThread) {
      // On the GPU:
      // threadsPerBlockOrElementsPerThread is the number of threads per block.
      // Each thread is looking at a single element: elementsPerThread is always 1.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      const Vec<TDim>& elementsPerThread = Vec<TDim>::ones();
      return WorkDiv<TDim>(blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
#else
      // On the CPU:
      // Run serially with a single thread per block: threadsPerBlock is always 1.
      // threadsPerBlockOrElementsPerThread is the number of elements per thread.
      const Vec<TDim>& threadsPerBlock = Vec<TDim>::ones();
      return WorkDiv<TDim>(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
#endif
    }

    /*
     * 1D helper to only access 1 element per block 
     * (should obviously only be needed for debug / printout).
     */
    template <typename TAcc>
    ALPAKA_FN_ACC bool once_per_block_1D(const TAcc& acc, Idx i) {
      const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      return (i % blockDimension == 0);
    }

    /*
     * Computes the range of the elements indexes, local to the block.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename TAcc>
    ALPAKA_FN_ACC std::pair<Vec1, Vec1> element_index_range_in_block(const TAcc& acc,
                                                                     const Vec1& elementIdxShift,
                                                                     const Idx dimIndex = 0u) {
      Vec1 firstElementIdxVec = Vec1::zeros();
      Vec1 endElementIdxUncutVec = Vec1::zeros();

      // Take into account the thread index in block.
      const Idx threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndex]);
      const Idx threadDimension(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[dimIndex]);

      // Compute the elements indexes in block.
      // Obviously relevant for CPU only.
      // For GPU, threadDimension = 1, and elementIdx = firstElementIdx = threadIdx + elementIdxShift.
      const Idx firstElementIdxLocal = threadIdxLocal * threadDimension;
      const Idx firstElementIdx = firstElementIdxLocal + elementIdxShift[0u];  // Add the shift!
      const Idx endElementIdxUncut = firstElementIdx + threadDimension;

      firstElementIdxVec[0u] = firstElementIdx;
      endElementIdxUncutVec[0u] = endElementIdxUncut;

      // Return element indexes, shifted by elementIdxShift.
      return {firstElementIdxVec, endElementIdxUncutVec};
    }

    /*
     * Computes the range of the elements indexes, local to the block.
     * Truncated by the max number of elements of interest.
     */
    template <typename TAcc>
    ALPAKA_FN_ACC std::pair<Vec1, Vec1> element_index_range_in_block_truncated(const TAcc& acc,
                                                                               const Vec1& maxNumberOfElements,
                                                                               const Vec1& elementIdxShift,
                                                                               const Idx dimIndex = 0u) {
      // Check dimension
      //static_assert(alpaka::Dim<TAcc>::value == Dim1::value,
      //              "Accelerator and maxNumberOfElements need to have same dimension.");
      auto [firstElementIdxLocalVec, endElementIdxLocalVec] =
          element_index_range_in_block(acc, elementIdxShift, dimIndex);

      // Truncate
      endElementIdxLocalVec[0u] = std::min(endElementIdxLocalVec[0u], maxNumberOfElements[0u]);

      // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
      return {firstElementIdxLocalVec, endElementIdxLocalVec};
    }

    /*
     * Computes the range of the elements indexes in grid.
     * Warning: the max index is not truncated by the max number of elements of interest.
     */
    template <typename TAcc>
    ALPAKA_FN_ACC std::pair<Vec1, Vec1> element_index_range_in_grid(const TAcc& acc,
                                                                    Vec1& elementIdxShift,
                                                                    const Idx dimIndex = 0u) {
      // Take into account the block index in grid.
      const Idx blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
      const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

      // Shift to get global indices in grid (instead of local to the block)
      elementIdxShift[0u] += blockIdxInGrid * blockDimension;

      // Return element indexes, shifted by elementIdxShift.
      return element_index_range_in_block(acc, elementIdxShift, dimIndex);
    }

    /*
     * Computes the range of the elements indexes in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename TAcc>
    ALPAKA_FN_ACC std::pair<Vec1, Vec1> element_index_range_in_grid_truncated(const TAcc& acc,
                                                                              const Vec1& maxNumberOfElements,
                                                                              Vec1& elementIdxShift,
                                                                              const Idx dimIndex = 0u) {
      // Check dimension
      //static_assert(dimIndex <= alpaka::Dim<TAcc>::value,
      //"Accelerator and maxNumberOfElements need to have same dimension.");
      auto [firstElementIdxGlobalVec, endElementIdxGlobalVec] =
          element_index_range_in_grid(acc, elementIdxShift, dimIndex);

      // Truncate
      endElementIdxGlobalVec[0u] = std::min(endElementIdxGlobalVec[0u], maxNumberOfElements[0u]);

      // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
      return {firstElementIdxGlobalVec, endElementIdxGlobalVec};
    }

    /*
     * Computes the range of the element(s) index(es) in grid.
     * Truncated by the max number of elements of interest.
     */
    template <typename TAcc>
    ALPAKA_FN_ACC std::pair<Vec1, Vec1> element_index_range_in_grid_truncated(const TAcc& acc,
                                                                              const Vec1& maxNumberOfElements,
                                                                              const Idx dimIndex = 0u) {
      Vec1 elementIdxShift = Vec1::zeros();
      return element_index_range_in_grid_truncated(acc, maxNumberOfElements, elementIdxShift, dimIndex);
    }

    /*********************************************
     *     1D HELPERS, LOOP ON ALL CPU ELEMENTS
     ********************************************/

    /*
     * Loop on all (CPU) elements.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Indexes are local to the BLOCK.
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_block(const TAcc& acc,
                                                                    const Idx maxNumberOfElements,
                                                                    const Idx elementIdxShift,
                                                                    const Func func,
                                                                    const Idx dimIndex = 0) {
      const auto& [firstElementIdx, endElementIdx] = cms::alpakatools::element_index_range_in_block_truncated(
          acc, Vec1::all(maxNumberOfElements), Vec1::all(elementIdxShift), dimIndex);

      for (Idx elementIdx = firstElementIdx[0u]; elementIdx < endElementIdx[0u]; ++elementIdx) {
        func(elementIdx);
      }
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_block(const TAcc& acc,
                                                                    const Idx maxNumberOfElements,
                                                                    const Func func,
                                                                    const Idx dimIndex = 0) {
      const Idx elementIdxShift = 0;
      cms::alpakatools::for_each_element_in_thread_1D_index_in_block(
          acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
    }

    /*
     * Loop on all (CPU) elements.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Indexes are expressed in GRID 'frame-of-reference'.
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_grid(const TAcc& acc,
                                                                   const Idx maxNumberOfElements,
                                                                   Idx elementIdxShift,
                                                                   const Func func,
                                                                   const Idx dimIndex = 0) {
      // Take into account the block index in grid to compute the element indices.
      const Idx blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
      const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);
      elementIdxShift += blockIdxInGrid * blockDimension;

      for_each_element_in_thread_1D_index_in_block(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_in_thread_1D_index_in_grid(const TAcc& acc,
                                                                   const Idx maxNumberOfElements,
                                                                   const Func func,
                                                                   const Idx dimIndex = 0) {
      const Idx elementIdxShift = 0;
      cms::alpakatools::for_each_element_in_thread_1D_index_in_grid(
          acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
    }

    /******************************************************************************
     *     1D HELPERS, LOOP ON ALL CPU ELEMENTS, AND ELEMENT/THREAD STRIDED ACCESS
     ******************************************************************************/

    /*
     * (CPU) Loop on all elements + (CPU/GPU) Strided access.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Stride to full problem size, by BLOCK size.
     * Indexes are local to the BLOCK.
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_block_stride(const TAcc& acc,
                                                        const Idx maxNumberOfElements,
                                                        const Idx elementIdxShift,
                                                        const Func func,
                                                        const Idx dimIndex = 0) {
      // Get thread / element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_block(acc, Vec1::all(elementIdxShift), dimIndex);

      // Stride = block size.
      const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);

      // Strided access.
      for (Idx threadIdx = firstElementIdxNoStride[0u], endElementIdx = endElementIdxNoStride[0u];
           threadIdx < maxNumberOfElements;
           threadIdx += blockDimension, endElementIdx += blockDimension) {
        // (CPU) Loop on all elements.
        if (endElementIdx > maxNumberOfElements) {
          endElementIdx = maxNumberOfElements;
        }
        for (Idx i = threadIdx; i < endElementIdx; ++i) {
          func(i);
        }
      }
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_block_stride(const TAcc& acc,
                                                        const Idx maxNumberOfElements,
                                                        const Func func,
                                                        const Idx dimIndex = 0) {
      const Idx elementIdxShift = 0;
      cms::alpakatools::for_each_element_1D_block_stride(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
    }

    /*
     * (CPU) Loop on all elements + (CPU/GPU) Strided access.
     * Elements loop makes sense in CPU case only. In GPU case, elementIdx = firstElementIdx = threadIdx + shift.
     * Stride to full problem size, by GRID size.
     * Indexes are local to the GRID.
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Idx elementIdxShift,
                                                       const Func func,
                                                       const Idx dimIndex = 0) {
      Vec1 elementIdxShiftVec = Vec1::all(elementIdxShift);

      // Get thread / element indices in block.
      const auto& [firstElementIdxNoStride, endElementIdxNoStride] =
          cms::alpakatools::element_index_range_in_grid(acc, elementIdxShiftVec, dimIndex);

      // Stride = grid size.
      const Idx gridDimension(alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[dimIndex]);

      // Strided access.
      for (Idx threadIdx = firstElementIdxNoStride[0u], endElementIdx = endElementIdxNoStride[0u];
           threadIdx < maxNumberOfElements;
           threadIdx += gridDimension, endElementIdx += gridDimension) {
        // (CPU) Loop on all elements.
        if (endElementIdx > maxNumberOfElements) {
          endElementIdx = maxNumberOfElements;
        }
        for (Idx i = threadIdx; i < endElementIdx; ++i) {
          func(i);
        }
      }
    }

    /*
     * Overload for elementIdxShift = 0
     */
    template <typename TAcc, typename Func>
    ALPAKA_FN_ACC void for_each_element_1D_grid_stride(const TAcc& acc,
                                                       const Idx maxNumberOfElements,
                                                       const Func func,
                                                       const Idx dimIndex = 0) {
      const Idx elementIdxShift = 0;
      cms::alpakatools::for_each_element_1D_grid_stride(acc, maxNumberOfElements, elementIdxShift, func, dimIndex);
    }

    /*
     * Case where the input index has reached the end of threadDimension: strides the input index.
     * Otherwise: do nothing.
     */
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool get_next_element_1D_index_stride(Idx& i,
                                                                         Idx& firstElementIdx,
                                                                         Idx& endElementIdx,
                                                                         const Idx stride,
                                                                         const Idx maxNumberOfElements) {
      bool isNextStrideElementValid = true;
      if (i == endElementIdx) {
        firstElementIdx += stride;
        endElementIdx += stride;
        i = firstElementIdx;
        if (i >= maxNumberOfElements) {
          isNextStrideElementValid = false;
        }
      }
      return isNextStrideElementValid;
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAWORKDIVHELPER_H
