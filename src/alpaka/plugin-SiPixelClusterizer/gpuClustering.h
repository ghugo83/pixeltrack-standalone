#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <cstdint>
#include <cstdio>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "AlpakaCore/HistoContainer.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"
#include "Geometry/phase1PixelTopology.h"

#define GPU_DEBUG true

namespace gpuClustering {

#ifdef GPU_DEBUG
  ALPAKA_STATIC_ACC_MEM_GLOBAL uint32_t gMaxHit = 0;
#endif

  struct countModules {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				  uint16_t const* __restrict__ id,
				  uint32_t* __restrict__ moduleStart,
				  int32_t* __restrict__ clusterId,
				  const unsigned int numElements) const {
      cms::alpakatools::for_each_element_1D_grid_stride(acc, numElements, [&](uint32_t i) {
	  clusterId[i] = i;
	  if (InvId != id[i]) {
	    int j = i - 1;
	    while (j >= 0 and id[j] == InvId)
	      --j;
	    if (j < 0 or id[j] != id[i]) {
	      // boundary...
	      //auto loc = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, moduleStart, MaxNumModules);   
	      auto loc = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, moduleStart, 2000u);
	      //auto loc = alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &moduleStart[0], 1u);  // TO DO: does that work the same???????
	      //assert(moduleStart[0] < MaxNumModules);
	      //printf("loc = %u, i = %u\n", loc, i);
	      moduleStart[loc + 1] = i;
	    }
	  }

	});

    }
  };

  //  __launch_bounds__(256,4)
  struct findClus {
    template <typename T_Acc>
    ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				  uint16_t const* __restrict__ id,           // module id of each pixel
				  uint16_t const* __restrict__ x,            // local coordinates of each pixel
				  uint16_t const* __restrict__ y,            //
				  uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
				  uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
				  uint32_t* __restrict__ moduleId,           // output: module id of each module
				  int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
				  const unsigned int numElements) const {
      const uint32_t blockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      if (blockIdx >= moduleStart[0])
	return;
      /*if (blockIdx == 0) {
	printf("Start within findClus. \n");
	}*/
      
      auto firstPixel = moduleStart[1 + blockIdx];
      auto thisModuleId = id[firstPixel];
      assert(thisModuleId < MaxNumModules);

      const uint32_t threadIdxLocal(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
       
#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
	if (threadIdxLocal == 0)
	  printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx);
#endif

      // find the index of the first pixel not belonging to this module (or invalid)
      auto&& msize = alpaka::block::shared::st::allocVar<unsigned int, __COUNTER__>(acc);
      msize = numElements;
      alpaka::block::sync::syncBlockThreads(acc);

      bool hasStopped = false;
      bool hasPrinted = false;
      // skip threads not associated to an existing pixel
      cms::alpakatools::for_each_element_1D_block_stride(acc, numElements, firstPixel, [&](uint32_t& i) {
	  if (!hasStopped) {
	    if (hasStopped && blockIdx == 0 && threadIdxLocal == 0 && !hasPrinted) { printf("BREAK DIDNT WORK!!!!!"); hasPrinted = true; }
	    if (id[i] != InvId) {  // skip invalid pixels
	      if (id[i] != thisModuleId) {  // find the first pixel in a different module
		alpaka::atomic::atomicOp<alpaka::atomic::op::Min>(acc, &msize, i);
		//i = numElements;  // break
		hasStopped = true;
	      }
	    }
	  }
	});
      /*if (blockIdx == 0) {
	printf("msize = %u \n", msize);
	}*/

      //init hist  (ymax=416 < 512 : 9bits)
      constexpr uint32_t maxPixInModule = 4000;
      constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
      using Hist = cms::alpakatools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
      auto&& hist = alpaka::block::shared::st::allocVar<Hist, __COUNTER__>(acc);
      //auto&& ws = alpaka::block::shared::st::allocVar<(typename Hist::Counter)[32], __COUNTER__>(acc);                 // TO DO: how to deal with typename??
      auto&& ws = alpaka::block::shared::st::allocVar<Hist::Counter[32], __COUNTER__>(acc);                 // TO DO: how to deal with typename??

      cms::alpakatools::for_each_element_1D_block_stride(acc, Hist::totbins(), [&](uint32_t j) {
	  hist.off[j] = 0;
	});
      alpaka::block::sync::syncBlockThreads(acc);

      assert((msize == numElements) or ((msize < numElements) and (id[msize] != thisModuleId)));

      // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
      if (0 == threadIdxLocal) {
	if (msize - firstPixel > maxPixInModule) {
	  printf("too many pixels in module %d: %d > %d\n", thisModuleId, msize - firstPixel, maxPixInModule);
	  msize = maxPixInModule + firstPixel;
	}
      }

      alpaka::block::sync::syncBlockThreads(acc);
      assert(msize - firstPixel <= maxPixInModule);

#ifdef GPU_DEBUG
      auto&& totGood = alpaka::block::shared::st::allocVar<uint32_t, __COUNTER__>(acc);
      totGood = 0;
      alpaka::block::sync::syncBlockThreads(acc);
#endif

      // fill histo
      cms::alpakatools::for_each_element_1D_block_stride(acc, msize, firstPixel, [&](uint32_t i) {
	  if (id[i] != InvId) {  // skip invalid pixels
	    hist.count(acc, y[i]);
#ifdef GPU_DEBUG
	    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &totGood, 1u);
#endif
	  }
	});
      alpaka::block::sync::syncBlockThreads(acc);
      cms::alpakatools::for_each_element_in_thread_1D_index_in_block(acc, 32u, [&](uint32_t i) {
	  ws[i] = 0;  // used by prefix scan...
	});
      alpaka::block::sync::syncBlockThreads(acc);
      hist.finalize(acc, ws);
      alpaka::block::sync::syncBlockThreads(acc);
#ifdef GPU_DEBUG
      assert(hist.size() == totGood);
      if (thisModuleId % 100 == 1)
	if (threadIdxLocal == 0)
	  printf("histo size %d\n", hist.size());
#endif
      cms::alpakatools::for_each_element_1D_block_stride(acc, msize, firstPixel, [&](uint32_t i) {
	  if (id[i] != InvId) {  // skip invalid pixels
	    hist.fill(acc, y[i], i - firstPixel);
	  }
	});

      //#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED // DEBUGGGGG removed cuda-specific treatment
      // assume that we can cover the whole module with up to 16 blockDimension-wide iterations
      constexpr unsigned int maxiter = 16;  // DEBUGGGGG added unsigned
      //#else
      //auto maxiter = hist.size();
      //#endif
      // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
      constexpr int maxNeighbours = 10;
      const uint32_t blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      assert((hist.size() / blockDimension) <= maxiter);
      // nearest neighbour
      uint16_t nn[maxiter][maxNeighbours];
      uint8_t nnn[maxiter];  // number of nn
      for (uint32_t k = 0; k < maxiter; ++k) {
	nnn[k] = 0;
	// DEBUGGGGG added inner loop
	for (uint32_t l = 0; l < maxNeighbours; ++l) {
	  nn[k][l] = 0;
	}
	// DEBUGGGGG added inner loop
      }

      alpaka::block::sync::syncBlockThreads(acc);  // for hit filling!

#ifdef GPU_DEBUG
      // look for anomalous high occupancy
      auto&& n40 = alpaka::block::shared::st::allocVar<uint32_t, __COUNTER__>(acc);
      auto&& n60 = alpaka::block::shared::st::allocVar<uint32_t, __COUNTER__>(acc);
      n40 = n60 = 0;
      alpaka::block::sync::syncBlockThreads(acc);
      cms::alpakatools::for_each_element_1D_block_stride(acc, Hist::nbins(), [&](uint32_t j) {
	  if (hist.size(j) > 60)
	    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &n60, 1u);
	  if (hist.size(j) > 40)
	    alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, &n40, 1u);
	});
      alpaka::block::sync::syncBlockThreads(acc);
      if (0 == threadIdxLocal) {
	if (n60 > 0)
	  printf("columns with more than 60 px %d in %d\n", n60, thisModuleId);
	else if (n40 > 0)
	  printf("columns with more than 40 px %d in %d\n", n40, thisModuleId);
      }
      alpaka::block::sync::syncBlockThreads(acc);
#endif

      if (thisModuleId % 100 == 1 && threadIdxLocal == 0) {
	printf("n40 = %u \n", n40);
	printf("n60 = %u \n", n60);
      }

      // fill NN
      uint32_t k = 0u;
      cms::alpakatools::for_each_element_1D_block_stride(acc, hist.size(), [&](uint32_t j) {
	  k = j / blockDimension;
	  assert(k < maxiter);
	  auto p = hist.begin() + j;
	  auto i = *p + firstPixel;
	  assert(id[i] != InvId);
	  assert(id[i] == thisModuleId);  // same module
	  int be = Hist::bin(y[i] + 1);
	  auto e = hist.end(be);
	  ++p;
	  assert(0 == nnn[k]);  // DEBUGGGG removed
	  for (; p < e; ++p) {
	    auto m = (*p) + firstPixel;
	    assert(m != i);
	    assert(int(y[m]) - int(y[i]) >= 0);
	    assert(int(y[m]) - int(y[i]) <= 1);
	    if (std::abs(int(x[m]) - int(x[i])) <= 1) {
	      auto l = nnn[k]++;
	      if (thisModuleId % 100 == 1 && threadIdxLocal == 0) {
		printf("k = %u \n", k);
		printf("nnn[k] = %u \n", nnn[k]);
	      }
	      assert(l < maxNeighbours);
	      nn[k][l] = *p;
	      if (thisModuleId % 100 == 1 && threadIdxLocal == 0) {
		printf("nn[k][l] = %u \n", nn[k][l]);
	      }
	    }
	  }
	});

      // for each pixel, look at all the pixels until the end of the module;
      // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
      // after the loop, all the pixel in each cluster should have the id equeal to the lowest
      // pixel in the cluster ( clus[i] == i ).
      bool more = true;
      int nloops = 0;     
      while (alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalOr>(acc, more)) {
	if (1 == nloops % 2) {
	  uint32_t k = 0u;
	  cms::alpakatools::for_each_element_1D_block_stride(acc, hist.size(), [&](uint32_t j) {
	      k = j / blockDimension;
	      auto p = hist.begin() + j;
	      auto i = *p + firstPixel;
	      auto m = clusterId[i];
	      while (m != clusterId[m])
		m = clusterId[m];
	      clusterId[i] = m;
	    });
	} else {
	  more = false;
	  uint32_t k = 0u;
	  cms::alpakatools::for_each_element_1D_block_stride(acc, hist.size(), [&](uint32_t j) {
	      k = j / blockDimension;
	      auto p = hist.begin() + j;
	      auto i = *p + firstPixel;
	      for (int kk = 0; kk < nnn[k]; ++kk) {
		auto l = nn[k][kk];
		auto m = l + firstPixel;
		assert(m != i);
		auto old = alpaka::atomic::atomicOp<alpaka::atomic::op::Min>(acc, &clusterId[m], clusterId[i]);
		if (old != clusterId[i]) {
		  // end the loop only if no changes were applied
		  more = true;
		}
		alpaka::atomic::atomicOp<alpaka::atomic::op::Min>(acc, &clusterId[i], old);
		if (thisModuleId % 100 == 1 && threadIdxLocal == 0) {
		  printf("i = %u, clusterId[i] = %u \n", i, clusterId[i]);
		}
	      }  // nnloop
	    });    // pixel loop
	}
	++nloops;
	if (thisModuleId % 100 == 1 && threadIdxLocal == 0) {
	  printf("nloops = %u", nloops);
	}
      }  // end while

#ifdef GPU_DEBUG
      {
	auto&& n0 = alpaka::block::shared::st::allocVar<int, __COUNTER__>(acc);
	if (threadIdxLocal == 0)
	  n0 = nloops;
	alpaka::block::sync::syncBlockThreads(acc);
	auto ok = n0 == nloops;
	assert(alpaka::block::sync::syncBlockThreadsPredicate<alpaka::block::sync::op::LogicalAnd>(acc, ok));
	if (thisModuleId % 100 == 1)
	  if (threadIdxLocal == 0)
	    printf("# loops %d\n", nloops);
      }
#endif

      auto&& foundClusters = alpaka::block::shared::st::allocVar<unsigned int, __COUNTER__>(acc);
      foundClusters = 0;
      alpaka::block::sync::syncBlockThreads(acc);

      // find the number of different clusters, identified by a pixels with clus[i] == i;
      // mark these pixels with a negative id.
      cms::alpakatools::for_each_element_1D_block_stride(acc, msize, firstPixel, [&](uint32_t i) {
	  if (id[i] != InvId) {  // skip invalid pixels
	    if (clusterId[i] == static_cast<int>(i)) {
	      auto old = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, &foundClusters, 0xffffffff);
	      //auto old = alpaka::atomic::atomicOp<alpaka::atomic::op::Inc>(acc, &foundClusters, 4294967295u);
	      clusterId[i] = -(old + 1);
	    }
	  }
	});
      alpaka::block::sync::syncBlockThreads(acc);
      //if (thisModuleId % 100 == 1 && threadIdxLocal == 0) {
      if (threadIdxLocal == 0) {
	printf("thisModuleId = %u, msize = %u, firstPixel = %u, foundClusters = %u \n", thisModuleId, msize, firstPixel, foundClusters);
      }

      // propagate the negative id to all the pixels in the cluster.
      cms::alpakatools::for_each_element_1D_block_stride(acc, msize, firstPixel, [&](uint32_t i) {
	  if (id[i] != InvId) {  // skip invalid pixels
	    if (clusterId[i] >= 0) {
	      // mark each pixel in a cluster with the same id as the first one
	      clusterId[i] = clusterId[clusterId[i]];
	    }
	  }
	});
      alpaka::block::sync::syncBlockThreads(acc);

      // adjust the cluster id to be a positive value starting from 0
      cms::alpakatools::for_each_element_1D_block_stride(acc, msize, firstPixel, [&](uint32_t i) {
	  if (id[i] == InvId) {  // skip invalid pixels
	    clusterId[i] = -9999;
	  }
	  else {
	    clusterId[i] = -clusterId[i] - 1;
	  }
	});
      alpaka::block::sync::syncBlockThreads(acc);

      if (threadIdxLocal == 0) {
	nClustersInModule[thisModuleId] = foundClusters;
	moduleId[blockIdx] = thisModuleId;
#ifdef GPU_DEBUG
	if (foundClusters > gMaxHit) {
	  gMaxHit = foundClusters;
	  if (foundClusters > 8)
	    printf("max hit %d in %d\n", foundClusters, thisModuleId);
	}
#endif
#ifdef GPU_DEBUG
	if (thisModuleId % 100 == 1)
	  printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
      }
    }
  };

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
