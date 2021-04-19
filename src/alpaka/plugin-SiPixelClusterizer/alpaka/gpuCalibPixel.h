#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"

#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
namespace gpuCalibPixel {

  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  // valid for run2
  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

struct calibDigis {
template <typename T_Acc>
ALPAKA_FN_ACC void operator()(const T_Acc& acc,
				bool isRun2,
				uint16_t* id,
				uint16_t const* __restrict__ x,
				uint16_t const* __restrict__ y,
				uint16_t* adc,
			      //SiPixelGainForHLTonGPU const* __restrict__ ped,
			      const SiPixelGainForHLTonGPU::DecodingStructure* __restrict__ v_pedestals, 
			      const SiPixelGainForHLTonGPU::RangeAndCols* __restrict__ rangeAndCols, 
			      const SiPixelGainForHLTonGPU::Fields* __restrict__ fields,
				int numElements,
				uint32_t* __restrict__ moduleStart,        // just to zero first
				uint32_t* __restrict__ nClustersInModule,  // just to zero them
				uint32_t* __restrict__ clusModuleStart     // just to zero first
				) const {
//int first = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);


  
  if (threadIdxGlobal == 0) {

    /*
    for (uint32_t i = 0; i < 48316; ++i) {
      printf("digis_d.moduleInd()[i] = %d \n", id[i]);
    }
    for (uint32_t i = 0; i < 48316; ++i) {
      printf("digis_d.c_xx()[i] = %d \n", x[i]);
    }
    for (uint32_t i = 0; i < 48316; ++i) {
      printf("digis_d.c_yy()[i] = %d \n", y[i]);
    }
    */

    /*
    for (uint32_t i = 0; i < 48316; ++i) {
      printf("KERNEL START adc[i] = %u \n", adc[i]);
      }*/


    /*
    //for (uint32_t i = 0; i < 1544192u; ++i) {
    for (uint32_t i = 0; i < 1u; ++i) {
      printf("gains->getVpedestals()->gain = %d \n", v_pedestals[i].gain);
      printf("gains->getVpedestals()->ped = %d \n", v_pedestals[i].ped);
    }

    for (uint32_t i = 0; i < 2000u; ++i) {
      printf("rangePtr[i].first.first = %d \n", rangeAndCols[i].first.first);
      printf("rangePtr[i].first.second = %d \n", rangeAndCols[i].first.second);
      printf("rangePtr[i].second = %d \n", rangeAndCols[i].second);
    }

    printf("gains->getFields()->minPed_ = %.6f \n", fields->minPed_);
    printf("gains->getFields()->maxPed_ = %.6f \n", fields->maxPed_);
    printf("gains->getFields()->minGain_ = %.6f \n", fields->minGain_);
    printf("gains->getFields()->maxGain_ = %.6f \n", fields->maxGain_);
    printf("gains->getFields()->pedPrecision = %.6f \n", fields->pedPrecision);
    printf("gains->getFields()->gainPrecision = %.6f \n", fields->gainPrecision);
    printf("gains->getFields()->numberOfRowsAveragedOver_ = %d \n", fields->numberOfRowsAveragedOver_);
    printf("gains->getFields()->nBinsToUseForEncoding_ = %d \n", fields->nBinsToUseForEncoding_);
    printf("gains->getFields()->deadFlag_ = %d \n", fields->deadFlag_);
    printf("gains->getFields()->noisyFlag_ = %d \n", fields->noisyFlag_);
    */


    }
    







  // zero for next kernels...
  if (threadIdxGlobal == 0) {
    clusModuleStart[0] = moduleStart[0] = 0;
  }

  //for (int i = first; i < gpuClustering::MaxNumModules; i += gridDim.x * blockDim.x) {
  cms::alpakatools::for_each_element_1D_grid_stride(acc, gpuClustering::MaxNumModules, [&](uint32_t i) {
      nClustersInModule[i] = 0;
    });

  //for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
  cms::alpakatools::for_each_element_1D_grid_stride(acc, numElements, [&](uint32_t i) {
      if (id[i] != InvId ) {
	float conversionFactor = (isRun2) ? (id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
	float offset = (isRun2) ? (id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

	bool isDeadColumn = false, isNoisyColumn = false;

	int row = x[i];
	int col = y[i];
	auto ret = SiPixelGainForHLTonGPU::getPedAndGain(v_pedestals, rangeAndCols, fields, id[i], col, row, isDeadColumn, isNoisyColumn);
	float pedestal = ret.first;
	float gain = ret.second;
	// float pedestal = 0; float gain = 1.;
	if (isDeadColumn | isNoisyColumn) {
	  id[i] = InvId;
	  adc[i] = 0;
	  printf("bad pixel at %d in %d\n", i, id[i]);
	} else {
	  float vcal = adc[i] * gain - pedestal * gain;
	  adc[i] = std::max(100, int(vcal * conversionFactor + offset));
	  //printf("KERNEL CALIB i = %u, adc[i] = %u \n", i, adc[i]);
	}
      }
    });


  }
};
}  // namespace gpuCalibPixel
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
