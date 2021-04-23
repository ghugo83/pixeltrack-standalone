#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

//#include "CAHitNtupletGeneratorOnGPU.h"
#include "AlpakaDataFormats/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/TrackingRecHit2DAlpaka.h"

#include "AlpakaCore/alpakaCommon.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

class CAHitNtupletAlpaka : public edm::EDProducer {
public:
  explicit CAHitNtupletAlpaka(edm::ProductRegistry& reg);
  ~CAHitNtupletAlpaka() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<TrackingRecHit2DAlpaka> tokenHitGPU_;
//edm::EDPutTokenT<PixelTrackHeterogeneous> tokenTrackGPU_;

//CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletAlpaka::CAHitNtupletAlpaka(edm::ProductRegistry& reg)
  : tokenHitGPU_{reg.consumes<TrackingRecHit2DAlpaka>()}//,
//tokenTrackGPU_{reg.produces<PixelTrackHeterogeneous>()},
//gpuAlgo_(reg) 
{}

void CAHitNtupletAlpaka::produce(edm::Event& iEvent, const edm::EventSetup& es) {
auto bf = 0.0114256972711507;  // 1/fieldInGeV

    auto const& hits = iEvent.get(tokenHitGPU_);

// TO DO: Async: Would need to add a queue as a parameter, not async for now!
//iEvent.emplace(tokenTrackGPU_, gpuAlgo_.makeTuples(hits, bf));
}

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpaka);
