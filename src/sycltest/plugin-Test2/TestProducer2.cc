#include <cassert>
#include <iostream>
#include <thread>

#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"

#include "SYCLCore/Product.h"
#include "SYCLCore/ScopedContext.h"

#include "gpuAlgo2.h"

namespace {
  std::atomic<int> nevents = 0;
}

class TestProducer2 : public edm::EDProducerExternalWork {
public:
  explicit TestProducer2(edm::ProductRegistry& reg);

private:
  void acquire(edm::Event const& event,
               edm::EventSetup const& eventSetup,
               edm::WaitingTaskWithArenaHolder holder) override;
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
  void endJob() override;

  edm::EDGetTokenT<cms::sycltools::Product<cms::sycltools::device::unique_ptr<float[]>>> getToken_;
};

TestProducer2::TestProducer2(edm::ProductRegistry& reg)
    : getToken_(reg.consumes<cms::sycltools::Product<cms::sycltools::device::unique_ptr<float[]>>>()) {}

void TestProducer2::acquire(edm::Event const& event,
                            edm::EventSetup const& eventSetup,
                            edm::WaitingTaskWithArenaHolder holder) {
  auto const& tmp = event.get(getToken_);

  cms::sycltools::ScopedContextAcquire ctx(tmp, std::move(holder));

  auto const& array = ctx.get(tmp);
  gpuAlgo2(ctx.stream());

  std::cout << "TestProducer2::acquire Event " << event.eventID() << " stream " << event.streamID() << " array "
            << array.get() << std::endl;
}

void TestProducer2::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  std::cout << "TestProducer2::produce Event " << event.eventID() << " stream " << event.streamID() << std::endl;
  ++nevents;
}

void TestProducer2::endJob() {
  std::cout << "TestProducer2::endJob processed " << nevents.load() << " events" << std::endl;
}

DEFINE_FWK_MODULE(TestProducer2);
