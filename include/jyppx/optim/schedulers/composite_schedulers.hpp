#ifndef JYPPX_OPTIM_SCHEDULERS_COMPOSITE_SCHEDULERS_HPP
#define JYPPX_OPTIM_SCHEDULERS_COMPOSITE_SCHEDULERS_HPP
#include <jyppx/optim/schedulers/lr_scheduler.hpp>
namespace jyppx::optim::schedulers {
class SequentialLR : public LRScheduler { public: SequentialLR(std::shared_ptr<Optimizer>,std::vector<std::shared_ptr<LRScheduler>>,std::vector<int64_t>); using LRScheduler::step; void step() override; std::map<std::string,double> state_dict()const override; void load_state_dict(const std::map<std::string,double>&)override; private: std::vector<std::shared_ptr<LRScheduler>> schedulers_; std::vector<int64_t> milestones_; };
class ChainedScheduler : public LRScheduler { public: ChainedScheduler(std::shared_ptr<Optimizer>,std::vector<std::shared_ptr<LRScheduler>>); using LRScheduler::step; void step()override; std::map<std::string,double> state_dict()const override; void load_state_dict(const std::map<std::string,double>&)override; private: std::vector<std::shared_ptr<LRScheduler>> schedulers_; };
}
namespace jyppx::optim { using SequentialLR=schedulers::SequentialLR; using ChainedScheduler=schedulers::ChainedScheduler; }
#endif
