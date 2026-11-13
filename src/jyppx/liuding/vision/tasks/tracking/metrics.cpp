// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/tracking/metrics.hpp>
#include <jyppx/liuding/vision/ops/assignment.hpp>
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <set>
#include <numeric>
namespace jyppx::liuding::vision::tasks::tracking {
namespace {
using I=std::int64_t;
Tensor Similarity(const TrackingFrame& f){auto s=torch::zeros({static_cast<I>(f.ground_truth.size()),static_cast<I>(f.predictions.size())},torch::kFloat64);auto a=s.accessor<double,2>();
  for(I i=0;i<s.size(0);++i)for(I j=0;j<s.size(1);++j){const auto& x=f.ground_truth[i].box;const auto& y=f.predictions[j].box;double inter=(std::max)(0.,(std::min)(x[2],y[2])-(std::max)(x[0],y[0]))*(std::max)(0.,(std::min)(x[3],y[3])-(std::max)(x[1],y[1]));a[i][j]=inter/((x[2]-x[0])*(x[3]-x[1])+(y[2]-y[0])*(y[3]-y[1])-inter+1e-12);}return s;}
}
std::map<std::string,double> evaluate_tracking(std::span<const TrackingFrame> frames,std::stop_token stop){
  std::map<std::uint64_t,I> gt_ids,pred_ids;I total_gt=0,total_pred=0,previous_frame=-1;std::uint64_t pairs=0;
  for(const auto& f:frames){if(f.frame_id<=previous_frame||f.frame_id<0||f.ground_truth.size()>1024||f.predictions.size()>1024)throw InvalidArgument("Tracking metrics require ordered bounded frames");previous_frame=f.frame_id;
    for(const auto* rows:{&f.ground_truth,&f.predictions}){std::set<std::uint64_t> seen;for(const auto& r:*rows){if(!r.track_id||!seen.insert(r.track_id).second)throw InvalidArgument("Duplicate/zero frame track ID");for(double v:r.box)if(!std::isfinite(v))throw InvalidArgument("Nonfinite tracking metric box");if(r.box[2]<=r.box[0]||r.box[3]<=r.box[1])throw InvalidArgument("Invalid tracking metric box extent");}}
    for(const auto& r:f.ground_truth)if(!gt_ids.contains(r.track_id))gt_ids.emplace(r.track_id,gt_ids.size());
    for(const auto& r:f.predictions)if(!pred_ids.contains(r.track_id))pred_ids.emplace(r.track_id,pred_ids.size());
    total_gt+=f.ground_truth.size();total_pred+=f.predictions.size();pairs+=f.ground_truth.size()*f.predictions.size();}
  if(gt_ids.size()>4096||pred_ids.size()>4096||pairs>100000000)throw ResourceExhausted("Tracking metrics sequence budget exceeded");
  auto gt_count=torch::zeros({static_cast<I>(gt_ids.size())},torch::kFloat64),pred_count=torch::zeros({static_cast<I>(pred_ids.size())},torch::kFloat64);
  auto potential=torch::zeros({static_cast<I>(gt_ids.size()),static_cast<I>(pred_ids.size())},torch::kFloat64),identity=torch::zeros_like(potential);
  I tp=0,fp=0,fn=0,switches=0;std::map<I,I> previous_matched;
  for(const auto& f:frames){if(stop.stop_requested())throw Cancelled("Tracking metrics cancelled");auto sim=Similarity(f);std::vector<I> gi,pi;
    for(const auto& r:f.ground_truth){gi.push_back(gt_ids.at(r.track_id));gt_count[gi.back()]+=1;}for(const auto& r:f.predictions){pi.push_back(pred_ids.at(r.track_id));pred_count[pi.back()]+=1;}
    if(!sim.numel()){fn+=gi.size();fp+=pi.size();continue;}
    auto normalized=sim/(sim.sum(0,true)+sim.sum(1,true)-sim).clamp_min(1e-12);
    auto costs=1-sim;for(I i=0;i<sim.size(0);++i)for(I j=0;j<sim.size(1);++j){potential[gi[i]][pi[j]]+=normalized[i][j];if(sim[i][j].item<double>()>=.5){identity[gi[i]][pi[j]]+=1;if(previous_matched.contains(gi[i])&&previous_matched[gi[i]]==pi[j])costs[i][j]-=1000;}else costs[i][j]=2;}
    auto match=vision_ops::linear_assignment(costs,{.max_pairs=1048576,.max_work=1000000000,.cost_limit=.500000001},stop);
    tp+=match.rows.numel();fn+=gi.size()-match.rows.numel();fp+=pi.size()-match.rows.numel();
    for(I k=0;k<match.rows.numel();++k){auto g=gi[match.rows[k].item<I>()],p=pi[match.columns[k].item<I>()];if(previous_matched.contains(g)&&previous_matched[g]!=p)++switches;previous_matched[g]=p;}
  }
  double idtp=0;if(identity.numel()){auto match=vision_ops::linear_assignment(-identity,{.max_pairs=16777216,.max_work=100000000000},stop);for(I k=0;k<match.rows.numel();++k)idtp+=identity[match.rows[k].item<I>()][match.columns[k].item<I>()].item<double>();}
  auto alignment=potential/(gt_count.unsqueeze(1)+pred_count.unsqueeze(0)-potential).clamp_min(1e-12);double hota=0,deta=0,assa=0,loca=0;
  for(I alpha=1;alpha<=19;++alpha){auto matches=torch::zeros_like(potential);I hits=0;double localization=0;
    for(const auto& f:frames){if(stop.stop_requested())throw Cancelled("HOTA evaluation cancelled");auto sim=Similarity(f);if(!sim.numel())continue;auto score=torch::zeros_like(sim);
      for(I i=0;i<sim.size(0);++i)for(I j=0;j<sim.size(1);++j)score[i][j]=alignment[gt_ids.at(f.ground_truth[i].track_id)][pred_ids.at(f.predictions[j].track_id)]*sim[i][j];
      auto assignment=vision_ops::linear_assignment(-score,{.max_pairs=1048576,.max_work=1000000000},stop);
      for(I k=0;k<assignment.rows.numel();++k){I i=assignment.rows[k].item<I>(),j=assignment.columns[k].item<I>();double similarity=sim[i][j].item<double>();if(similarity+1e-12<alpha*.05)continue;
        ++hits;localization+=similarity;matches[gt_ids.at(f.ground_truth[i].track_id)][pred_ids.at(f.predictions[j].track_id)]+=1;}}
    double det=total_gt+total_pred-hits?static_cast<double>(hits)/(total_gt+total_pred-hits):0.;
    double association=hits?(matches.square()/(gt_count.unsqueeze(1)+pred_count.unsqueeze(0)-matches).clamp_min(1e-12)).sum().item<double>()/hits:0.;
    hota+=std::sqrt(det*association)/19;deta+=det/19;assa+=association/19;loca+=(hits?localization/hits:1.)/19;}
  return{{"mota",total_gt?1.-static_cast<double>(fn+fp+switches)/total_gt:0.},{"idf1",total_gt+total_pred?2*idtp/(total_gt+total_pred):0.},
    {"hota",hota},{"deta",deta},{"assa",assa},{"loca",loca},{"tp",static_cast<double>(tp)},{"fp",static_cast<double>(fp)},{"fn",static_cast<double>(fn)},{"id_switches",static_cast<double>(switches)}};
}
std::vector<TrackingFrame> read_mot_sequence(const std::filesystem::path& path,bool gt,std::size_t max_rows){if(!max_rows||max_rows>10000000||std::filesystem::file_size(path)>(512ULL<<20))throw ResourceExhausted("MOT input bound exceeded");std::ifstream input(path);if(!input)throw IoError("Cannot read MOT CSV");std::map<I,TrackingFrame> frames;std::string line;std::size_t rows=0;
  while(std::getline(input,line)){if(line.empty())continue;if(++rows>max_rows||line.size()>16384)throw ResourceExhausted("MOT CSV row bound exceeded");std::replace(line.begin(),line.end(),',',' ');std::istringstream stream(line);I frame,id;double x,y,w,h,score;
    if(!(stream>>frame>>id>>x>>y>>w>>h>>score)||frame<0||id<1||w<=0||h<=0||!std::isfinite(x+y+w+h+score))throw SerializationError("Invalid MOT CSV row");if(gt&&score<=0)continue;auto& f=frames[frame];f.frame_id=frame;TrackResult r{static_cast<std::uint64_t>(id),0,{x,y,x+w,y+h},score};(gt?f.ground_truth:f.predictions).push_back(r);}
  std::vector<TrackingFrame> out;for(auto& [id,f]:frames)out.push_back(std::move(f));return out;}
void write_mot_sequence(const std::filesystem::path& path,std::span<const TrackingFrame> frames){std::ofstream out(path);if(!out)throw IoError("Cannot write MOT CSV");out.precision(17);for(const auto& f:frames)for(const auto& r:f.predictions)out<<f.frame_id<<','<<r.track_id<<','<<r.box[0]<<','<<r.box[1]<<','<<r.box[2]-r.box[0]<<','<<r.box[3]-r.box[1]<<','<<r.confidence<<",-1,-1,-1\n";if(!out)throw IoError("MOT CSV publication failed");}
}
