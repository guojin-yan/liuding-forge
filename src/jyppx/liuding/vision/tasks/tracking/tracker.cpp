// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/tracking/tracker.hpp>
#include <jyppx/liuding/vision/ops/assignment.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <set>
namespace jyppx::liuding::vision::tasks::tracking {
namespace {
using I=std::int64_t;using J=nlohmann::json;
double IoU(const std::array<double,4>& a,const std::array<double,4>& b){double w=(std::max)(0.,(std::min)(a[2],b[2])-(std::max)(a[0],b[0])),h=(std::max)(0.,(std::min)(a[3],b[3])-(std::max)(a[1],b[1]));
  double inter=w*h;return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-12);}
Tensor Vector(const std::array<double,4>& b,bool xywh){double w=b[2]-b[0],h=b[3]-b[1];return torch::tensor({(b[0]+b[2])*.5,(b[1]+b[3])*.5,xywh?w:w/h,h},torch::kFloat64);}
struct Detection{std::array<double,4> box;double score;I category;Tensor appearance;};
struct Track{std::uint64_t id=0;I category=0,last_frame=0,hits=0,state=0;bool confirmed=false;double score=0;Tensor mean,cov,appearance;
  std::array<double,4> box(bool xywh)const{double h=mean[3].item<double>(),w=xywh?mean[2].item<double>():mean[2].item<double>()*h,x=mean[0].item<double>(),y=mean[1].item<double>();w=(std::max)(1e-6,w);h=(std::max)(1e-6,h);return{x-w/2,y-h/2,x+w/2,y+h/2};}};
Tensor Noise(const Tensor& mean,bool xywh,bool initial,bool update){double h=(std::max)(1.,std::abs(mean[3].item<double>())),w=xywh?(std::max)(1.,std::abs(mean[2].item<double>())):h;
  double p=initial?2./20:1./20,v=initial?10./160:1./160;
  if(update)return torch::tensor({h/20,h/20,xywh?w/20:.1,h/20},torch::kFloat64).square();
  return torch::tensor({p*w,p*h,xywh?p*w:1e-2,p*h,v*w,v*h,xywh?v*w:1e-5,v*h},torch::kFloat64).square();}
void Predict(Track& t,bool xywh){if(t.state!=0){t.mean[7]=0;if(xywh)t.mean[6]=0;}auto f=torch::eye(8,torch::kFloat64);for(I i=0;i<4;++i)f[i][i+4]=1;
  auto q=Noise(t.mean,xywh,false,false).diag();t.mean=f.matmul(t.mean);t.cov=f.matmul(t.cov).matmul(f.t())+q;}
void Update(Track& t,const Detection& d,I frame,bool xywh){auto measurement=Vector(d.box,xywh);auto projected=t.cov.slice(0,0,4).slice(1,0,4)+Noise(t.mean,xywh,false,true).diag();
  auto gain=torch::linalg_solve(projected,t.cov.slice(0,0,4)).t();t.mean=t.mean+gain.matmul(measurement-t.mean.slice(0,0,4));t.cov=t.cov-gain.matmul(projected).matmul(gain.t());t.cov=(t.cov+t.cov.t())*.5;
  t.last_frame=frame;t.score=d.score;t.state=0;++t.hits;t.confirmed=true;
  if(d.appearance.defined()){t.appearance=t.appearance.defined()?.9*t.appearance+.1*d.appearance:d.appearance;t.appearance=t.appearance/t.appearance.norm().clamp_min(1e-12);}}
J Config(const TrackerOptions& o){return{{"kind",o.kind==TrackerKind::byte_track?"byte_track":"bot_sort.native-gmc-v1"},{"high",o.high_threshold},{"low",o.low_threshold},{"new",o.new_threshold},{"match",o.match_threshold},{"second",o.second_match_threshold},{"proximity",o.proximity_threshold},{"appearance",o.appearance_threshold},{"buffer",o.track_buffer},{"tracks",o.max_tracks},{"detections",o.max_detections},{"fuse",o.fuse_score},{"gmc",o.global_motion_compensation}};}
}
struct Tracker::Impl{TrackerOptions o;std::string sequence;I frame=-1;std::uint64_t next_id=1;std::vector<Track> tracks;explicit Impl(TrackerOptions options):o(options){
  for(double v:{o.high_threshold,o.low_threshold,o.new_threshold,o.match_threshold,o.second_match_threshold,o.proximity_threshold,o.appearance_threshold})if(!std::isfinite(v)||v<0||v>1)throw InvalidArgument("Invalid tracker thresholds");
  if(o.low_threshold>=o.high_threshold||o.new_threshold<o.high_threshold||!o.track_buffer||o.track_buffer>10000||o.max_tracks<1||o.max_tracks>1024||o.max_detections<1||o.max_detections>1024)throw InvalidArgument("Invalid tracker lifecycle bounds");}
};
Tracker::Tracker(TrackerOptions o):impl_(std::make_unique<Impl>(o)){}Tracker::~Tracker()=default;Tracker::Tracker(Tracker&&)noexcept=default;Tracker& Tracker::operator=(Tracker&&)noexcept=default;
void Tracker::reset(std::string sequence){if(sequence.empty()||sequence.size()>4096)throw InvalidArgument("Tracker needs bounded sequence identity");impl_->sequence=std::move(sequence);impl_->tracks.clear();impl_->frame=-1;impl_->next_id=1;}
std::vector<TrackResult> Tracker::step(const ImageResult& result,I frame,const Tensor& embeddings,const Tensor& affine){Impl staged=*impl_;for(auto& track:staged.tracks){track.mean=track.mean.clone();track.cov=track.cov.clone();if(track.appearance.defined())track.appearance=track.appearance.clone();}auto& p=staged;bool xywh=p.o.kind==TrackerKind::bot_sort;
  if(p.sequence.empty()||frame<0||frame<=p.frame||result.task!=TaskKind::detect||result.boxes.dim()!=2||result.boxes.size(1)!=4||result.boxes.size(0)>p.o.max_detections||result.scores.numel()!=result.boxes.size(0)||result.classes.numel()!=result.boxes.size(0))throw InvalidArgument("Invalid tracking sequence/frame/detections");
  auto boxes=result.boxes.to(torch::kCPU,torch::kFloat64),scores=result.scores.to(torch::kCPU,torch::kFloat64),classes=result.classes.to(torch::kCPU,torch::kInt64);
  if(!torch::isfinite(boxes).all().item<bool>()||!torch::isfinite(scores).all().item<bool>())throw InvalidArgument("Nonfinite tracking detections");
  Tensor features;if(embeddings.defined()){if(embeddings.dim()!=2||embeddings.size(0)!=boxes.size(0)||embeddings.size(1)>4096||!torch::isfinite(embeddings).all().item<bool>())throw InvalidArgument("Invalid ReID detection embeddings");features=embeddings.to(torch::kCPU,torch::kFloat64);features=features/features.square().sum(1,true).sqrt().clamp_min(1e-12);}
  if(xywh&&!features.defined()&&boxes.size(0))throw InvalidArgument("BoT-SORT mode requires detection ReID embeddings");
  Tensor motion;if(xywh&&p.o.global_motion_compensation){if(!affine.defined()||affine.sizes()!=torch::IntArrayRef({2,3})||!torch::isfinite(affine).all().item<bool>())throw InvalidArgument("BoT-SORT GMC requires an explicit affine (identity allowed)");motion=affine.to(torch::kCPU,torch::kFloat64);}
  std::vector<Detection> detections;std::vector<I> high,low;
  for(I i=0;i<boxes.size(0);++i){double score=scores[i].item<double>();if(score<0||score>1||classes[i].item<I>()<0||boxes[i][2].item<double>()<=boxes[i][0].item<double>()||boxes[i][3].item<double>()<=boxes[i][1].item<double>())throw InvalidArgument("Invalid tracker score/box/class");
    detections.push_back({{boxes[i][0].item<double>(),boxes[i][1].item<double>(),boxes[i][2].item<double>(),boxes[i][3].item<double>()},score,classes[i].item<I>(),features.defined()?features[i]:Tensor{}});if(score>=p.o.high_threshold)high.push_back(i);else if(score>=p.o.low_threshold)low.push_back(i);}
  auto gap=p.frame<0?1:frame-p.frame;if(gap>10000)throw ResourceExhausted("Tracker frame gap exceeds prediction budget");
  for(auto& t:p.tracks)if(t.state!=2){if(static_cast<std::uint64_t>(frame-t.last_frame)>p.o.track_buffer&&t.state==1){t.state=2;continue;}
    for(I k=0;k<gap;++k)Predict(t,xywh);
    if(motion.defined()){auto r=motion.slice(1,0,2);auto transform=torch::eye(8,torch::kFloat64);transform.slice(0,0,2).slice(1,0,2).copy_(r);transform.slice(0,4,6).slice(1,4,6).copy_(r);
      auto absolute=r.abs();transform.slice(0,2,4).slice(1,2,4).copy_(absolute);transform.slice(0,6,8).slice(1,6,8).copy_(absolute);t.mean=transform.matmul(t.mean);t.mean.slice(0,0,2).add_(motion.select(1,2));t.cov=transform.matmul(t.cov).matmul(transform.t());}}
  std::set<I> used_tracks,used_detections;
  auto associate=[&](const std::vector<I>& track_rows,const std::vector<I>& rows,double limit,bool use_appearance,bool fuse){if(track_rows.empty()||rows.empty())return;
    auto costs=torch::full({static_cast<I>(track_rows.size()),static_cast<I>(rows.size())},2.,torch::kFloat64);
    auto access=costs.accessor<double,2>();for(I i=0;i<static_cast<I>(track_rows.size());++i)for(I j=0;j<static_cast<I>(rows.size());++j){auto& t=p.tracks[track_rows[i]];auto& d=detections[rows[j]];if(t.category!=d.category)continue;auto iou=IoU(t.box(xywh),d.box);double cost=1-iou*(fuse?d.score:1.);
      if(use_appearance&&t.appearance.defined()&&d.appearance.defined()&&iou>=1-p.o.proximity_threshold){double distance=(1-t.appearance.dot(d.appearance).item<double>())/2;if(distance<=p.o.appearance_threshold)cost=(std::min)(cost,distance);}access[i][j]=cost;}
    auto match=vision_ops::linear_assignment(costs,{.cost_limit=limit});for(I i=0;i<match.rows.numel();++i){I t=track_rows[match.rows[i].item<I>()],d=rows[match.columns[i].item<I>()];Update(p.tracks[t],detections[d],frame,xywh);used_tracks.insert(t);used_detections.insert(d);}};
  std::vector<I> pool,tentative;for(I i=0;i<static_cast<I>(p.tracks.size());++i)if(p.tracks[i].state!=2){if(p.tracks[i].confirmed)pool.push_back(i);else tentative.push_back(i);}
  associate(pool,high,p.o.match_threshold,xywh,p.o.fuse_score);std::vector<I> unmatched;
  for(I i:pool)if(!used_tracks.contains(i)&&p.tracks[i].state==0)unmatched.push_back(i);associate(unmatched,low,p.o.second_match_threshold,false,false);
  std::vector<I> remaining;for(I i:high)if(!used_detections.contains(i))remaining.push_back(i);associate(tentative,remaining,.7,xywh,p.o.fuse_score);
  for(I i:pool)if(!used_tracks.contains(i))p.tracks[i].state=1;for(I i:tentative)if(!used_tracks.contains(i))p.tracks[i].state=2;
  for(I i:remaining)if(!used_detections.contains(i)&&detections[i].score>=p.o.new_threshold){const auto& d=detections[i];Track t;t.id=p.next_id++;t.category=d.category;t.last_frame=frame;t.hits=1;t.score=d.score;t.confirmed=p.frame<0;
    auto measurement=Vector(d.box,xywh);t.mean=torch::cat({measurement,torch::zeros({4},torch::kFloat64)});t.cov=Noise(t.mean,xywh,true,false).diag();t.appearance=d.appearance;p.tracks.push_back(std::move(t));}
  std::erase_if(p.tracks,[](const Track& t){return t.state==2;});if(p.tracks.size()>p.o.max_tracks)throw ResourceExhausted("Tracker live track budget exceeded");
  // Remove duplicate active/lost trajectories; keep the longer observed track.
  for(I i=0;i<static_cast<I>(p.tracks.size());++i)for(I j=i+1;j<static_cast<I>(p.tracks.size());++j)if(p.tracks[i].state!=p.tracks[j].state&&p.tracks[i].category==p.tracks[j].category&&IoU(p.tracks[i].box(xywh),p.tracks[j].box(xywh))>.85){if(p.tracks[i].hits>=p.tracks[j].hits)p.tracks[j].state=2;else p.tracks[i].state=2;}
  std::erase_if(p.tracks,[](const Track& t){return t.state==2;});p.frame=frame;std::vector<TrackResult> out;
  for(const auto& t:p.tracks)if(t.state==0&&t.confirmed)out.push_back({t.id,t.category,t.box(xywh),t.score});*impl_=std::move(staged);return out;
}
std::string Tracker::snapshot()const{const auto& p=*impl_;J j={{"schema","vision.tracker/v1"},{"config",Config(p.o)},{"sequence",p.sequence},{"frame",p.frame},{"next_id",p.next_id},{"tracks",J::array()}};
  for(const auto& t:p.tracks){auto values=[](const Tensor& x){std::vector<double> v;if(x.defined()){auto a=x.contiguous().reshape({-1});v.assign(a.data_ptr<double>(),a.data_ptr<double>()+a.numel());}return v;};
    j["tracks"].push_back({{"id",t.id},{"category",t.category},{"last_frame",t.last_frame},{"hits",t.hits},{"state",t.state},{"confirmed",t.confirmed},{"score",t.score},{"mean",values(t.mean)},{"covariance",values(t.cov)},{"appearance",values(t.appearance)}});}return j.dump();}
void Tracker::restore(std::string_view text){if(text.size()>(64ULL<<20))throw ResourceExhausted("Tracker snapshot too large");auto j=J::parse(text);auto fresh=std::make_unique<Impl>(impl_->o);
  if(j.at("schema")!="vision.tracker/v1"||j.at("config")!=Config(fresh->o))throw CompatibilityError("Tracker snapshot profile differs");fresh->sequence=j.at("sequence").get<std::string>();fresh->frame=j.at("frame").get<I>();fresh->next_id=j.at("next_id").get<std::uint64_t>();
  if(fresh->sequence.empty()||fresh->sequence.size()>4096||fresh->frame< -1||!fresh->next_id||j.at("tracks").size()>fresh->o.max_tracks)throw SerializationError("Invalid tracker snapshot control state");std::set<std::uint64_t> ids;
  for(const auto& row:j.at("tracks")){Track t;t.id=row.at("id").get<std::uint64_t>();t.category=row.at("category").get<I>();t.last_frame=row.at("last_frame").get<I>();t.hits=row.at("hits").get<I>();t.state=row.at("state").get<I>();t.confirmed=row.at("confirmed").get<bool>();t.score=row.at("score").get<double>();
    auto mean=row.at("mean").get<std::vector<double>>(),cov=row.at("covariance").get<std::vector<double>>(),appearance=row.at("appearance").get<std::vector<double>>();
    if(mean.size()!=8||cov.size()!=64||appearance.size()>4096||!t.id||t.id>=fresh->next_id||!ids.insert(t.id).second||t.category<0||t.hits<1||t.state<0||t.state>1||t.last_frame>fresh->frame||t.last_frame<0||!std::isfinite(t.score)||t.score<0||t.score>1)throw SerializationError("Invalid tracker snapshot entry");
    t.mean=torch::tensor(mean,torch::kFloat64);t.cov=torch::tensor(cov,torch::kFloat64).view({8,8});if(!appearance.empty())t.appearance=torch::tensor(appearance,torch::kFloat64);
    if(!torch::isfinite(t.mean).all().item<bool>()||!torch::isfinite(t.cov).all().item<bool>()||(t.appearance.defined()&&!torch::isfinite(t.appearance).all().item<bool>()))throw SerializationError("Nonfinite tracker snapshot");fresh->tracks.push_back(std::move(t));}
  impl_=std::move(fresh);
}
Tensor estimate_affine_motion(const Tensor& previous,const Tensor& current,I radius,I step){
  if(previous.dim()!=3||previous.size(0)!=3||previous.sizes()!=current.sizes()||radius<1||radius>16||step<8||step>128||previous.numel()>3*2048*2048)throw InvalidArgument("Invalid native affine GMC image/grid/budget");
  auto a=previous.to(torch::kCPU,torch::kFloat64).mean(0).contiguous(),b=current.to(torch::kCPU,torch::kFloat64).mean(0).contiguous();
  auto identity=torch::tensor({1.,0.,0.,0.,1.,0.},torch::kFloat64).view({2,3});std::vector<double> source,target;I border=radius+3;
  for(I y=border;y<a.size(0)-border && source.size()<3*512;y+=step)for(I x=border;x<a.size(1)-border && source.size()<3*512;x+=step){auto patch=a.slice(0,y-2,y+3).slice(1,x-2,x+3);if(patch.var().item<double>()<1e-5)continue;
    double best=1e100;I dx=0,dy=0;for(I oy=-radius;oy<=radius;++oy)for(I ox=-radius;ox<=radius;++ox){auto diff=patch-b.slice(0,y+oy-2,y+oy+3).slice(1,x+ox-2,x+ox+3);double error=diff.square().mean().item<double>();if(error<best){best=error;dx=ox;dy=oy;}}
    source.insert(source.end(),{static_cast<double>(x),static_cast<double>(y),1.});target.insert(target.end(),{static_cast<double>(x+dx),static_cast<double>(y+dy)});if(source.size()>3*512)break;}
  if(source.size()<18)return identity;auto x=torch::tensor(source,torch::kFloat64).view({-1,3}),y=torch::tensor(target,torch::kFloat64).view({-1,2});auto weights=torch::ones({x.size(0)},torch::kFloat64);Tensor affine;
  for(I iteration=0;iteration<5;++iteration){auto wx=x*weights.unsqueeze(1);affine=torch::linalg_solve(x.t().matmul(wx)+torch::eye(3,torch::kFloat64)*1e-6,wx.t().matmul(y));auto residual=(x.matmul(affine)-y).square().sum(1).sqrt();weights=1./(residual/2).clamp_min(1);}
  auto linear=affine.slice(0,0,2).t();double determinant=torch::linalg_det(linear).item<double>();if(!torch::isfinite(affine).all().item<bool>()||determinant<.5||determinant>2)return identity;return affine.t().contiguous();
}
}
