// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "task_geometry.hpp"
#include <torch/nn/functional/upsampling.h>
#include <torch/nn/functional/padding.h>
#include <algorithm>
#include <unordered_map>
namespace jyppx::liuding::data::image::detail {
namespace {
using I=std::int64_t;
Tensor Dense(Tensor value,const ImageGeometryStep& s,double fill) {
  using K=ImageGeometryKind;
  if(s.kind==K::resize) return torch::nn::functional::interpolate(value.unsqueeze(0),
      torch::nn::functional::InterpolateFuncOptions().size(std::vector<I>{s.extent.height,s.extent.width}).mode(torch::kNearestExact)).squeeze(0);
  if(s.kind==K::crop) return value.slice(1,s.top,s.top+s.extent.height).slice(2,s.left,s.left+s.extent.width).clone();
  if(s.kind==K::pad) return torch::nn::functional::pad(value,
      torch::nn::functional::PadFuncOptions({s.left,s.right,s.top,s.bottom}).value(fill));
  return value.flip({s.kind==K::horizontal_flip?2:1});
}
Tensor MapBoxes(const Tensor& value,I offset,const GeometryTrace& trace,Tensor& keep) {
  if(!value.defined()) return {};
  auto m=trace.matrix(); auto x=value.to(torch::kFloat64);
  auto a=x.slice(1,offset,offset+2),b=x.slice(1,offset+2,offset+4);
  // Joint geometry is axis aligned. Reflections exchange both box corners.
  auto ax=a.select(1,0)*m[0]+m[2], ay=a.select(1,1)*m[4]+m[5];
  auto bx=b.select(1,0)*m[0]+m[2], by=b.select(1,1)*m[4]+m[5];
  auto e=trace.output_extent();
  auto boxes=torch::stack({torch::minimum(ax,bx).clamp(0,e.width),torch::minimum(ay,by).clamp(0,e.height),
                          torch::maximum(ax,bx).clamp(0,e.width),torch::maximum(ay,by).clamp(0,e.height)},1);
  keep=(boxes.select(1,2)>boxes.select(1,0))&(boxes.select(1,3)>boxes.select(1,1));
  auto mapped=value.clone();mapped.slice(1,offset,offset+4).copy_(boxes.to(value.scalar_type()));
  return mapped.index({keep}).contiguous();
}
}
void MapTaskPayload(const VisionSample& source,VisionSample& result,
                    std::span<const ImageGeometryStep> steps,const JointTransformOptions& options,
                    const GeometryTrace& trace,std::stop_token stop) {
  if(!source.task_payload)return;
  const auto& p=*source.task_payload;
  auto bytes=p.validate(options.limits.max_output_bytes);
  for(const auto& t:p.tensors())if(t.defined()&&!t.is_cpu())throw InvalidArgument("Joint task geometry needs CPU targets");
  if(p.pretrain_mask.defined()&&!steps.empty())throw InvalidArgument("Generate SimMIM masks after image geometry");
  if(p.depth.defined()&&(p.depth.size(1)!=source.height||p.depth.size(2)!=source.width))
    throw InvalidArgument("Depth and image canvas differ");
  if(p.track_ids.defined()&&(!source.annotation||p.track_ids.numel()!=static_cast<I>(source.annotation->objects.size())))
    throw InvalidArgument("Track identities must align with annotation rows");
  auto q=std::make_shared<VisionTaskPayload>(p);
  auto pixels=static_cast<std::uint64_t>(result.width)*result.height;
  for(const auto& step:steps)if(step.extent.width&&step.extent.height)pixels=(std::max)(pixels,static_cast<std::uint64_t>(step.extent.width)*step.extent.height);
  auto extra=p.depth.defined()?pixels*(p.depth.element_size()+1):0;
  if(extra>options.limits.max_output_bytes || bytes>options.limits.max_output_bytes-extra ||
     bytes+extra>options.limits.max_working_bytes/3)
    throw ResourceExhausted("Task geometry allocation budget exceeded");
  if(options.resource_budget){
    q->resource_leases=std::make_shared<ImageResourceLeaseSet>();
    q->resource_leases->push_back(options.resource_budget->reserve(ImageResourceKind::decoded_bytes,bytes+extra));
  }
  if(p.depth.defined()){
    q->depth=torch::where(p.depth_valid,p.depth,torch::zeros_like(p.depth));
    q->depth_valid=p.depth_valid.clone();
    for(const auto& s:steps){
      if(stop.stop_requested())throw Cancelled("Task geometry cancelled");
      q->depth=Dense(q->depth,s,0);q->depth_valid=Dense(q->depth_valid.to(torch::kFloat32),s,0).to(torch::kBool);
    }
  }
  Tensor keep;
  q->proposals=MapBoxes(p.proposals,0,trace,keep);
  if(p.proposal_scores.defined())q->proposal_scores=p.proposal_scores.index({keep}).contiguous();
  q->visual_prompts=MapBoxes(p.visual_prompts,1,trace,keep);
  if(p.camera_intrinsics.defined()){
    auto m=trace.matrix();auto affine=torch::from_blob(m.data(),{3,3},torch::kFloat64).clone().to(p.camera_intrinsics.options());
    q->camera_intrinsics=affine.matmul(p.camera_intrinsics);
  }
  if(p.track_ids.defined()){
    std::unordered_map<I,I> rows;
    for(I i=0;i<static_cast<I>(source.annotation->objects.size());++i)
      if(!rows.emplace(source.annotation->objects[i].source_id,i).second)throw InvalidArgument("Track geometry needs unique object source IDs");
    std::vector<I> indices;
    for(const auto& object:result.annotation->objects)indices.push_back(rows.at(object.source_id));
    q->track_ids=p.track_ids.index_select(0,torch::tensor(indices,torch::kInt64));
  }
  q->validate(options.limits.max_output_bytes);result.task_payload=std::move(q);
}
}
