// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "extended_io.hpp"
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>
#if defined(JYPPX_VISION_WITH_EXTENDED)
#include <jyppx/liuding/vision/models/extended/region.hpp>
#endif
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <cmath>
namespace jyppx::liuding::vision::tasks::detail {
namespace {
using I=std::int64_t;using V=torch::jit::Value;using N=torch::jit::Node;
constexpr auto tag="liuding_vision::extended_symbolic";
c10::Symbol Symbol(){auto s=c10::Symbol::fromQualString(tag);static const bool once=[s]{torch::jit::registerOperator(torch::jit::Operator(s,[](const N*)->torch::jit::Operation{return [](torch::jit::Stack&){throw InternalError("Unlowered extended ONNX symbolic node");};},c10::AliasAnalysisKind::PURE_FUNCTION));return true;}();(void)once;return s;}
struct Graph {
  torch::jit::Graph& g;
  V* constant(const Tensor& t){return g.insertConstant(t);}
  V* scalar(double x){return constant(torch::scalar_tensor(x,torch::kFloat32));}
  V* ints(std::vector<I> x){return constant(torch::tensor(x,torch::kInt64));}
  N* node(const char* op,std::vector<V*> in){auto n=g.create(Symbol(),in,1);n->s_(c10::Symbol::attr("onnx_kind"),op);g.appendNode(n);n->output()->setType(c10::TensorType::get());return n;}
  V* op(const char* name,std::vector<V*> in){return node(name,std::move(in))->output();}
  V* slice(V* x,I a,I b,I axis=1){return op("Slice",{x,ints({a}),ints({b}),ints({axis})});}
  V* column(V* x,I i){auto n=node("Gather",{x,constant(torch::scalar_tensor(i,torch::kInt64))});n->i_(c10::Symbol::attr("axis"),1);return n->output();}
  V* cast(V* x,I type){auto n=node("Cast",{x});n->i_(c10::Symbol::attr("to"),type);return n->output();}
};
#if defined(JYPPX_VISION_WITH_EXTENDED)
std::vector<Tensor> Fast(nn::ModelAdapter& model,const Tensor& image,const Tensor& rois){
  namespace ex=models::extended;auto f=ex::fast_rcnn_features(model,image);const bool classic=ex::fast_rcnn_profile(model)==ex::FastRcnnProfile::vgg16_roi_pool;
  auto state=torch::jit::tracer::getTracingState();
  if(!state){auto pool=classic?vision_ops::roi_pool(f[0],rois,{7,7},1./16):vision_ops::multiscale_roi_align(f,rois,{4,8,16,32});auto h=ex::fast_rcnn_box_head(model,pool);return{rois,h[0],h[1]};}
  std::vector<V*> fv;for(auto& t:f)fv.push_back(torch::jit::tracer::getValueTrace(t));auto rv=torch::jit::tracer::getValueTrace(rois);
  auto resume=torch::jit::tracer::pauseTracing();struct Guard{std::function<void()> f;~Guard(){if(f)f();}} guard{resume};
  Graph g{*state->graph};V* pooled=nullptr;
  if(classic){auto n=g.node("MaxRoiPool",{fv[0],rv});n->is_(c10::Symbol::attr("pooled_shape"),{7,7});n->f_(c10::Symbol::attr("spatial_scale"),1./16);pooled=n->output();}
  else{auto box=g.slice(rv,1,5);auto wh=g.op("Max",{g.op("Sub",{g.slice(box,2,4),g.slice(box,0,2)}),g.scalar(0)});auto area=g.op("Mul",{g.column(wh,0),g.column(wh,1)});
    auto level=g.op("Floor",{g.op("Add",{g.scalar(4),g.op("Div",{g.op("Log",{g.op("Add",{g.op("Div",{g.op("Sqrt",{area}),g.scalar(224)}),g.scalar(1e-6)})}),g.scalar(std::log(2.))})})});level=g.op("Clip",{level,g.scalar(2),g.scalar(5)});
    auto batch=g.cast(g.column(rv,0),7);
    for(I i=0;i<4;++i){auto n=g.node("RoiAlign",{fv[i],box,batch});n->i_(c10::Symbol::attr("output_height"),7);n->i_(c10::Symbol::attr("output_width"),7);n->i_(c10::Symbol::attr("sampling_ratio"),2);n->f_(c10::Symbol::attr("spatial_scale"),1./(4LL<<i));n->s_(c10::Symbol::attr("mode"),"avg");n->s_(c10::Symbol::attr("coordinate_transformation_mode"),"output_half_pixel");
      auto mask=g.op("Reshape",{g.cast(g.op("Equal",{level,g.scalar(i+2)}),1),g.ints({rois.size(0),1,1,1})});auto part=g.op("Mul",{n->output(),mask});pooled=pooled?g.op("Add",{pooled,part}):part;}}
  auto native=classic?vision_ops::roi_pool(f[0],rois,{7,7},1./16):vision_ops::multiscale_roi_align(f,rois,{4,8,16,32});pooled->setType(c10::TensorType::create(native));guard.f();guard.f={};torch::jit::tracer::setValueTrace(native,pooled);auto h=ex::fast_rcnn_box_head(model,native);return{rois,h[0],h[1]};
}
#endif
}
bool extended_bundle(const RecipeSpec& r){const auto& n=r.model.architecture;return r.schema_version>=2||n.starts_with("fcos_")||n.starts_with("retinanet_")||n.starts_with("ssd300_")||n.starts_with("fast_rcnn_")||n.starts_with("rtdetr_")||n.starts_with("yoloe_")||n.starts_with("yoloworld_");}
ExtendedExport extended_export(const TaskBinding& b,const RecipeSpec& r,const VisionExportOptions& p,const Tensor& x){
  ExtendedExport e;e.examples={x};e.input_names={"images"};e.decoder="vision.extended/v1";const auto name=r.model.architecture;auto model=b.model;
  e.forward=[model](std::span<const Tensor> in){return raw_tensors(model->invoke(nn::Invocation{{in[0]},{}}).value);};
  if(name.starts_with("fast_rcnn_")){
#if defined(JYPPX_VISION_WITH_EXTENDED)
    if(p.rcnn_proposals*x.size(0)>4096)throw ResourceExhausted("Fast R-CNN export proposal bound exceeded");std::vector<float> data;for(I i=0;i<x.size(0);++i)for(I j=0;j<p.rcnn_proposals;++j)data.insert(data.end(),{static_cast<float>(i),0,0,static_cast<float>(x.size(3)),static_cast<float>(x.size(2))});
    e.examples.push_back(torch::tensor(data).view({-1,5}));e.input_names.push_back("proposals");e.output_names={"rois","class_logits","box_deltas"};e.forward=[model](std::span<const Tensor> in){return Fast(*model,in[0],in[1]);};
#else
    throw CompatibilityError("Fast R-CNN model pack disabled");
#endif
  }else if(name.starts_with("rtdetr_")){e.output_names={"class_logits","normalized_cxcywh"};e.forward=[model](std::span<const Tensor> in){auto v=raw_tensors(model->invoke(nn::Invocation{{in[0],torch::empty({0,6},in[0].options())},{}}).value);return std::vector<Tensor>{v[0][-1],v[1][-1]};};
  }else if(name.starts_with("yoloe_")||name.starts_with("yoloworld_")){bool visual=name.find("visual")!=name.npos,free=name.find("prompt_free")!=name.npos;bool segment=r.model.task==TaskKind::instance_segment;
    if(!free){e.input_names.push_back(visual?"visual_masks":"text_embeddings");e.examples.push_back(visual?torch::ones({x.size(0),r.model.num_classes,x.size(2)/8,x.size(3)/8}):torch::ones({x.size(0),r.model.num_classes,512}));}
    e.output_names={"raw_head"};if(segment)e.output_names.push_back("mask_prototypes");e.forward=[model,visual,free,segment](std::span<const Tensor> in){auto empty=torch::empty({0},in[0].options());auto raw=raw_tensors(model->invoke(nn::Invocation{{in[0],free||visual?empty:in[1],visual?in[1]:empty},{}}).value);raw.resize(segment?2:1);return raw;};
  }else if(r.model.task==TaskKind::pretrain_simmim){e.input_names.push_back("patch_mask");e.examples.push_back(torch::ones({x.size(0),x.size(2)/r.extension.model_patch_size,x.size(3)/r.extension.model_patch_size},torch::kInt64));e.output_names={"reconstruction"};e.forward=[model](std::span<const Tensor> in){return raw_tensors(model->invoke(nn::Invocation{{in[0],in[1].to(torch::kBool)},{}}).value);};
  }else if(name.starts_with("fcos_")){e.output_names={"class_logits","box_deltas","centerness","anchors","level_counts"};
  }else if(name.starts_with("retinanet_")||name.starts_with("ssd300_")){e.output_names={"class_logits","box_deltas","anchors","level_counts"};e.forward=[model](std::span<const Tensor> in){auto v=raw_tensors(model->invoke(nn::Invocation{{in[0]},{}}).value);return std::vector<Tensor>{v[0],v[1],v[3],v[4]};};
  }else if(r.model.task==TaskKind::classify||r.model.task==TaskKind::classify_multilabel||r.model.task==TaskKind::semantic_segment||r.model.task==TaskKind::depth){e.output_names={"prediction"};e.forward=[model](std::span<const Tensor> in){return std::vector<Tensor>{raw_tensors(model->invoke(nn::Invocation{{in[0]},{}}).value)[0]};};
  }else if(r.model.task==TaskKind::reid){e.output_names={"embedding"};e.forward=[model](std::span<const Tensor> in){return std::vector<Tensor>{raw_tensors(model->invoke(nn::Invocation{{in[0]},{}}).value)[1]};};}
  return e;
}
void add_extended_onnx_lowerings(std::map<std::string,onnx_export::OperatorLowering>& m){m[tag]=[](N& src,torch::jit::Graph& g){auto attr=c10::Symbol::attr("onnx_kind");auto n=g.create(c10::Symbol::onnx(src.s(attr)),src.inputs(),src.outputs().size());n->copyAttributes(src);n->removeAttribute(attr);n->insertBefore(&src);return std::vector<V*>{n->output()};};}
}
