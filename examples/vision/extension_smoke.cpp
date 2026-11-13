// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
#include <jyppx/liuding/vision/tasks/training/distillation.hpp>
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/tracking/metrics.hpp>
#include <jyppx/liuding/vision/ops/assignment.hpp>
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <jyppx/liuding/nn/recompute.hpp>
#include <jyppx/liuding/vision/models/extended/dense.hpp>
#if defined(JYPPX_VISION_SMOKE_ONNX)
#include <jyppx/liuding/deployment/onnx_runtime/typed_session.hpp>
#endif
#include <jyppx/liuding/vision/tasks/workflow/suite.hpp>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
namespace j=jyppx::liuding;namespace t=j::vision::tasks;namespace tr=t::tracking;
void Require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
t::RecipeSpec Recipe(const std::string& name){t::RecipeSpec r;r.schema_version=2;r.model.architecture=name;r.model.num_classes=3;r.extension.training_manifest="__smoke__";r.training.source="__smoke__";r.classification_root="__smoke__";r.batch_size=2;r.data.letterbox.canvas={64,64};r.data.augmentation.projective={};r.data.augmentation.multi_source={};r.data.augmentation.projective.scale_gain=0;r.data.augmentation.projective.translate_fraction=0;
  for(const auto& c:t::vision_catalog())if(c.architecture==name){r.model.task=c.task;if(c.fixed_input_size[0])r.data.letterbox.canvas={static_cast<std::uint32_t>(c.fixed_input_size[1]),static_cast<std::uint32_t>(c.fixed_input_size[0])};else if(c.minimum_input_size>64)r.data.letterbox.canvas={static_cast<std::uint32_t>(c.minimum_input_size),static_cast<std::uint32_t>(c.minimum_input_size)};break;}
  if(name.starts_with("simmim_vit"))r.extension.model_patch_size=16;return r;}

#if defined(JYPPX_VISION_SMOKE_ONNX)
void VerifyExport(t::TaskBinding& binding,const t::RecipeSpec& recipe,const std::filesystem::path& root){
  namespace ort=j::onnx_runtime;using I=std::int64_t;using J=nlohmann::json;
  std::ifstream manifest(root/"manifest.json");J m;manifest>>m;
  std::ifstream stream(root/"model.onnx",std::ios::binary|std::ios::ate);auto length=stream.tellg();Require(length>0,"Missing ONNX bytes");std::vector<std::byte> bytes(static_cast<std::size_t>(length));stream.seekg(0);stream.read(reinterpret_cast<char*>(bytes.data()),length);
  ort::TypedSessionOptions limits;limits.max_model_bytes=2ULL<<30;limits.max_input_bytes=limits.max_output_bytes=1ULL<<30;
  ort::TypedSession session(bytes,{.hex=m.at("sha256").get<std::string>()},limits);binding.model->state().eval();torch::NoGradGuard guard;
  I h=recipe.data.letterbox.canvas.height,w=recipe.data.letterbox.canvas.width;auto image=torch::randn({1,3,h,w});std::vector<j::Tensor> inputs{image};j::nn::Invocation call{{image},{}};const auto name=recipe.model.architecture;
  if(name.starts_with("fast_rcnn_")){auto roi=torch::tensor(std::vector<float>{0,2,2,float(w)/2,float(h)/2,0,float(w)/4,float(h)/4,float(w)-2,float(h)-2}).view({2,5});inputs.push_back(roi);call.positional.insert(call.positional.end(),{roi,torch::empty({0,6}),torch::tensor({float(h),float(w)}).view({1,2})});}
  else if(name.starts_with("rtdetr_"))call.positional.push_back(torch::empty({0,6}));
  else if(name.starts_with("yoloe_")||name.starts_with("yoloworld_")){bool visual=name.find("visual")!=name.npos,free=name.find("prompt_free")!=name.npos;auto empty=torch::empty({0});auto prompt=visual?torch::rand({1,recipe.model.num_classes,h/8,w/8}):torch::randn({1,recipe.model.num_classes,512});if(!free)inputs.push_back(prompt);call.positional.insert(call.positional.end(),{free||visual?empty:prompt,visual?prompt:empty});}
  else if(recipe.model.task==t::TaskKind::pretrain_simmim){auto mask=torch::randint(0,2,{1,h/recipe.extension.model_patch_size,w/recipe.extension.model_patch_size},torch::kInt64);inputs.push_back(mask);call.positional.push_back(mask.to(torch::kBool));}
  auto expected=t::raw_tensors(binding.model->invoke(call).value);
  if(name.starts_with("fast_rcnn_"))expected={expected[3],expected[4],expected[5]};
  else if(name.starts_with("rtdetr_"))expected={expected[0][-1],expected[1][-1]};
  else if(name.starts_with("retinanet_")||name.starts_with("ssd300_"))expected={expected[0],expected[1],expected[3],expected[4]};
  else if(name.starts_with("yoloe_")||name.starts_with("yoloworld_"))expected.resize(recipe.model.task==t::TaskKind::instance_segment?2:1);
  else if(recipe.model.task==t::TaskKind::reid)expected={expected[1]};
  else if(!name.starts_with("fcos_"))expected.resize(1);
  std::vector<ort::TypedTensorInput> arguments;std::vector<std::vector<I>> shapes;shapes.reserve(inputs.size());
  for(std::size_t i=0;i<inputs.size();++i){inputs[i]=inputs[i].contiguous();auto& value=inputs[i];shapes.push_back(value.sizes().vec());ort::TypedInputValues values;if(value.scalar_type()==torch::kInt64)values=std::span(value.const_data_ptr<I>(),static_cast<std::size_t>(value.numel()));else values=std::span(value.const_data_ptr<float>(),static_cast<std::size_t>(value.numel()));arguments.push_back({session.inputs().at(i).name,values,shapes.back()});}
  auto actual=session.run(arguments);Require(actual.size()==expected.size(),"ORT output arity mismatch");double error=0;std::vector<j::Tensor> outputs;
  for(std::size_t i=0;i<actual.size();++i)std::visit([&](auto& values){using T=typename std::decay_t<decltype(values)>::value_type;auto dtype=std::is_same_v<T,float>?torch::kFloat32:std::is_same_v<T,I>?torch::kInt64:torch::kInt32;outputs.push_back(torch::from_blob(values.data(),actual[i].shape,dtype).clone());Require(outputs.back().sizes()==expected[i].sizes(),"ORT raw output shape mismatch");},actual[i].values);
  if(name.starts_with("rtdetr_")){
    // TopK tie order differs between runtimes. Queries form a set; match each
    // logits/box pair jointly, retaining the same numeric tolerance below.
    auto cost=(outputs[1][0].unsqueeze(1)-expected[1][0].unsqueeze(0)).square().sum(-1)+
              (outputs[0][0].unsqueeze(1)-expected[0][0].unsqueeze(0)).square().sum(-1);
    auto match=j::vision_ops::linear_assignment(cost);
    Require(torch::equal(match.rows,torch::arange(cost.size(0),torch::kInt64)),"Incomplete RT-DETR query assignment");
    for(auto& value:expected)value=value.index_select(1,match.columns);
    std::cout<<"RT-DETR compared jointly matched logits/box query pairs\n";
  }
  for(std::size_t i=0;i<outputs.size();++i){auto& output=outputs[i];if(output.is_floating_point()){error=(std::max)(error,(output-expected[i]).abs().max().item<double>());if(!torch::allclose(output,expected[i],.002,.0002)){std::cerr<<"output="<<i<<" max_abs="<<(output-expected[i]).abs().max().item<double>()<<" mean_abs="<<(output-expected[i]).abs().mean().item<double>()<<" first native="<<expected[i].reshape({-1}).slice(0,0,8)<<" first ORT="<<output.reshape({-1}).slice(0,0,8)<<'\n';throw std::runtime_error("ORT raw values differ from native forward");}}else Require(torch::equal(output,expected[i]),"ORT integer output mismatch");}
  j::data::image::VisionSample sample;sample.width=w;sample.height=h;sample.image_path="export-fixture";sample.image=torch::randint(0,256,{3,h,w},torch::kUInt8);auto payload=std::make_shared<j::data::image::VisionTaskPayload>();
  if(name.starts_with("fast_rcnn_")){payload->proposals=inputs[1].slice(1,1,5);payload->proposal_generator="fixture";payload->proposal_inventory="fixture";}
  if(name.starts_with("yoloe_")||name.starts_with("yoloworld_")){payload->prompt_embeddings=torch::randn({recipe.model.num_classes,512});for(I c=0;c<recipe.model.num_classes;++c)payload->text_prompts.push_back("class "+std::to_string(c));payload->visual_prompts=torch::cat({torch::arange(recipe.model.num_classes,torch::kFloat32).view({-1,1}),torch::tensor(std::vector<float>{0,0,float(w)/2,float(h)/2}).view({1,4}).expand({recipe.model.num_classes,4})},1);payload->label_space_id="fixture";payload->text_encoder_id="fixture";}
  sample.task_payload=payload;auto predicted=J::parse(t::predict_vision_bundle(root,std::span(&sample,1)));Require(predicted.at("results").size()==1,"Offline bundle prediction failed");std::cout<<"export + ORT raw comparison + offline bundle passed, max_abs="<<error<<'\n';
}


#endif
void Distillation(){auto student_recipe=Recipe("resnet18");student_recipe.extension.feature_outputs=true;student_recipe.extension.feature_projection=2048;auto teacher_recipe=Recipe("resnet50");teacher_recipe.extension.feature_outputs=true;auto student=t::bind_task(student_recipe),teacher=t::bind_task(teacher_recipe);auto step=t::make_distillation_step(student.step,teacher.model,{.temperature=2,.logits_weight=1,.feature_weight=1});std::vector<j::Tensor> frozen;for(auto& value:teacher.model->state().parameters())frozen.push_back(value.detach().clone());for(auto& value:teacher.model->state().buffers())frozen.push_back(value.detach().clone());j::data::BatchEnvelope envelope;j::training::StepContext context;j::nn::Invocation call{{torch::randn({2,3,64,64})},{}};auto output=step({*student.model,envelope,call,torch::tensor({0,1},torch::kInt64),context});Require(torch::isfinite(output.loss).item<bool>(),"Nonfinite distillation loss");output.loss.backward();auto parameters=student.model->state().named_parameters();Require(parameters["distillation_projection.weight"].grad().defined()&&parameters["distillation_projection.weight"].grad().abs().sum().item<double>()>0,"KD projection has no gradient");std::size_t i=0;for(auto& value:teacher.model->state().parameters())Require(!value.grad().defined()&&torch::equal(value,frozen[i++]),"Teacher parameter mutated");for(auto& value:teacher.model->state().buffers())Require(torch::equal(value,frozen[i++]),"Teacher buffer mutated");std::cout<<"logits + feature KD frozen teacher smoke passed\n";}

void Core(){auto costs=torch::tensor({4.,1.,3.5,2.,0.,5.}).view({2,3});auto match=j::vision_ops::linear_assignment(costs);Require(match.columns.equal(torch::tensor({1,0},torch::kInt64)),"Rectangular optimum mismatch");auto partial=j::vision_ops::linear_assignment(torch::tensor({.1,.2,.2,100.}).view({2,2}),{.cost_limit=.5});Require(partial.rows.numel()==2,"Partial matching discarded a valid global optimum");
  auto v=torch::tensor({1.,2.,3.,4.}).view({1,4,1,1}).requires_grad_(true),loc=torch::full({1,1,1,1,1,2},.5).requires_grad_(true),w=torch::ones({1,1,1,1,1}).requires_grad_(true);std::array<std::array<std::int64_t,2>,1> shapes{{{2,2}}};auto out=j::vision_ops::multiscale_deformable_attention(v,shapes,loc,w);Require(torch::allclose(out,torch::full_like(out,2.5)),"Deformable sampling reference mismatch");out.sum().backward();Require(v.grad().defined()&&loc.grad().defined()&&w.grad().defined()&&loc.grad().abs().sum().item<double>()>0,"Deformable coordinate/value/weight gradient missing");
  auto region=std::make_shared<torch::nn::SequentialImpl>();region->push_back(torch::nn::Linear(4,5));region->push_back(torch::nn::Dropout(.4));region->push_back(torch::nn::Linear(5,2));auto x=torch::randn({3,4});torch::manual_seed(73);auto plain=region->forward(x);plain.square().mean().backward();std::vector<j::Tensor> gradients;for(auto& p:region->parameters()){gradients.push_back(p.grad().clone());p.mutable_grad()=j::Tensor{};}auto next=torch::randn({4});torch::manual_seed(73);auto recomputed=j::nn::recompute(x,region,[](torch::nn::Module& m,const j::Tensor& a){return dynamic_cast<torch::nn::SequentialImpl&>(m).forward(a);});recomputed.square().mean().backward();Require(torch::allclose(plain,recomputed),"Recompute dropout forward mismatch");Require(torch::equal(next,torch::randn({4})),"Recompute changed caller RNG progression");std::size_t i=0;for(auto& p:region->parameters())Require(p.grad().defined()&&torch::allclose(p.grad(),gradients[i++],1e-5,1e-6),"Recompute parameter gradient mismatch with data input requiring no gradient");
  j::data::image::VisionTaskPayload payload;payload.multilabel_values=torch::tensor({1.,0.,0.});payload.known_labels=torch::tensor({true,true,false},torch::kBool);auto round=j::data::image::task_payload_from_value(j::data::image::task_payload_value(payload));Require(torch::equal(round->known_labels,payload.known_labels),"Typed payload round trip mismatch");
  auto logits=torch::tensor({0.,0.,100.}).view({1,3}).requires_grad_(true);auto loss=t::multilabel_loss(logits,payload.multilabel_values.unsqueeze(0),payload.known_labels.unsqueeze(0));loss.backward();Require(logits.grad()[0][2].item<float>()==0,"Unknown label contributed gradient");
  tr::Tracker tracker;tracker.reset("fixture");t::ImageResult d;d.boxes=torch::tensor({0.,0.,20.,20.}).view({1,4});d.scores=torch::tensor({.9});d.classes=torch::zeros({1},torch::kInt64);auto first=tracker.step(d,0);Require(first.size()==1,"Tracker first frame did not start ID");auto snapshot=tracker.snapshot();tr::Tracker restored;restored.restore(snapshot);auto a=tracker.step(d,1),b=restored.step(d,1);Require(a.size()==1&&b.size()==1&&a[0].track_id==b[0].track_id,"Tracker restore changed identity");std::array<tr::TrackingFrame,2> frames{{{0,first,first},{1,a,a}}};auto metrics=tr::evaluate_tracking(frames);for(auto key:{"mota","idf1","hota"})Require(std::abs(metrics.at(key)-1)<1e-6,"Perfect tracking metric reference mismatch");
  auto r=Recipe("fcos_resnet50_fpn");auto parsed=t::parse_recipe(t::recipe_json(r),std::filesystem::current_path());Require(parsed.model.architecture==r.model.architecture&&parsed.schema_version==2,"Schema2 recipe round trip mismatch");std::cout<<"core smoke passed\n";
}
int main(int argc,char** argv){try{torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::manual_seed(7);std::string command=argc>1?argv[1]:"core";if(command=="core"){Core();return 0;}if(command=="kd"){Distillation();return 0;}if(argc<3)throw std::runtime_error("Expected model name");auto r=Recipe(argv[2]);auto b=t::bind_task(r);auto h=r.data.letterbox.canvas.height,w=r.data.letterbox.canvas.width;
  if(command=="export"){if(argc!=4)throw std::runtime_error("Expected new output directory");t::VisionExportOptions p;p.batch_size=1;p.rcnn_proposals=2;p.max_bytes=2ULL<<30;auto root=std::filesystem::u8path(argv[3]);t::export_vision(b,r,root,p);
#if defined(JYPPX_VISION_SMOKE_ONNX)
VerifyExport(b,r,root);
#endif
return 0;}
  auto x=torch::randn({2,3,h,w});j::nn::Invocation in{{x},{}};std::optional<c10::IValue> target;using K=t::TaskKind;auto y=torch::tensor({0.,0.,8.,8.,32.,32.,1.,1.,12.,12.,40.,40.}).view({2,6});
  if(r.model.architecture.starts_with("rtdetr_")){in.positional.push_back(y);target=y;}
  else if(r.model.architecture.starts_with("fast_rcnn_")){auto rois=torch::cat({y.slice(1,0,1),y.slice(1,2,6)},1);in.positional.insert(in.positional.end(),{rois,y,torch::tensor({static_cast<float>(h),static_cast<float>(w)}).view({1,2}).expand({2,2})});target=y;}
  else if(r.model.architecture.starts_with("yoloe_")||r.model.architecture.starts_with("yoloworld_")){in.positional.push_back(torch::randn({2,3,512}));in.positional.push_back(torch::ones({2,3,h/8,w/8}));target=c10::ivalue::Tuple::create({y,torch::empty({0}),torch::empty({0}),torch::empty({0})});}
  else if(r.model.task==K::pretrain_simmim){auto mask=torch::ones({2,h/r.extension.model_patch_size,w/r.extension.model_patch_size},torch::kBool);in.positional.push_back(mask);target=c10::ivalue::Tuple::create({x,mask});}
  else if(r.model.task==K::semantic_segment)target=torch::zeros({2,h,w},torch::kInt64);
  else if(r.model.task==K::depth)target=c10::ivalue::Tuple::create({torch::ones({2,1,h,w}),torch::ones({2,1,h,w},torch::kBool)});
  else if(r.model.task==K::classify_multilabel)target=c10::ivalue::Tuple::create({torch::zeros({2,3}),torch::ones({2,3},torch::kBool)});
  else if(r.model.task==K::classify)target=torch::tensor({0,1},torch::kInt64);
  else if(r.model.task==K::reid){b.step=t::make_reid_step(.3,0);target=torch::tensor({0,1},torch::kInt64);}
  else target=y;
  b.model->state().train();torch::optim::SGD optimizer(b.model->state().parameters(),torch::optim::SGDOptions(.001));auto before=b.model->state().parameters()[0].detach().clone();j::data::BatchEnvelope envelope;j::training::StepContext context;auto out=b.step({*b.model,envelope,in,target,context});Require(out.loss.defined()&&torch::isfinite(out.loss).item<bool>(),"Nonfinite smoke loss");out.loss.backward();optimizer.step();Require(!torch::equal(before,b.model->state().parameters()[0]),"Optimizer did not update native model");std::cout<<argv[2]<<" loss="<<out.loss.item<double>()<<" update passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
