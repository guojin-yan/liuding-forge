// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "artifact_io.hpp"
#include "extended_io.hpp"
#include <jyppx/liuding/deployment/onnx_runtime/typed_session.hpp>
#include <jyppx/liuding/vision/tasks/data/extended.hpp>
#include <jyppx/liuding/vision/tasks/open_vocabulary.hpp>
#include <jyppx/liuding/vision/tasks/rtdetr.hpp>
#include <jyppx/liuding/vision/tasks/dense_detection.hpp>
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/original_v5.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
namespace jyppx::liuding::vision::tasks::detail {
std::string predict_extended_bundle(const std::filesystem::path& root,
    std::span<const data::image::VisionSample> samples,std::stop_token stop){
  using J=nlohmann::json;using I=std::int64_t;namespace im=data::image;namespace ort=onnx_runtime;
  auto bytes=read_artifact(root/"manifest.json",4U<<20);auto m=J::parse(bytes.begin(),bytes.end());
  if(m.at("schema")!="vision.onnx_bundle/v3"||m.at("model_file")!="model.onnx"||m.at("io_profile")!="cpu.dense_typed/v1"||samples.empty()||samples.size()!=m.at("fixed_batch").get<std::size_t>())throw CompatibilityError("Extended offline bundle identity/batch mismatch");
  auto r=parse_recipe(m.at("recipe").dump(),root);r.device=Device(torch::kCPU);const auto name=r.model.architecture;
  auto divisor=m.at("size_divisor").get<I>();
  if(divisor<1||divisor>256||r.data.letterbox.canvas.width%divisor||r.data.letterbox.canvas.height%divisor)throw CompatibilityError("Extended bundle input divisor mismatch");
  r.data.size_divisor=static_cast<std::uint32_t>(divisor);
  const bool region=name=="faster_rcnn_resnet50_fpn";
  const bool end_to_end=name.starts_with("rtdetr_")||(name.starts_with("yolo26")&&r.model.task!=TaskKind::classify);
  if(m.at("opset")!=17||m.at("fixed_batch").get<I>()<1||m.at("fixed_batch").get<I>()>64||m.at("end_to_end")!=end_to_end||m.at("decoder")!=(region?"rcnn.fixed_proposals/v2":"vision.extended/v1")||m.at("architecture_revision").get<std::string>().empty())throw CompatibilityError("Extended bundle decoder/revision/opset contract mismatch");
  for(const auto& c:vision_catalog())if(c.architecture==name&&c.task==r.model.task&&c.variant==r.model.variant&&(c.revision!=m.at("architecture_revision").get<std::string>()||c.input_divisor!=divisor))throw CompatibilityError("Extended bundle architecture revision/divisor mismatch");
  std::vector<std::string> expected_inputs{"images"},expected_outputs;
  if(name.starts_with("fast_rcnn_")){expected_inputs.push_back("proposals");expected_outputs={"rois","class_logits","box_deltas"};}
  else if(name.starts_with("rtdetr_"))expected_outputs={"class_logits","normalized_cxcywh"};
  else if(name.starts_with("yoloe_")||name.starts_with("yoloworld_")){if(name.find("prompt_free")==name.npos)expected_inputs.push_back(name.find("visual")!=name.npos?"visual_masks":"text_embeddings");expected_outputs={"raw_head"};if(r.model.task==TaskKind::instance_segment)expected_outputs.push_back("mask_prototypes");}
  else if(name.starts_with("fcos_"))expected_outputs={"class_logits","box_deltas","centerness","anchors","level_counts"};
  else if(name.starts_with("retinanet_")||name.starts_with("ssd300_"))expected_outputs={"class_logits","box_deltas","anchors","level_counts"};
  else if(r.model.task==TaskKind::pretrain_simmim){expected_inputs.push_back("patch_mask");expected_outputs={"reconstruction"};}
  else if(r.model.task==TaskKind::reid)expected_outputs={"embedding"};
  else if(r.model.task==TaskKind::classify||r.model.task==TaskKind::classify_multilabel||r.model.task==TaskKind::semantic_segment||r.model.task==TaskKind::depth)expected_outputs={"prediction"};
  else if(region)expected_outputs={"proposals","class_logits","box_deltas","valid_proposals"};
  if(m.at("input_names")!=expected_inputs||(!expected_outputs.empty()&&m.at("output_names")!=expected_outputs))throw CompatibilityError("Extended bundle semantic I/O names mismatch");
  const bool classify=r.model.task==TaskKind::classify||r.model.task==TaskKind::classify_multilabel||r.model.task==TaskKind::reid;
  if(m.at("width")!=r.data.letterbox.canvas.width||m.at("height")!=r.data.letterbox.canvas.height||m.at("class_count")!=r.model.num_classes||m.at("geometry")!=(classify?"deterministic_center_crop":"letterbox_with_inverse_trace")||m.at("pixel_contract")!=J{{"color","RGB"},{"layout","NCHW"},{"dtype","float32"},{"range","unit_then_recipe_normalization"}})throw CompatibilityError("Extended bundle preprocessing contract mismatch");
  auto start=std::chrono::steady_clock::now();auto data=yolo_evaluation_options(r.data);data.allow_unannotated_negatives=true;data.num_classes=r.model.num_classes;data.classification.enabled=classify;data.augmentation.dataset_id="onnx-bundle:"+m.at("sha256").get<std::string>();
  std::vector<im::ImageBatchKey> keys;for(auto& s:samples)keys.push_back({.id=s.image_path});auto batch=prepare_yolo_batch(samples,keys,data,stop);data::BatchEnvelope envelope;envelope.value=im::image_batch_value(batch,data.limits);
  std::vector<Tensor> tensors{batch.images};
  if(name.starts_with("fast_rcnn_")){auto invocation=make_extended_image_mapper(r.model.task,data,true)(envelope);tensors.push_back(invocation.positional[1].toTensor());}
  else if(name.starts_with("yoloe_")||name.starts_with("yoloworld_")){PromptInputOptions p;p.classes=r.model.num_classes;p.visual=name.find("visual")!=name.npos;p.prompt_free=name.find("prompt_free")!=name.npos;auto invocation=make_prompt_image_mapper(p)(envelope);if(!p.prompt_free)tensors.push_back(invocation.positional[p.visual?2:1].toTensor());}
  else if(r.model.task==TaskKind::pretrain_simmim){auto& e=r.extension;auto invocation=make_extended_image_mapper(r.model.task,data,false,e.model_patch_size,e.mask_patch_size,e.mask_ratio,r.seed)(envelope);tensors.push_back(invocation.positional[1].toTensor().to(torch::kInt64));}
  const auto prepared=std::chrono::steady_clock::now();auto model=read_artifact(root/"model.onnx",4ULL<<30);ort::TypedSessionOptions limits;limits.max_model_bytes=4ULL<<30;limits.max_input_bytes=limits.max_output_bytes=1ULL<<30;
  ort::TypedSession session(model,{.hex=m.at("sha256").get<std::string>()},limits);
  auto in_names=m.at("input_names").get<std::vector<std::string>>(),out_names=m.at("output_names").get<std::vector<std::string>>();auto in_shapes=m.at("input_shapes").get<std::vector<std::vector<I>>>(),out_shapes=m.at("output_shapes").get<std::vector<std::vector<I>>>();auto in_types=m.at("input_types").get<std::vector<int>>(),out_types=m.at("output_types").get<std::vector<int>>();
  const I batch_size=static_cast<I>(samples.size()),height=r.data.letterbox.canvas.height,width=r.data.letterbox.canvas.width;
  if(in_shapes.empty()||in_shapes[0]!=std::vector<I>{batch_size,3,height,width}||in_types.empty()||in_types[0]!=1)throw CompatibilityError("Extended bundle image signature mismatch");
  if(out_shapes.size()==1){std::vector<I> expected;
    if(r.model.task==TaskKind::classify||r.model.task==TaskKind::classify_multilabel)expected={batch_size,r.model.num_classes};
    else if(r.model.task==TaskKind::semantic_segment)expected={batch_size,r.model.num_classes,height,width};
    else if(r.model.task==TaskKind::depth)expected={batch_size,1,height,width};
    else if(r.model.task==TaskKind::pretrain_simmim)expected={batch_size,3,height,width};
    else if(r.model.task==TaskKind::reid)expected={batch_size,512};
    if(!expected.empty()&&(out_shapes[0]!=expected||out_types!=std::vector<int>{1}))throw CompatibilityError("Extended bundle task output signature mismatch");}
  if(in_names.size()!=tensors.size()||in_shapes.size()!=in_names.size()||in_types.size()!=in_names.size()||session.inputs().size()!=in_names.size()||out_names.size()!=out_shapes.size()||out_names.size()!=out_types.size()||session.outputs().size()!=out_names.size())throw CompatibilityError("Extended bundle I/O arity mismatch");
  std::vector<ort::TypedTensorInput> inputs;std::vector<std::vector<I>> shapes;shapes.reserve(tensors.size());
  for(std::size_t i=0;i<tensors.size();++i){tensors[i]=tensors[i].to(torch::kCPU).contiguous();const auto& t=tensors[i];shapes.emplace_back(t.sizes().begin(),t.sizes().end());const auto& s=session.inputs()[i];int dtype=t.scalar_type()==torch::kFloat32?1:t.scalar_type()==torch::kInt64?7:t.scalar_type()==torch::kInt32?6:0;
    if(shapes[i]!=in_shapes[i]||s.name!=in_names[i]||s.shape!=in_shapes[i]||static_cast<int>(s.type)!=in_types[i]||dtype!=in_types[i]||!s.leading_symbol.empty())throw CompatibilityError("Extended bundle fixed input shape/type mismatch (external proposals must have the exported row count)");
    ort::TypedInputValues values;if(dtype==1)values=std::span(t.const_data_ptr<float>(),static_cast<std::size_t>(t.numel()));else if(dtype==7)values=std::span(t.const_data_ptr<I>(),static_cast<std::size_t>(t.numel()));else values=std::span(t.const_data_ptr<std::int32_t>(),static_cast<std::size_t>(t.numel()));inputs.push_back({in_names[i],values,shapes[i]});}
  for(std::size_t i=0;i<out_names.size();++i){auto& s=session.outputs()[i];if(s.name!=out_names[i]||s.shape!=out_shapes[i]||static_cast<int>(s.type)!=out_types[i]||!s.leading_symbol.empty())throw CompatibilityError("Extended bundle fixed output signature mismatch");}
  auto loaded=std::chrono::steady_clock::now();auto values=session.run(inputs,stop);std::vector<Tensor> raw;for(auto& output:values)std::visit([&](auto& v){using T=typename std::decay_t<decltype(v)>::value_type;auto dtype=std::is_same_v<T,float>?torch::kFloat32:std::is_same_v<T,I>?torch::kInt64:torch::kInt32;raw.push_back(torch::from_blob(v.data(),output.shape,dtype).clone());},output.values);auto executed=std::chrono::steady_clock::now();
  std::array<I,2> size{batch.images.size(2),batch.images.size(3)};std::vector<ImageResult> results;
  if(name.starts_with("fast_rcnn_")||name=="faster_rcnn_resnet50_fpn"){Tensor rois,cls,delta;if(name.starts_with("fast_rcnn_")){if(raw.size()!=3)throw CompatibilityError("Fast R-CNN bundle output arity mismatch");rois=raw[0];cls=raw[1];delta=raw[2];}else{if(raw.size()!=4)throw CompatibilityError("RCNN bundle output arity mismatch");auto valid=raw[3]>0;rois=raw[0].index({valid});cls=raw[1].index({valid});delta=raw[2].index({valid});}auto sizes=torch::tensor(std::vector<float>{static_cast<float>(size[0]),static_cast<float>(size[1])}).view({1,2}).expand({batch.images.size(0),2});std::vector<Tensor> v{torch::empty({batch.images.size(0),0}),{},{},rois,cls,delta,{},{},sizes};results=decode_faster_rcnn_results(v,r.prediction);
  }else if(name.starts_with("rtdetr_")){if(raw.size()!=2)throw CompatibilityError("RT-DETR bundle output arity mismatch");std::vector<Tensor> v(8);v[0]=raw[0].unsqueeze(0);v[1]=raw[1].unsqueeze(0);results=decode_rtdetr(v,size,r.prediction);
  }else if(name.starts_with("fcos_")||name.starts_with("retinanet_")||name.starts_with("ssd300_")){auto kind=name.starts_with("fcos_")?DenseDetectionCriterion::fcos:name.starts_with("ssd300_")?DenseDetectionCriterion::ssd:DenseDetectionCriterion::retinanet_v1;if(kind!=DenseDetectionCriterion::fcos)raw.insert(raw.begin()+2,torch::empty({0}));results=decode_dense_detection(raw,size,kind,r.prediction);
  }else if(static_cast<unsigned>(r.model.task)>4){if(r.model.task==TaskKind::reid)raw.insert(raw.begin(),torch::zeros({batch.images.size(0),r.model.num_classes}));results=decode_extended_results(raw,size,r.model.task,r.prediction,r.extension.multilabel_thresholds);
  }else if(r.model.task==TaskKind::classify)results=decode_classification_results(raw[0],size,r.prediction);
  else{YoloTaskOptions t;t.task=r.model.task;t.detection.num_classes=r.model.num_classes;t.mask_channels=r.model.mask_channels;t.keypoints=r.model.keypoints;t.keypoint_dimensions=r.model.keypoint_dimensions;t.generation26=name.starts_with("yolo26");t.end_to_end=m.at("end_to_end").get<bool>();t.detection.reg_max=t.generation26?1:16;t.detection.strides=r.model.variant.find("p2")!=name.npos?std::vector<I>{4,8,16,32}:r.model.variant.find("p6")!=name.npos?std::vector<I>{8,16,32,64}:std::vector<I>{8,16,32};
    if(name.starts_with("yolov5")){OriginalV5TaskOptions o;o.num_classes=r.model.num_classes;if(r.model.variant=="p6"){o.strides={8,16,32,64};o.anchors={{19,27,44,40,38,94},{96,68,86,152,180,137},{140,301,303,264,238,542},{436,615,739,380,925,792}};o.objectness_balance={4,1,.25,.06};}results=decode_detection_results(decode_original_v5(raw[0],size,o),size,r.prediction);}else results=decode_yolo_results(raw,size,t,r.prediction);}
  if(results.size()!=samples.size())throw CompatibilityError("Extended decoder batch arity mismatch");J json=J::array();for(std::size_t i=0;i<results.size();++i){results[i].source_id=samples[i].image_path;if(!classify&&r.model.task!=TaskKind::pretrain_simmim)results[i]=restore_result(results[i],*batch.items[i].geometry,r.prediction.max_mask_elements);else results[i].extent={samples[i].width,samples[i].height};json.push_back(J::parse(result_json(results[i])));}
  auto end=std::chrono::steady_clock::now();auto ms=[](auto d){return std::chrono::duration<double,std::milli>(d).count();};return J{{"schema","vision.bundle_prediction/v2"},{"sha256",m.at("sha256")},{"results",json},{"timing_ms",{{"preprocess",ms(prepared-start)},{"session_load",ms(loaded-prepared)},{"model",ms(executed-loaded)},{"postprocess",ms(end-executed)}}}}.dump(2);
}
}
