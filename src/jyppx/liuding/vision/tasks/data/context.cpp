// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/data/context.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#if defined(JYPPX_VISION_WITH_EXTENDED)
#include <jyppx/liuding/vision/models/extended/text.hpp>
#endif
#include <nlohmann/json.hpp>
#include <set>
namespace jyppx::liuding::vision::tasks {
namespace {
using J=nlohmann::json;using I=std::int64_t;
Tensor Rows(const J& a,I width){if(!a.is_array()||a.size()>65536||width<1||width>4096)throw InvalidArgument("Input context tensor bounds invalid");std::vector<float> data;for(const auto& row:a){if(!row.is_array()||row.size()!=width)throw InvalidArgument("Input context tensor width mismatch");for(auto& v:row)data.push_back(v.get<float>());}return torch::tensor(data).view({static_cast<I>(a.size()),width});}
J Rows(const Tensor& source){J result=J::array();if(!source.defined())return result;auto x=source.to(torch::kCPU,torch::kFloat32).contiguous();for(I row=0;row<x.size(0);++row){auto p=x[row].const_data_ptr<float>();result.push_back(std::vector<float>(p,p+x.size(1)));}return result;}
}
std::shared_ptr<const data::image::VisionTaskPayload> parse_input_context(std::string_view text){if(text.size()>(32U<<20))throw ResourceExhausted("Input context exceeds 32 MiB");auto j=J::parse(text);std::set<std::string> allowed{"schema","label_space_id","text_encoder_id","text_prompts","prompt_embeddings","visual_prompts","proposals","proposal_generator","proposal_inventory"};for(auto& field:j.items())if(!allowed.contains(field.key()))throw InvalidArgument("Unknown input context field: "+field.key());if(j.at("schema")!="vision.input_context/v1")throw CompatibilityError("Unknown input context schema");auto p=std::make_shared<data::image::VisionTaskPayload>();p->label_space_id=j.value("label_space_id",std::string{});p->text_encoder_id=j.value("text_encoder_id",std::string{});p->text_prompts=j.value("text_prompts",std::vector<std::string>{});p->proposal_generator=j.value("proposal_generator",std::string{});p->proposal_inventory=j.value("proposal_inventory",std::string{});if(j.contains("prompt_embeddings"))p->prompt_embeddings=Rows(j["prompt_embeddings"],512);if(j.contains("visual_prompts"))p->visual_prompts=Rows(j["visual_prompts"],5);if(j.contains("proposals"))p->proposals=Rows(j["proposals"],4);p->validate(32U<<20);return p;}
std::string input_context_json(const data::image::VisionTaskPayload& p){p.validate(32U<<20);J j{{"schema","vision.input_context/v1"},{"label_space_id",p.label_space_id},{"text_encoder_id",p.text_encoder_id},{"text_prompts",p.text_prompts}};if(p.prompt_embeddings.defined())j["prompt_embeddings"]=Rows(p.prompt_embeddings);if(p.visual_prompts.defined())j["visual_prompts"]=Rows(p.visual_prompts);if(p.proposals.defined()){j["proposals"]=Rows(p.proposals);j["proposal_generator"]=p.proposal_generator;j["proposal_inventory"]=p.proposal_inventory;}return j.dump(2);}
std::shared_ptr<const data::image::VisionTaskPayload> prepare_text_context(const RecipeSpec& r,std::span<const std::string> prompts){
#if defined(JYPPX_VISION_WITH_EXTENDED)
  namespace ex=models::extended;if(prompts.empty()||prompts.size()!=static_cast<std::size_t>(r.model.num_classes)||prompts.size()>1024||r.extension.tokenizer_merges.empty()||r.extension.text_encoder_weights.empty())throw InvalidArgument("Text context needs complete vocabulary and local tokenizer/text encoder assets");
  ex::ClipTokenizer tokenizer(r.extension.tokenizer_merges,r.extension.tokenizer_sha256);auto encoder=ex::adapt(ex::ClipText());auto options=r.extension.teacher_initialization;options={};options.weights=r.extension.text_encoder_weights;options.sha256=r.extension.text_encoder_sha256;initialize_vision(encoder,options,"clip_text_vit_b32");encoder.state().eval();torch::NoGradGuard guard;auto p=std::make_shared<data::image::VisionTaskPayload>();p->text_prompts.assign(prompts.begin(),prompts.end());p->text_encoder_id=tokenizer.identity()+":clip-text-b32:"+r.extension.text_encoder_sha256;p->prompt_embeddings=encoder.invoke(nn::Invocation{{tokenizer.encode(p->text_prompts)},{}}).value.toTensor();auto identity=J{{"encoder",p->text_encoder_id},{"literal_prompts",p->text_prompts},{"normalization","unit-l2"}}.dump();p->label_space_id=sha256_hex(std::as_bytes(std::span(identity)));p->validate();return p;
#else
  (void)r;(void)prompts;throw CompatibilityError("Native text encoding requires models_vision");
#endif
}
}
