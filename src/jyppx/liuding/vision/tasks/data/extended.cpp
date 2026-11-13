// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/data/extended.hpp>
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <bit>
#include <cmath>
#include <jyppx/liuding/foundation/random.hpp>
#include <set>
#include <unordered_map>
namespace jyppx::liuding::vision::tasks {
namespace {
namespace im=data::image;using J=nlohmann::json;using I=std::int64_t;
std::vector<std::byte> Read(const std::filesystem::path& path,std::uint64_t limit){auto n=std::filesystem::file_size(path);if(n>limit)throw ResourceExhausted("Task dataset file exceeds byte limit");
  std::ifstream f(path,std::ios::binary);if(!f)throw IoError("Cannot open task dataset file");std::vector<std::byte> b(n);f.read(reinterpret_cast<char*>(b.data()),n);if(!f)throw IoError("Cannot read task dataset file");return b;}
std::filesystem::path Resolve(const std::filesystem::path& root,const std::string& text){auto p=std::filesystem::weakly_canonical(root/std::filesystem::u8path(text));auto relative=p.lexically_relative(root);
  if(text.empty()||relative.empty()||relative.is_absolute()||*relative.begin()=="..")throw InvalidArgument("Task dataset path escapes image root");return p;}
void Keys(const J& j,std::initializer_list<const char*> keys){if(!j.is_object())throw InvalidArgument("Task manifest entry must be an object");std::set<std::string> allowed(keys.begin(),keys.end());for(auto& field:j.items())if(!allowed.contains(field.key()))throw InvalidArgument("Unknown task manifest field: "+field.key());}
Tensor Rows(const J& j,I columns){if(!j.is_array()||j.size()>65536)throw ResourceExhausted("Task tensor row budget exceeded");std::vector<float> v;
  for(const auto& row:j){if(!row.is_array()||row.size()!=columns)throw InvalidArgument("Task tensor column count mismatch");for(const auto& value:row)v.push_back(value.get<float>());}auto tensor=torch::tensor(v).view({static_cast<I>(j.size()),columns});if(!torch::isfinite(tensor).all().item<bool>())throw InvalidArgument("Nonfinite task tensor rows");return tensor;}
std::uint64_t Seed(const im::ImageBatchItem& item,std::uint64_t seed){auto id=item.key.id+":"+std::to_string(item.key.occurrence)+":"+std::to_string(item.key.view);auto h=sha256_hex(std::as_bytes(std::span(id)));return seed^std::stoull(h.substr(0,16),nullptr,16);}
Tensor Masks(const im::ImageBatch& b,I patch,I block,double ratio,std::uint64_t seed){std::vector<Tensor> masks;
  for(const auto& item:b.items)masks.push_back(simmim_mask({b.images.size(2),b.images.size(3)},patch,block,ratio,Seed(item,seed)));return torch::stack(masks).to(b.images.device());}
const im::VisionTaskPayload& Payload(const im::ImageBatchItem& item){if(!item.task_payload)throw InvalidArgument("Task requires supplementary targets");return *item.task_payload;}
}
TaskDatasetImport read_task_dataset(const std::filesystem::path& path,im::ImageDatasetOptions decoding,std::stop_token stop){
  auto bytes=Read(path,64ULL<<20);auto doc=J::parse(reinterpret_cast<const char*>(bytes.data()),reinterpret_cast<const char*>(bytes.data()+bytes.size()));
  Keys(doc,{"schema","image_root","categories","images","label_space_id","text_encoder_id","semantic_ignore","semantic_encoding","semantic_color_map","semantic_id_map","depth_representation","depth_unit_scale","proposal_generator","proposal_inventory","prompt_embeddings","prompt_dimension","text_prompts"});
  if(doc.at("schema")!="vision.dataset/v1")throw InvalidArgument("Unknown task dataset schema");
  auto root=std::filesystem::weakly_canonical(path.parent_path()/std::filesystem::u8path(doc.at("image_root").get<std::string>()));
  auto annotations=std::make_shared<im::AnnotationIndex>();
  for(const auto& c:doc.at("categories")){I id=annotations->categories.size();annotations->categories.push_back({.source_id=c.is_string()?id:c.at("source_id").get<I>(),.name=c.is_string()?c.get<std::string>():c.at("name").get<std::string>()});}
  if(annotations->categories.empty()||annotations->categories.size()>10000||!doc.at("images").is_array()||doc.at("images").empty()||doc.at("images").size()>1000000)throw InvalidArgument("Task dataset categories/image bounds invalid");
  struct Extra{std::shared_ptr<im::VisionTaskPayload> p;std::string mask,depth;std::vector<std::size_t> labels;};
  auto extras=std::make_shared<std::unordered_map<std::string,Extra>>();Sha256 digest;digest.update(bytes);std::set<std::string> sequences;std::vector<I> identities;std::uint64_t target_bytes=0,payload_bytes=0;
  Tensor shared_prompts;if(doc.contains("prompt_embeddings"))shared_prompts=Rows(doc["prompt_embeddings"],doc.at("prompt_dimension").get<I>());
  auto text_prompts=doc.value("text_prompts",std::vector<std::string>{});
  for(const auto& row:doc.at("images")){
    Keys(row,{"image","width","height","labels","identity","boxes","polygons","rles","proposals","visual_prompts","camera_intrinsics","sequence_id","camera_id","frame_id","timestamp_seconds","track_ids","semantic_mask","depth"});
    if(stop.stop_requested())throw Cancelled("Task dataset indexing cancelled");auto image=row.at("image").get<std::string>();Resolve(root,image);
    im::VisionAnnotationRecord a{.source_image_id=static_cast<I>(annotations->images.size()),.image_path=image,.width=row.at("width").get<std::uint32_t>(),.height=row.at("height").get<std::uint32_t>()};
    if(!a.width||!a.height||a.width>65536||a.height>65536||static_cast<std::uint64_t>(a.width)*a.height>268435456)throw InvalidArgument("Task image extent out of bounds");
    Extra e;e.p=std::make_shared<im::VisionTaskPayload>();auto& p=*e.p;
    p.label_space_id=doc.value("label_space_id",std::string{});p.text_encoder_id=doc.value("text_encoder_id",std::string{});
    p.semantic_ignore=doc.value("semantic_ignore",255LL);p.depth_unit_scale=doc.value("depth_unit_scale",1.);
    auto representation=doc.value("depth_representation",std::string("depth_meters"));
    if(representation=="depth_meters")p.depth_representation=im::DepthRepresentation::depth_meters;
    else if(representation=="inverse_depth")p.depth_representation=im::DepthRepresentation::inverse_depth;
    else if(representation=="relative_inverse_depth")p.depth_representation=im::DepthRepresentation::relative_inverse_depth;
    else throw InvalidArgument("Unknown depth representation");
    if(row.contains("labels")){auto y=row["labels"].get<std::vector<float>>();if(y.size()!=annotations->categories.size())throw InvalidArgument("Multilabel vector must cover the complete vocabulary");
      for(float value:y)if(!std::isfinite(value)||(value!=-1&&(value<0||value>1)))throw InvalidArgument("Labels accept -1 unknown or finite [0,1] probabilities");p.multilabel_values=torch::tensor(y);p.known_labels=p.multilabel_values>=0;p.multilabel_values=torch::where(p.known_labels,p.multilabel_values,torch::zeros_like(p.multilabel_values));}
    if(row.contains("identity")){auto id=row["identity"].get<I>();if(id<0||id>=static_cast<I>(annotations->categories.size()))throw InvalidArgument("Identity outside training class vocabulary");e.labels={static_cast<std::size_t>(id)};identities.push_back(id);}
    else identities.push_back(-1);
    if(row.contains("boxes")){auto boxes=Rows(row["boxes"],5);for(I i=0;i<boxes.size(0);++i){auto c=boxes[i][0].item<double>();if(c!=std::floor(c)||c<0||c>=annotations->categories.size())throw InvalidArgument("Object category outside vocabulary");
      a.objects.push_back({.source_id=i+1,.category=static_cast<std::size_t>(c),.box=im::ImageBox{boxes[i][1].item<double>(),boxes[i][2].item<double>(),boxes[i][3].item<double>(),boxes[i][4].item<double>()}});}}
    if(row.contains("polygons")){const auto& objects=row["polygons"];if(!objects.is_array()||objects.size()!=a.objects.size())throw InvalidArgument("Instance polygons must align with box object rows");std::size_t points=0;for(std::size_t i=0;i<objects.size();++i){if(!objects[i].is_array())throw InvalidArgument("Object polygons must be arrays");for(const auto& polygon:objects[i]){if(!polygon.is_array()||polygon.size()<3||polygon.size()>65536||points+polygon.size()>1000000)throw ResourceExhausted("Instance polygon point bounds exceeded");points+=polygon.size();std::vector<im::ImagePoint> ring;for(const auto& point:polygon){if(!point.is_array()||point.size()!=2)throw InvalidArgument("Polygon point must be xy");double x=point[0].get<double>(),y=point[1].get<double>();if(!std::isfinite(x)||!std::isfinite(y))throw InvalidArgument("Nonfinite polygon coordinate");ring.push_back({x,y});}a.objects[i].polygons.push_back(std::move(ring));}}}
    if(row.contains("rles")){const auto& objects=row["rles"];if(!objects.is_array()||objects.size()!=a.objects.size())throw InvalidArgument("Instance RLE must align with box object rows");for(std::size_t i=0;i<objects.size();++i)if(!objects[i].is_null()){im::ImageMaskRle rle;rle.width=a.width;rle.height=a.height;rle.counts=objects[i].get<std::vector<std::uint64_t>>();if(rle.counts.size()>1000000)throw ResourceExhausted("Instance RLE run bound exceeded");a.objects[i].rle=std::move(rle);}}
    if(row.contains("proposals")){p.proposals=Rows(row["proposals"],4);p.proposal_generator=doc.at("proposal_generator").get<std::string>();p.proposal_inventory=doc.at("proposal_inventory").get<std::string>();}
    if(row.contains("visual_prompts"))p.visual_prompts=Rows(row["visual_prompts"],5);
    p.prompt_embeddings=shared_prompts;p.text_prompts=text_prompts;
    if(row.contains("camera_intrinsics"))p.camera_intrinsics=Rows(row["camera_intrinsics"],3);
    p.sequence_id=row.value("sequence_id",std::string{});p.camera_id=row.value("camera_id",std::string{});p.frame_id=row.value("frame_id",0LL);p.timestamp_seconds=row.value("timestamp_seconds",0.);
    if(!p.sequence_id.empty())sequences.insert(p.sequence_id);
    if(row.contains("track_ids"))p.track_ids=torch::tensor(row["track_ids"].get<std::vector<I>>(),torch::kInt64);
    e.mask=row.value("semantic_mask",std::string{});e.depth=row.value("depth",std::string{});
    for(const auto& file:{e.mask,e.depth})if(!file.empty()){auto target=Read(Resolve(root,file),1ULL<<30);if(target.size()>(8ULL<<30)-target_bytes)throw ResourceExhausted("Aggregate task target inventory exceeds 8 GiB");target_bytes+=target.size();digest.update(target);}
    auto size=p.validate();if(size>(2ULL<<30)-payload_bytes)throw ResourceExhausted("Task manifest logical payload exceeds 2 GiB");payload_bytes+=size;if(!extras->emplace(image,std::move(e)).second)throw InvalidArgument("Duplicate task dataset image path");annotations->images.push_back(std::move(a));
  }
  im::MaskDecodeOptions mask_options;mask_options.ignore_id=doc.value("semantic_ignore",255LL);
  auto encoding=doc.value("semantic_encoding",std::string("class_index"));if(encoding=="palette_index")mask_options.encoding=im::MaskEncoding::palette_index;else if(encoding=="rgb_mapping"){mask_options.encoding=im::MaskEncoding::rgb_mapping;for(const auto& row:doc.at("semantic_color_map")){Keys(row,{"color","id"});mask_options.colors.push_back({row.at("color").get<std::array<std::uint8_t,3>>(),row.at("id").get<I>()});}}else if(encoding!="class_index")throw InvalidArgument("Unknown semantic mask encoding");
  Tensor id_map;if(doc.contains("semantic_id_map")){std::vector<I> map(65536,*mask_options.ignore_id);std::set<I> seen;for(const auto& row:doc["semantic_id_map"]){if(!row.is_array()||row.size()!=2)throw InvalidArgument("Semantic ID map expects source/train pairs");I from=row[0].get<I>(),to=row[1].get<I>();if(from<0||from>=65536||!seen.insert(from).second||(to!=*mask_options.ignore_id&&(to<0||to>=static_cast<I>(annotations->categories.size()))))throw InvalidArgument("Invalid semantic ID mapping");map[from]=to;}id_map=torch::tensor(map,torch::kInt64);}
  auto previous=decoding.decoded_sample_transform;
  decoding.decoded_sample_transform=[extras,root,previous,mask_options,id_map](im::VisionSample s,const auto& file,auto token){
    const auto& e=extras->at(s.image_path);auto p=std::make_shared<im::VisionTaskPayload>(*e.p);s.labels=e.labels;s.label_categories=s.categories;
    if(!e.mask.empty()){s.mask_target=im::read_mask_file(Resolve(root,e.mask),mask_options,token);s.target_path=e.mask;
      if(id_map.defined()){auto ids=s.mask_target->ids;if(ids.min().template item<I>()<0||ids.max().template item<I>()>=65536)throw InvalidArgument("Semantic source IDs exceed explicit 16-bit remapping domain");s.mask_target->ids=id_map.index_select(0,ids.flatten()).view_as(ids);}}

    if(!e.depth.empty()){
      if(std::endian::native!=std::endian::little)throw NotImplemented("Task safetensors reader needs little endian host");
      auto data=Read(Resolve(root,e.depth),1ULL<<30);interop::SafeTensorsLimits limits;limits.max_file_bytes=1ULL<<30;limits.max_payload_bytes=1ULL<<30;limits.max_tensor_bytes=1ULL<<30;
      interop::SafeTensorsView view(data,limits);for(const auto& info:view.directory().tensors){if(info.name!="depth"&&info.name!="valid")throw InvalidArgument("Depth archive only accepts depth and valid");
        if(info.shape!=std::vector<std::uint64_t>{1,s.height,s.width}||info.dtype!=(info.name=="depth"?interop::SafeTensorDtype::f32:interop::SafeTensorDtype::boolean))throw InvalidArgument("Depth archive shape/dtype mismatch");
        auto raw=view.tensor_bytes(info.name);auto tensor=torch::from_blob(const_cast<std::byte*>(raw.data()),{1,s.height,s.width},info.name=="depth"?torch::kFloat32:torch::kBool).clone();if(info.name=="depth")p->depth=tensor;else p->depth_valid=tensor;}
      if(!p->depth.defined()||!p->depth_valid.defined())throw InvalidArgument("Depth archive needs depth and valid tensors");
    }
    p->validate();s.task_payload=std::move(p);return previous?previous(std::move(s),file,token):std::move(s);
  };
  auto identity=sha256_digest_hex(digest.finish());annotations->source_sha256=identity;auto categories=annotations->categories;
  auto dataset=std::make_shared<im::ImageDataset>(root,std::move(*annotations),decoding);
  auto representation=doc.value("depth_representation",std::string("depth_meters"));
  return{dataset,root,std::move(categories),identity,representation=="depth_meters"?im::DepthRepresentation::depth_meters:representation=="inverse_depth"?im::DepthRepresentation::inverse_depth:im::DepthRepresentation::relative_inverse_depth,doc.value("semantic_ignore",255LL),std::vector<std::string>(sequences.begin(),sequences.end()),std::move(identities)};
}
training::TargetMapper make_extended_target_mapper(TaskKind task,I classes,I patch,I block,double ratio,std::uint64_t seed){return[=](const data::BatchEnvelope& e)->std::optional<c10::IValue>{auto b=im::image_batch_from_value(e.value);
  if(task==TaskKind::semantic_segment){if(!b.masks || !b.masks->defined())throw InvalidArgument("Semantic task needs discrete masks");return b.masks->to(b.images.device());}
  if(task==TaskKind::reid)return b.labels.to(b.images.device());
  if(task==TaskKind::pretrain_simmim)return c10::ivalue::Tuple::create({b.images,Masks(b,patch,block,ratio,core::derive_seed(seed,e.epoch))});
  std::vector<Tensor> first,second;
  for(const auto& item:b.items){const auto& p=Payload(item);
    if(task==TaskKind::classify_multilabel){if(p.multilabel_values.numel()!=classes)throw InvalidArgument("Multilabel class dimension mismatch");first.push_back(p.multilabel_values);second.push_back(p.known_labels);}
    else if(task==TaskKind::depth){if(!p.depth.defined())throw InvalidArgument("Depth sample lacks numeric depth");first.push_back(p.depth*p.depth_unit_scale);second.push_back(p.depth_valid);}
    else throw InvalidArgument("Unsupported supplementary target mapper");}
  return c10::ivalue::Tuple::create({torch::stack(first).to(b.images.device()),torch::stack(second).to(b.images.device())});};}
training::BatchMapper make_extended_image_mapper(TaskKind task,const YoloDataOptions& data,bool proposals,I patch,I block,double ratio,std::uint64_t seed){return[=](const data::BatchEnvelope& e){auto b=im::image_batch_from_value(e.value);
  if(task==TaskKind::pretrain_simmim)return nn::Invocation{{b.images,Masks(b,patch,block,ratio,core::derive_seed(seed,e.epoch))},{}};
  if(!proposals)return nn::Invocation{{b.images},{}};
  std::vector<Tensor> rows;std::vector<float> extents;I i=0;
  for(const auto& item:b.items){const auto& p=Payload(item);if(!p.proposals.defined()||p.proposal_generator.empty()||p.proposal_inventory.empty())throw InvalidArgument("Fast R-CNN requires explicit proposal generator and inventory");
    auto boxes=p.proposals.to(b.images.device());rows.push_back(torch::cat({torch::full({boxes.size(0),1},static_cast<double>(i++),boxes.options()),boxes},1));extents.push_back(item.valid_extent.height);extents.push_back(item.valid_extent.width);}
  return nn::Invocation{{b.images,torch::cat(rows),build_yolo_targets(b,data),torch::tensor(extents,b.images.options()).view({i,2})},{}};};}
}
