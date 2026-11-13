// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/tracking/workflow.hpp>
#include <cmath>
#include <algorithm>
namespace jyppx::liuding::vision::tasks::tracking {
FramePrediction track_vision_frame(Tracker& tracker,TaskBinding& detector,
    const data::image::VisionSample& frame,const RecipeSpec& recipe,std::int64_t id,
    const TrackerOptions& options,TaskBinding* reid,const RecipeSpec* reid_recipe,
    const data::image::VisionSample* previous,std::stop_token stop){
  if(recipe.model.task!=TaskKind::detect||detector.capability.task!=TaskKind::detect)throw InvalidArgument("Tracking requires a detection binding");
  auto r=recipe;r.prediction.confidence=options.low_threshold;r.prediction.max_detections=options.max_detections;
  auto detection=predict_vision(detector,std::span(&frame,1),r,true,stop).at(0);Tensor features,motion;
  if(options.kind==TrackerKind::bot_sort){if(!reid||!reid_recipe||reid_recipe->model.task!=TaskKind::reid)throw InvalidArgument("BoT-SORT requires an explicit ReID binding and recipe");
    std::vector<Tensor> rows;for(std::int64_t i=0;i<detection.boxes.size(0);++i){if(stop.stop_requested())throw Cancelled("Tracking ReID crops cancelled");auto box=detection.boxes[i];auto x1=std::clamp(static_cast<std::int64_t>(std::floor(box[0].item<double>())),0LL,static_cast<std::int64_t>(frame.width)-1),y1=std::clamp(static_cast<std::int64_t>(std::floor(box[1].item<double>())),0LL,static_cast<std::int64_t>(frame.height)-1);auto x2=std::clamp(static_cast<std::int64_t>(std::ceil(box[2].item<double>())),x1+1,static_cast<std::int64_t>(frame.width)),y2=std::clamp(static_cast<std::int64_t>(std::ceil(box[3].item<double>())),y1+1,static_cast<std::int64_t>(frame.height));data::image::VisionSample crop;crop.image=frame.image.slice(1,y1,y2).slice(2,x1,x2);crop.width=static_cast<std::uint32_t>(x2-x1);crop.height=static_cast<std::uint32_t>(y2-y1);crop.pixels=frame.pixels;crop.image_path=frame.image_path+"#roi="+std::to_string(i);rows.push_back(predict_vision(*reid,std::span(&crop,1),*reid_recipe,false,stop).at(0).embedding);}
    if(!rows.empty())features=torch::stack(rows);if(options.global_motion_compensation)motion=previous?estimate_affine_motion(previous->image,frame.image):torch::tensor({1.,0.,0.,0.,1.,0.},torch::kFloat64).view({2,3});}
  auto tracks=tracker.step(detection,id,features,motion);return{std::move(detection),std::move(tracks)};
}
}
