// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <numeric>
namespace jyppx::liuding::vision::tasks::detail {
namespace {
using N = torch::jit::Node;
using V = torch::jit::Value;
using G = torch::jit::Graph;
using I = std::int64_t;
c10::IValue Literal(V* v) {
  if (auto value = torch::jit::toIValue(v)) return *value;
  if (v->node()->kind() == torch::jit::prim::ListConstruct) {
    std::vector<I> values;
    for (auto* entry : v->node()->inputs())
      values.push_back(Literal(entry).toInt());
    return values;
  }
  throw CompatibilityError(
      "Vision ONNX profile needs a fixed scalar/shape option");
}
double Number(V* v){auto value=Literal(v);if(value.isInt())return static_cast<double>(value.toInt());if(value.isDouble())return value.toDouble();throw CompatibilityError("Expected a constant ONNX numeric option");}
V* Constant(G& g, N& before, const Tensor& tensor) {
  auto* n = g.create(c10::Symbol::onnx("Constant"), {}, 1);
  n->t_(c10::Symbol::attr("value"), tensor);
  n->output()->setType(c10::TensorType::create(tensor));
  n->insertBefore(&before);
  return n->output();
}
V* Integers(G& g, N& n, std::vector<I> v) {
  return Constant(g, n, torch::tensor(v, torch::kInt64));
}
N* Op(G& g, N& before, const char* kind, std::vector<V*> inputs,
      std::size_t count = 1) {
  auto* n = g.create(c10::Symbol::onnx(kind), inputs, count);
  n->insertBefore(&before);
  for (auto* v : n->outputs()) v->setType(before.output(0)->type());
  return n;
}
std::vector<I> Shape(V* value) {
  auto type = value->type()->cast<c10::TensorType>();
  auto shape = type ? type->sizes().concrete_sizes() : std::nullopt;
  if (!shape)
    throw CompatibilityError(
        "Vision ONNX fixed profile needs concrete tensor extents");
  return *shape;
}
bool IsNone(V* x){auto value=torch::jit::toIValue(x);return value&&value->isNone();}
V* Scalar(G& g,N& n,V* value,V* like) {
  if(value->type()->cast<c10::TensorType>())return value;
  auto v=Literal(value);auto type=like->type()->cast<c10::TensorType>();
  auto dtype=type&&type->scalarType()?*type->scalarType():torch::kFloat32;
  if(v.isNone())throw CompatibilityError("Unexpected empty ONNX scalar");
  return Constant(g,n,torch::scalar_tensor(v.isBool()?c10::Scalar(v.toBool()):v.isInt()?c10::Scalar(v.toInt()):c10::Scalar(v.toDouble()),torch::TensorOptions().dtype(dtype)));
}
std::vector<V*> Flatten(N& n, G& g) {
  auto shape = Shape(n.input(0));
  I begin = Literal(n.input(1)).toInt(), end = Literal(n.input(2)).toInt();
  const auto rank = static_cast<I>(shape.size());
  if (begin < 0) begin += rank;
  if (end < 0) end += rank;
  if (begin < 0 || end < begin || end >= rank)
    throw CompatibilityError("Invalid ONNX flatten dimensions");
  I product = 1;
  for (I i = begin; i <= end; ++i) product *= shape[i];
  shape.erase(shape.begin() + begin, shape.begin() + end + 1);
  shape.insert(shape.begin() + begin, product);
  return {Op(g, n, "Reshape", {n.input(0), Integers(g, n, shape)})->output()};
}
std::vector<V*> Slice(N& n, G& g) {
  return {
      Op(g, n, "Slice",
         {n.input(0), Integers(g, n, {(IsNone(n.input(2))?0:Literal(n.input(2)).toInt())}),
          Integers(g, n, {(IsNone(n.input(3))?INT64_MAX:Literal(n.input(3)).toInt())}),
          Integers(g, n, {Literal(n.input(1)).toInt()}),
          Integers(g, n,
                   {n.inputs().size() > 4 ? Literal(n.input(4)).toInt() : 1})})
          ->output()};
}
std::vector<V*> Select(N& n, G& g) {
  auto* result =
      Op(g, n, "Gather",
         {n.input(0), Constant(g, n,
                               torch::scalar_tensor(Literal(n.input(2)).toInt(),
                                                    torch::kInt64))});
  result->i_(c10::Symbol::attr("axis"), Literal(n.input(1)).toInt());
  return {result->output()};
}
std::vector<V*> Split(N& n, G& g) {
  const auto axis = Literal(n.input(2)).toInt();
  auto shape = Shape(n.input(0));
  const I size =
      shape.at(static_cast<std::size_t>((axis + shape.size()) % shape.size()));
  std::vector<I> sizes;
  auto requested = Literal(n.input(1));
  if (requested.isIntList())
    sizes = requested.toIntVector();
  else {
    const I count = requested.toInt();
    if (count < 1) throw CompatibilityError("Invalid ONNX split size");
    const I step = std::string_view(n.kind().toQualString()) == "aten::chunk"
                       ? (size + count - 1) / count
                       : count;
    for (I start = 0; start < size; start += step)
      sizes.push_back((std::min)(step, size - start));
  }
  auto* split =
      Op(g, n, "Split", {n.input(0), Integers(g, n, sizes)}, sizes.size());
  split->i_(c10::Symbol::attr("axis"), axis);
  for (auto* v : split->outputs()) v->setType(c10::TensorType::get());
  auto* list = g.create(torch::jit::prim::ListConstruct, split->outputs(), 1);
  list->output()->setType(n.output()->type());
  list->insertBefore(&n);
  return {list->output()};
}
std::vector<V*> Unpack(N& n, G&) {
  auto* list = n.input()->node();
  if (list->kind() != torch::jit::prim::ListConstruct ||
      list->inputs().size() != n.outputs().size())
    throw CompatibilityError("ONNX list arity differs from fixed profile");
  return {list->inputs().begin(), list->inputs().end()};
}
std::vector<V*> Resize(N& n, G& g) {
  const auto mode = std::string_view(n.kind().toQualString());
  auto size = Literal(n.input(1));
  auto shape = Shape(n.input(0));
  if (!size.isIntList())
    throw CompatibilityError(
        "Vision ONNX resize requires explicit output_size");
  auto spatial = size.toIntVector();
  shape.erase(shape.begin() + 2, shape.end());
  shape.insert(shape.end(), spatial.begin(), spatial.end());
  auto* result =
      Op(g, n, "Resize",
         {n.input(0), Constant(g, n, torch::empty({0}, torch::kFloat32)),
          Constant(g, n, torch::empty({0}, torch::kFloat32)),
          Integers(g, n, shape)});
  const bool nearest = mode.find("nearest") != std::string_view::npos;
  result->s_(c10::Symbol::attr("mode"), nearest ? "nearest" : "linear");
  const bool aligned = !nearest && n.inputs().size() > 2 &&
                       Literal(n.input(2)).isBool() &&
                       Literal(n.input(2)).toBool();
  result->s_(c10::Symbol::attr("coordinate_transformation_mode"),
             nearest   ? "asymmetric"
             : aligned ? "align_corners"
                       : "half_pixel");
  if (nearest) result->s_(c10::Symbol::attr("nearest_mode"), "floor");
  return {result->output()};
}
}  // namespace
std::map<std::string, onnx_export::OperatorLowering> vision_onnx_lowerings() {
  std::map<std::string, onnx_export::OperatorLowering> result;
  result["aten::flatten"] = Flatten;
  result["aten::slice"] = Slice;
  result["aten::select"] = Select;
  for (auto s : {"aten::split", "aten::split_with_sizes", "aten::chunk"})
    result[s] = Split;
  result["prim::ListUnpack"] = Unpack;
  result["prim::TupleUnpack"] = Unpack;
  for (auto s : {"aten::upsample_nearest2d", "aten::upsample_bilinear2d"})
    result[s] = Resize;
  for (auto s : {"aten::relu_", "aten::silu_"})
    result[s] = [s](N& n, G& g) {
      if (std::string_view(s) == "aten::relu_")
        return std::vector<V*>{Op(g, n, "Relu", {n.input(0)})->output()};
      auto* sigmoid = Op(g, n, "Sigmoid", {n.input(0)})->output();
      return std::vector<V*>{Op(g, n, "Mul", {n.input(0), sigmoid})->output()};
    };
  result["aten::expand"] = [](N& n, G& g) {
    auto shape = Literal(n.input(1)).toIntVector();
    auto source = Shape(n.input(0));
    for (std::size_t i = 0; i < shape.size(); ++i)
      if (shape[i] < 0) shape[i] = source.at(i + source.size() - shape.size());
    return std::vector<V*>{
        Op(g, n, "Expand", {n.input(0), Integers(g, n, shape)})->output()};
  };
  result["aten::index_select"] = [](N& n, G& g) {
    auto* p = Op(g, n, "Gather", {n.input(0), n.input(2)});
    p->i_(c10::Symbol::attr("axis"), Literal(n.input(1)).toInt());
    return std::vector<V*>{p->output()};
  };
  for (auto name : {"aten::zeros", "aten::ones", "aten::full",
                    "aten::zeros_like", "aten::ones_like", "aten::full_like"})
    result[name] = [name](N& n, G& g) {
      const std::string_view tag(name);
      auto type = n.output()->type()->cast<c10::TensorType>();
      if (!type || !type->scalarType())
        throw CompatibilityError("Fixed tensor factory needs an explicit dtype");
      const auto fill = tag.starts_with("aten::full") ? Number(n.input(1)) :
                        tag.starts_with("aten::ones") ? 1.0 : 0.0;
      auto tensor = torch::full(Shape(n.output()), fill,
                               torch::TensorOptions().dtype(*type->scalarType()));
      return std::vector<V*>{Constant(g, n, tensor)};
    };
  result["aten::roll"] = [](N& n,G& g){auto shape=Shape(n.input(0));auto shifts=Literal(n.input(1)).toIntVector(),axes=Literal(n.input(2)).toIntVector();if(shifts.size()!=axes.size())throw CompatibilityError("Fixed ONNX roll requires explicit axes");auto x=n.input(0);for(std::size_t i=0;i<axes.size();++i){auto axis=axes[i];if(axis<0)axis+=shape.size();if(axis<0||axis>=static_cast<I>(shape.size()))throw CompatibilityError("Invalid roll axis");auto width=shape[axis];if(width<1)throw CompatibilityError("ONNX roll requires nonempty axes");auto shift=(shifts[i]%width+width)%width;if(!shift)continue;auto a=Op(g,n,"Slice",{x,Integers(g,n,{width-shift}),Integers(g,n,{width}),Integers(g,n,{axis})})->output();auto b=Op(g,n,"Slice",{x,Integers(g,n,{0}),Integers(g,n,{width-shift}),Integers(g,n,{axis})})->output();auto concat=Op(g,n,"Concat",{a,b});concat->i_(c10::Symbol::attr("axis"),axis);x=concat->output();}return std::vector<V*>{x};};
  result["aten::reshape_as"] = [](N& n,G& g){return std::vector<V*>{Op(g,n,"Reshape",{n.input(0),Integers(g,n,Shape(n.input(1)))})->output()};};
  result["aten::square"] = [](N& n,G& g){return std::vector<V*>{Op(g,n,"Mul",{n.input(0),n.input(0)})->output()};};
  result["aten::rsqrt"] = [](N& n,G& g){return std::vector<V*>{Op(g,n,"Reciprocal",{Op(g,n,"Sqrt",{n.input(0)})->output()})->output()};};
  result["aten::reciprocal"] = [](N& n,G& g){return std::vector<V*>{Op(g,n,"Reciprocal",{n.input(0)})->output()};};
  result["aten::rsub"] = [](N& n,G& g){auto x=n.input(0);if(n.inputs().size()>2 && Number(n.input(2))!=1.)x=Op(g,n,"Mul",{x,Scalar(g,n,n.input(2),x)})->output();return std::vector<V*>{Op(g,n,"Sub",{Scalar(g,n,n.input(1),n.input(0)),x})->output()};};
  for(auto name:{"aten::clamp","aten::clamp_min","aten::clamp_max"})result[name]=[name](N& n,G& g){auto x=n.input(0);std::string_view tag(name);if(tag!="aten::clamp_max"&&!IsNone(n.input(1)))x=Op(g,n,"Max",{x,Scalar(g,n,n.input(1),n.input(0))})->output();if(tag!="aten::clamp_min"){auto i=tag=="aten::clamp"?2:1;if(!IsNone(n.input(i)))x=Op(g,n,"Min",{x,Scalar(g,n,n.input(i),n.input(0))})->output();}return std::vector<V*>{x};};
  for(auto pair:{std::pair{"aten::ge","GreaterOrEqual"},std::pair{"aten::le","LessOrEqual"},std::pair{"aten::__and__","And"},std::pair{"aten::bitwise_and","And"},std::pair{"aten::__or__","Or"},std::pair{"aten::bitwise_or","Or"}})result[pair.first]=[op=pair.second](N& n,G& g){return std::vector<V*>{Op(g,n,op,{n.input(0),Scalar(g,n,n.input(1),n.input(0))})->output()};};
  for(auto name:{"aten::bitwise_not","aten::logical_not"})result[name]=[](N& n,G& g){return std::vector<V*>{Op(g,n,"Not",{n.input(0)})->output()};};
  result["aten::where"] = [](N& n,G& g){return std::vector<V*>{Op(g,n,"Where",{n.input(0),Scalar(g,n,n.input(1),n.input(2)),Scalar(g,n,n.input(2),n.input(1))})->output()};};
  result["aten::masked_fill"] = [](N& n,G& g){return std::vector<V*>{Op(g,n,"Where",{n.input(1),Scalar(g,n,n.input(2),n.input(0)),n.input(0)})->output()};};
  result["aten::gather"] = [](N& n,G& g){auto p=Op(g,n,"GatherElements",{n.input(0),n.input(2)});p->i_(c10::Symbol::attr("axis"),Literal(n.input(1)).toInt());return std::vector<V*>{p->output()};};
  result["aten::topk"] = [](N& n,G& g){auto p=Op(g,n,"TopK",{n.input(0),Integers(g,n,{Literal(n.input(1)).toInt()})},2);p->i_(c10::Symbol::attr("axis"),Literal(n.input(2)).toInt());p->i_(c10::Symbol::attr("largest"),Literal(n.input(3)).toBool());p->i_(c10::Symbol::attr("sorted"),Literal(n.input(4)).toBool());return std::vector<V*>{p->output(0),p->output(1)};};
  for(auto name:{"aten::max","aten::min"})result[name]=[name](N& n,G& g){bool max=std::string_view(name)=="aten::max";std::vector<I> axes;bool keep=false;if(n.inputs().size()>1){axes={Literal(n.input(1)).toInt()};if(n.inputs().size()>2)keep=Literal(n.input(2)).toBool();}auto p=Op(g,n,max?"ReduceMax":"ReduceMin",{n.input(0)});if(!axes.empty())p->is_(c10::Symbol::attr("axes"),axes);p->i_(c10::Symbol::attr("keepdims"),keep);if(n.outputs().size()==1)return std::vector<V*>{p->output()};auto arg=Op(g,n,max?"ArgMax":"ArgMin",{n.input(0)});arg->i_(c10::Symbol::attr("axis"),axes[0]);arg->i_(c10::Symbol::attr("keepdims"),keep);return std::vector<V*>{p->output(),arg->output()};};
  result["aten::grid_sampler"] = [](N& n,G& g){auto p=Op(g,n,"GridSample",{n.input(0),n.input(1)});auto mode=Literal(n.input(2)).toInt(),padding=Literal(n.input(3)).toInt();if(mode<0||mode>2||padding<0||padding>2)throw CompatibilityError("Unsupported grid sampler options");p->s_(c10::Symbol::attr("mode"),mode==0?"bilinear":mode==1?"nearest":"bicubic");p->s_(c10::Symbol::attr("padding_mode"),padding==0?"zeros":padding==1?"border":"reflection");p->i_(c10::Symbol::attr("align_corners"),Literal(n.input(4)).toBool());return std::vector<V*>{p->output()};};
  result["aten::grid_sampler_2d"]=result["aten::grid_sampler"];
  result["aten::pixel_shuffle"] = [](N& n,G& g){auto p=Op(g,n,"DepthToSpace",{n.input(0)});p->i_(c10::Symbol::attr("blocksize"),Literal(n.input(1)).toInt());p->s_(c10::Symbol::attr("mode"),"CRD");return std::vector<V*>{p->output()};};
  result["aten::constant_pad_nd"] = [](N& n,G& g){auto pad=Literal(n.input(1)).toIntVector();auto shape=Shape(n.input(0));std::vector<I> pads(shape.size()*2);for(std::size_t i=0;i<pad.size()/2;++i){pads[shape.size()-1-i]=pad[i*2];pads[shape.size()*2-1-i]=pad[i*2+1];}auto p=Op(g,n,"Pad",{n.input(0),Integers(g,n,pads),Scalar(g,n,n.input(2),n.input(0))});p->s_(c10::Symbol::attr("mode"),"constant");return std::vector<V*>{p->output()};};
  result["aten::avg_pool2d"] = [](N& n,G& g){if(n.inputs().size()>6&&!IsNone(n.input(6)))throw CompatibilityError("ONNX average pooling divisor override is unsupported");auto p=Op(g,n,"AveragePool",{n.input(0)});auto kernel=Literal(n.input(1)).toIntVector(),stride=Literal(n.input(2)).toIntVector(),pads=Literal(n.input(3)).toIntVector();if(stride.empty())stride=kernel;auto repeated=pads;pads.insert(pads.end(),repeated.begin(),repeated.end());p->is_(c10::Symbol::attr("kernel_shape"),kernel);p->is_(c10::Symbol::attr("strides"),stride);p->is_(c10::Symbol::attr("pads"),pads);p->i_(c10::Symbol::attr("ceil_mode"),Literal(n.input(4)).toBool());p->i_(c10::Symbol::attr("count_include_pad"),Literal(n.input(5)).toBool());return std::vector<V*>{p->output()};};
  result["aten::group_norm"] = [](N& n,G& g){auto shape=Shape(n.input(0));I groups=Literal(n.input(1)).toInt();auto x=Op(g,n,"Reshape",{n.input(0),Integers(g,n,{shape[0],groups,-1})})->output();auto mean=Op(g,n,"ReduceMean",{x});mean->is_(c10::Symbol::attr("axes"),{2});mean->i_(c10::Symbol::attr("keepdims"),1);auto centered=Op(g,n,"Sub",{x,mean->output()})->output();auto variance=Op(g,n,"ReduceMean",{Op(g,n,"Mul",{centered,centered})->output()});variance->is_(c10::Symbol::attr("axes"),{2});variance->i_(c10::Symbol::attr("keepdims"),1);auto eps=Constant(g,n,torch::scalar_tensor(Number(n.input(4)),torch::kFloat32));auto denominator=Op(g,n,"Sqrt",{Op(g,n,"Add",{variance->output(),eps})->output()})->output();auto y=Op(g,n,"Reshape",{Op(g,n,"Div",{centered,denominator})->output(),Integers(g,n,shape)})->output();std::vector<I> affine(shape.size(),1);affine[1]=shape[1];if(!IsNone(n.input(2)))y=Op(g,n,"Mul",{y,Op(g,n,"Reshape",{n.input(2),Integers(g,n,affine)})->output()})->output();if(!IsNone(n.input(3)))y=Op(g,n,"Add",{y,Op(g,n,"Reshape",{n.input(3),Integers(g,n,affine)})->output()})->output();return std::vector<V*>{y};};
  result["aten::softplus"] = [](N& n,G& g){auto beta=Number(n.input(1)),threshold=Number(n.input(2));auto scalar=[&](double v){return Constant(g,n,torch::scalar_tensor(v,torch::kFloat32));};auto x=Op(g,n,"Mul",{n.input(0),scalar(beta)})->output();auto nonlinear=Op(g,n,"Div",{Op(g,n,"Softplus",{x})->output(),scalar(beta)})->output();return std::vector<V*>{Op(g,n,"Where",{Op(g,n,"Greater",{x,scalar(threshold)})->output(),n.input(0),nonlinear})->output()};};
  result["aten::max_pool2d_with_indices"] = [](N& n, G& g) {
    auto* p = Op(g, n, "MaxPool", {n.input(0)}, 2);
    auto k = Literal(n.input(1)).toIntVector(),
         s = Literal(n.input(2)).toIntVector(),
         pad = Literal(n.input(3)).toIntVector();
    if (s.empty()) s = k;
    auto pads = pad;
    pads.insert(pads.end(), pad.begin(), pad.end());
    p->is_(c10::Symbol::attr("kernel_shape"), k);
    p->is_(c10::Symbol::attr("strides"), s);
    p->is_(c10::Symbol::attr("pads"), pads);
    p->is_(c10::Symbol::attr("dilations"), Literal(n.input(4)).toIntVector());
    p->i_(c10::Symbol::attr("ceil_mode"), Literal(n.input(5)).toBool());
    return std::vector<V*>{p->output(0), p->output(1)};
  };
  return result;
}
}  // namespace jyppx::liuding::vision::tasks::detail
