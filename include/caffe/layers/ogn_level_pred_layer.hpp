#ifndef OGN_LEVEL_PRED_LAYER_HPP_
#define OGN_LEVEL_PRED_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"

namespace caffe {

template <typename Dtype>
class OGNLevelPredLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNLevelPredLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNLevelPred"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
  string _key_layer_name;

  int _num_pixels;
  int _batch_size;
};

}  // namespace caffe

#endif  // OGN_LEVEL_PRED_LAYER_HPP_
