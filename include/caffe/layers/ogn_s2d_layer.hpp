#ifndef OGN_S2D_LAYER_HPP_
#define OGN_S2D_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"

namespace caffe {

template <typename Dtype>
class OGNS2DLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNS2DLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNS2D"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
  string _key_layer_name;

  int _num_pixels;
  int _batch_size;
  int _num_channels;
  int _current_level;
  int _current_res;
};

}  // namespace caffe

#endif  // OGN_S2D_LAYER_HPP_
