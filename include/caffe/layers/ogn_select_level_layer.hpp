#ifndef OGN_SELECT_LEVEL_LAYER_HPP_
#define OGN_SELECT_LEVEL_LAYER_HPP_

#include <set>
#include "caffe/layers/ogn_layer.hpp"

namespace caffe {

template <typename Dtype>
class OGNSelectLevelLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNSelectLevelLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNSelectLevel"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
  string _key_layer_name;

  int _batch_size;
  int _current_level;
  int _current_res;
  int _num_pixels;

  vector<int> _batch_num_pixels;
};

}  // namespace caffe

#endif  // OGN_SELECT_LEVEL_LAYER_HPP_
