#ifndef OGN_CONCAT_LAYER_HPP_
#define OGN_CONCAT_LAYER_HPP_

#include <set>
#include "caffe/layers/ogn_layer.hpp"

namespace caffe {

template <typename Dtype>
class OGNConcatLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNConcatLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNConcat"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  string _key1_layer_name;
  string _key2_layer_name;

  int _num_channels;
  int _num_pixels, _l1_num_pixels, _l2_num_pixels;
  int _batch_size;
};

}  // namespace caffe

#endif  // OGN_CONCAT_LAYER_HPP_
