#ifndef OGN_DATA_LABEL_LAYER_HPP_
#define OGN_DATA_LABEL_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
class OGNDataLabelLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNDataLabelLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNDataLabel"; }
  // virtual inline int ExactNumBottomBlobs() const { return 1; }
  // virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
   void load_data_from_disk();
   int select_next_batch_models(std::vector<int> labels);

   std::vector<Octree> _octrees;
   std::vector<Octree> _batch_octrees;
   std::vector<int> _batch_labels;
   std::vector<std::string> _file_names;
   std::vector<int> _model_labels;

   bool _done_initial_reshape;
   int _model_counter;
   int _current_model_ind;
};

}  // namespace caffe

#endif  // OGN_DATA_LABEL_LAYER_HPP
