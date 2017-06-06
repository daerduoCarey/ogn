#include "caffe/layers/ogn_s2d_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
void OGNS2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tOGNS2DLayer:: LayerSetUp: \t";

	_batch_size = bottom[0]->shape(0);
	_num_channels = bottom[0]->shape(1);
	_num_pixels = bottom[0]->shape(2);
	
	_current_level = this->layer_param_.ogn_s2d_param().level();
	_current_res = pow(2, _current_level);

	_key_layer_name = this->layer_param_.ogn_s2d_param().key_layer();
}

template <typename Dtype>
void OGNS2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	vector<int> output_shape;
	output_shape.push_back(_batch_size);
	output_shape.push_back(_num_channels);
	output_shape.push_back(_current_res);
	output_shape.push_back(_current_res);
	output_shape.push_back(_current_res);
	top[0]->Reshape(output_shape);
}

template <typename Dtype>
void OGNS2DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tOGNS2DLayer:: Forward_cpu: \t";

	this->_octree_keys.clear();
	this->_octree_prop.clear();

	boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(_key_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

	const Dtype* input_arr = bottom[0]->cpu_data();
	Dtype* output_arr = top[0]->mutable_cpu_data();

	memset(output_arr, 0, sizeof(Dtype) * _batch_size * _num_channels * _current_res * _current_res * _current_res);

	for (int n = 0; n < _batch_size; ++n) {
		GeneralOctree<int>* l_tree = &(l_ptr->get_keys_octree(n));
		for (GeneralOctree<int>::iterator it=l_tree->begin(); it!=l_tree->end(); ++it) {
			KeyType key = it->first;
			int level = l_tree->compute_level(key);
			if (key != GeneralOctree<int>::INVALID_KEY()) {
				if (level == _current_level) {
					int value_ind = l_tree->get_value(key);
					OctreeCoord coord = l_tree->compute_coord(key);
					for (int ch = 0; ch < _num_channels; ++ch) {
						int left_ind = n*_num_channels*_current_res*_current_res*_current_res + 
							ch*_current_res*_current_res*_current_res + 
							coord.x*_current_res*_current_res + coord.y*_current_res + coord.z;
						int right_ind = n*_num_channels*_num_pixels + ch*_num_pixels + value_ind;
						output_arr[left_ind] = input_arr[right_ind];
					}
				} else {
					CHECK(false) << prefix << " level (" << level << ") does not equal to input level (" << _current_level << ")!" << std::endl;
				}
			}
		}
	}
}

template <typename Dtype>
void OGNS2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "OGNS2DLayer<Dtype>::Backward_cpu";

	boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(_key_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

	const Dtype* input_arr = top[0]->cpu_diff();
	Dtype* output_arr = bottom[0]->mutable_cpu_diff();

	memset(output_arr, 0, sizeof(Dtype) * _batch_size * _num_channels * _num_pixels);

	for (int n = 0; n < _batch_size; ++n) {
		GeneralOctree<int>* l_tree = &(l_ptr->get_keys_octree(n));
		for (GeneralOctree<int>::iterator it=l_tree->begin(); it!=l_tree->end(); ++it) {
			KeyType key = it->first;
			if (key != GeneralOctree<int>::INVALID_KEY()) {
				OctreeCoord coord = l_tree->compute_coord(key);
				int value_ind = l_tree->get_value(key);
				for (int ch = 0; ch < _num_channels; ++ch) {
					int left_ind = n*_num_channels*_current_res*_current_res*_current_res + 
						ch*_current_res*_current_res*_current_res + 
						coord.x*_current_res*_current_res + coord.y*_current_res + coord.z;
					int right_ind = n*_num_channels*_num_pixels + ch*_num_pixels + value_ind;
					output_arr[right_ind] += input_arr[left_ind];
				}
			}
		}
	}
}

INSTANTIATE_CLASS(OGNS2DLayer);
REGISTER_LAYER_CLASS(OGNS2D);

}  // namespace caffe
