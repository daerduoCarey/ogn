#include "caffe/layers/ogn_level_pred_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
void OGNLevelPredLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tOGNLevelPredLayer:: LayerSetUp: \t";
	_key_layer_name = this->layer_param_.ogn_level_pred_param().key_layer();
}

template <typename Dtype>
void OGNLevelPredLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	_batch_size = bottom[0]->shape(0);
	
	string prefix = "\t\tOGNLevelPredLayer:: Reshape: \t";
	if (bottom[0]->shape(1) != OGN_NUM_CLASSES) {
		CHECK(false) << prefix << "Input should be have channel length OGN_NUM_CLASSES = " << OGN_NUM_CLASSES << std::endl;
	}

	_num_pixels = bottom[0]->shape(2);

	vector<int> output_shape;
	output_shape.push_back(_batch_size);
	output_shape.push_back(1);
	output_shape.push_back(_num_pixels);
	top[0]->Reshape(output_shape);
}

template <typename Dtype>
void OGNLevelPredLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tOGNLevelPredLayer:: Forward_cpu: \t";

	this->_octree_keys.clear();
	this->_octree_prop.clear();

	boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(_key_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

	const Dtype* input_arr = bottom[0]->cpu_data();
	Dtype* output_arr = top[0]->mutable_cpu_data();

	memset(output_arr, 0, sizeof(Dtype) * _batch_size * 1 * _num_pixels);

	for (int n = 0; n < _batch_size; ++n) {
		GeneralOctree<int> octree_keys;
		GeneralOctree<int> octree_prop;
		int count = 0;

		GeneralOctree<int>* l_tree = &(l_ptr->get_keys_octree(n));
		for (GeneralOctree<int>::iterator it=l_tree->begin(); it!=l_tree->end(); ++it) {
			KeyType key = it->first;
			if (key != GeneralOctree<int>::INVALID_KEY()) {
				int value_ind = l_tree->get_value(key);
				int empty_ind = n*3*_num_pixels + 0*_num_pixels + value_ind;
				Dtype empty_rate = input_arr[empty_ind];
				int filled_ind = n*3*_num_pixels + 1*_num_pixels + value_ind;
				Dtype filled_rate = input_arr[filled_ind];
				int mixed_ind = n*3*_num_pixels + 2*_num_pixels + value_ind;
				Dtype mixed_rate = input_arr[mixed_ind];

				if (filled_rate > mixed_rate && filled_rate > empty_rate) {
					octree_keys.add_element(key, count);
					octree_prop.add_element(key, PROP_TRUE);
					int output_ind = n*_num_pixels + count;
					output_arr[output_ind] = filled_rate;
					++ count;
				}
			}
		}

		this->_octree_keys.push_back(octree_keys);
		this->_octree_prop.push_back(octree_prop);
	}
}

template <typename Dtype>
void OGNLevelPredLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "\t\tOGNLevelPredLayer:: Backward_cpu: \t";

	boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(_key_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

	const Dtype* input_arr = top[0]->cpu_diff();
	Dtype* output_arr = bottom[0]->mutable_cpu_diff();

	memset(output_arr, 0, sizeof(Dtype) * _batch_size * 3 * _num_pixels);

	for (int n = 0; n < _batch_size; ++n) {
		GeneralOctree<int>* cur_tree = &(this->_octree_keys[n]);
		GeneralOctree<int>* l_tree = &(l_ptr->get_keys_octree(n));
		for (GeneralOctree<int>::iterator it=cur_tree->begin(); it!=cur_tree->end(); ++it) {
			KeyType key = it->first;
			if (key != GeneralOctree<int>::INVALID_KEY()) {
				int parent_value_ind = l_tree->get_value(key);
				int cur_value_ind = cur_tree->get_value(key);
				int output_ind = n*3*_num_pixels + 1*_num_pixels + parent_value_ind;
				int input_ind = n*_num_pixels + cur_value_ind;
				output_arr[output_ind] += input_arr[input_ind];
			}
		}
	}
}

INSTANTIATE_CLASS(OGNLevelPredLayer);
REGISTER_LAYER_CLASS(OGNLevelPred);

}  // namespace caffe
