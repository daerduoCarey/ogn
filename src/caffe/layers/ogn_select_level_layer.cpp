#include "caffe/layers/ogn_select_level_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
void OGNSelectLevelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	LOG(FATAL) << "OGNSelectLevelLayer is only implemented for data type byte! Not implemented for general type!" << std::endl;
}

template <>
void OGNSelectLevelLayer<byte>::LayerSetUp(const vector<Blob<byte>*>& bottom,
      const vector<Blob<byte>*>& top) {

	string prefix = "\t\tOGNSelectLevelLayer:: LayerSetUp: \t";

	_batch_size = bottom[0]->shape(0);
	_current_level = this->layer_param_.ogn_select_level_param().level();
	_current_res = pow(2, _current_level);
}

template <typename Dtype>
void OGNSelectLevelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	LOG(FATAL) << "OGNSelectLevelLayer is only implemented for data type byte! Not implemented for general type!" << std::endl;
}

template <>
void OGNSelectLevelLayer<byte>::Reshape(const vector<Blob<byte>*>& bottom,
      const vector<Blob<byte>*>& top) {

	string prefix = "\t\tognselectlevellayer:: Reshape: \t";

	this->_octree_keys.clear();
	this->_octree_prop.clear();

	this->_batch_num_pixels.clear();

	boost::shared_ptr<Layer<byte> > base_ptr = this->parent_net()->layer_by_name(_key_layer_name);
	boost::shared_ptr<OGNLayer<byte> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<byte> >(base_ptr);

	_num_pixels = 0;
	for (int n = 0; n < _batch_size; ++n) {
		int count = 0;
		GeneralOctree<int> key_octree;
		GeneralOctree<int> prop_octree;

		GeneralOctree<int>* current_tree = &(l_ptr->get_keys_octree(n));
		for(GeneralOctree<int>::iterator it = current_tree->begin(); it != current_tree->end(); ++it) {
			KeyType key = it->first;
			int level = current_tree->compute_level(key);
			if (level != _current_level) continue;
			
			byte value = it->second;
			if (key != GeneralOctree<int>::INVALID_KEY() && value) {
				key_octree.add_element(key, count);
				prop_octree.add_element(key, PROP_TRUE);
				++ count;
			}
		}

		this->_octree_keys.push_back(key_octree);
		this->_octree_prop.push_back(prop_octree);

		this->_batch_num_pixels.push_back(count);
		if (count > _num_pixels) {
			_num_pixels = count;
		}
	}

	vector<int> output_shape;
	output_shape.push_back(_batch_size);
	output_shape.push_back(1);
	output_shape.push_back(_num_pixels);
	top[0]->Reshape(output_shape);
}

template <typename Dtype>
void OGNSelectLevelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	LOG(FATAL) << "OGNSelectLevelLayer is only implemented for data type byte! Not implemented for general type!" << std::endl;
}

template <>
void OGNSelectLevelLayer<byte>::Forward_cpu(const vector<Blob<byte>*>& bottom,
      const vector<Blob<byte>*>& top) {

	byte* output_arr = top[0]->mutable_cpu_data();

	memset(output_arr, 0, sizeof(byte) * _batch_size * 1 * _num_pixels);
	for(int n = 0; n < _batch_size; ++n) {
		caffe_set(this->_batch_num_pixels[n], byte(1), output_arr + n * _num_pixels);
	}
}

template <typename Dtype>
void OGNSelectLevelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	LOG(FATAL) << "OGNSelectLevelLayer is only implemented for data type byte! Not implemented for general type!" << std::endl;
}

template <>
void OGNSelectLevelLayer<byte>::Backward_cpu(const vector<Blob<byte>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<byte>*>& bottom) {
	LOG(FATAL) << "Backward not implemented";
}

INSTANTIATE_CLASS(OGNSelectLevelLayer);
REGISTER_LAYER_CLASS(OGNSelectLevel);

}  // namespace caffe
