#include "caffe/layers/ogn_concat_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
void OGNConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tOGNConcatLayer:: LayerSetUp: \t";

	_key1_layer_name = this->layer_param_.ogn_concat_param().key1_layer();
	_key2_layer_name = this->layer_param_.ogn_concat_param().key2_layer();

	if (bottom.size() != 2) {
		CHECK(false) << prefix << "Number of inputs to the layer should be 2!" << std::endl;
	}

	_batch_size = bottom[0]->shape(0);
	if (bottom[1]->shape(0) != _batch_size) {
		CHECK(false) << prefix << "Batch size for the two bottom blobs are different: " << 
			_batch_size << ", " << bottom[1]->shape(0) << std::endl;
	}

	_num_channels = bottom[0]->shape(1);
	if (bottom[1]->shape(1) != _num_channels) {
		CHECK(false) << prefix << "Channel length for the two bottom blobs are different: " << 
			_num_channels << ", " << bottom[1]->shape(1) << std::endl;
	}

}

template <typename Dtype>
void OGNConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	_l1_num_pixels = bottom[0]->shape(2);
	_l2_num_pixels = bottom[1]->shape(2);

	_num_pixels = _l1_num_pixels + _l2_num_pixels;

	vector<int> output_shape;
	output_shape.push_back(_batch_size);
	output_shape.push_back(_num_channels);
	output_shape.push_back(_num_pixels);
	top[0]->Reshape(output_shape);
}

template <typename Dtype>
void OGNConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tOGNConcatLayer:: Forward_cpu: \t";

	std::cout<<prefix<<": "<<_key1_layer_name<<", "<<_key2_layer_name<<std::endl;

	this->_octree_keys.clear();
	this->_octree_prop.clear();

	boost::shared_ptr<Layer<Dtype> > base1_ptr = this->parent_net()->layer_by_name(_key1_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l1_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base1_ptr);
	boost::shared_ptr<Layer<Dtype> > base2_ptr = this->parent_net()->layer_by_name(_key2_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l2_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base2_ptr);

	const Dtype* input_arr1 = bottom[0]->cpu_data();
	const Dtype* input_arr2 = bottom[1]->cpu_data();

	Dtype* output_arr = top[0]->mutable_cpu_data();
	memset(output_arr, 0, sizeof(Dtype) * _batch_size * _num_channels * _num_pixels);

	for (int n = 0; n < _batch_size; ++n) {
		int counter = 0;

		GeneralOctree<int> octree_keys;
		GeneralOctree<int> octree_prop;

		set<KeyType> l1_keys;
		GeneralOctree<int>* l1_tree = &(l1_ptr->get_keys_octree(n));
		for (GeneralOctree<int>::iterator it=l1_tree->begin(); it!=l1_tree->end(); ++it) {
			KeyType key = it->first;
			if (key != GeneralOctree<int>::INVALID_KEY()) {
				l1_keys.insert(key);
				octree_keys.add_element(key, counter);
				octree_prop.add_element(key, PROP_TRUE);
				int value_ind = l1_tree->get_value(key);
				for (int ch = 0; ch < _num_channels; ++ch) {
					int left_ind = n*_num_channels*_num_pixels + ch*_num_pixels + counter;
					int right_ind = n*_num_channels*_l1_num_pixels + ch*_l1_num_pixels + value_ind;
					output_arr[left_ind] = input_arr1[right_ind];
				}
				++ counter;
			}
		}
		
		GeneralOctree<int>* l2_tree = &(l2_ptr->get_keys_octree(n));
		for (GeneralOctree<int>::iterator it=l2_tree->begin(); it!=l2_tree->end(); ++it) {
			KeyType key = it->first;
			if (key != GeneralOctree<int>::INVALID_KEY()) {
				if (l1_keys.find(key) == l1_keys.end()) {
					octree_keys.add_element(key, counter);
					octree_prop.add_element(key, PROP_TRUE);
					int value_ind = l2_tree->get_value(key);
					for (int ch = 0; ch < _num_channels; ++ch) {
						int left_ind = n*_num_channels*_num_pixels + ch*_num_pixels + counter;
						int right_ind = n*_num_channels*_l2_num_pixels + ch*_l2_num_pixels + value_ind;
						output_arr[left_ind] = input_arr2[right_ind];
					}
					++ counter;
				} else {
					CHECK(false) << prefix << "key " << key << " occurs in both two bottom inputs! Should be impossible!" << std::endl;
				}
			}
		}

		this->_octree_keys.push_back(octree_keys);
		this->_octree_prop.push_back(octree_prop);
	}
}

template <typename Dtype>
void OGNConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "\t\tOGNConcatLayer:: Backward_cpu: \t";

	const Dtype* input_arr = top[0]->cpu_diff();
	
	Dtype* output_arr1 = bottom[0]->mutable_cpu_diff();
	Dtype* output_arr2 = bottom[1]->mutable_cpu_diff();
	memset(output_arr1, 0, sizeof(Dtype) * _batch_size * _num_channels * _l1_num_pixels);
	memset(output_arr2, 0, sizeof(Dtype) * _batch_size * _num_channels * _l2_num_pixels);

	boost::shared_ptr<Layer<Dtype> > base1_ptr = this->parent_net()->layer_by_name(_key1_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l1_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base1_ptr);
	boost::shared_ptr<Layer<Dtype> > base2_ptr = this->parent_net()->layer_by_name(_key2_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l2_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base2_ptr);

	for (int n = 0; n < _batch_size; ++n) {
		for (GeneralOctree<int>::iterator it=this->_octree_keys[n].begin(); it!=this->_octree_keys[n].end(); ++it) {
			KeyType key = it->first;
			int cur_value_ind = this->_octree_keys[n].get_value(key);
			if(key != GeneralOctree<int>::INVALID_KEY()) {
				if (l1_ptr->get_keys_octree(n).get_value(key) >= 0) {
					int value_ind = l1_ptr->get_keys_octree(n).get_value(key);
					for (int ch = 0; ch < _num_channels; ++ch) {
						int left_ind = n*_num_channels*_num_pixels + ch*_num_pixels + cur_value_ind;
						int right_ind = n*_num_channels*_l1_num_pixels + ch*_l1_num_pixels + value_ind;
						output_arr1[right_ind] += input_arr[left_ind];
					}
				} else if (l2_ptr->get_keys_octree(n).get_value(key) >= 0) {
					int value_ind = l2_ptr->get_keys_octree(n).get_value(key);
					for (int ch = 0; ch < _num_channels; ++ch) {
						int left_ind = n*_num_channels*_num_pixels + ch*_num_pixels + cur_value_ind;
						int right_ind = n*_num_channels*_l2_num_pixels + ch*_l2_num_pixels + value_ind;
						output_arr2[right_ind] += input_arr[left_ind];
					}
				} else {
					CHECK(false) << prefix << "key " << key <<" is not found in either of the two input blobs." << std::endl;
				}
			}
		}
	}
}

INSTANTIATE_CLASS(OGNConcatLayer);
REGISTER_LAYER_CLASS(OGNConcat);

}  // namespace caffe
