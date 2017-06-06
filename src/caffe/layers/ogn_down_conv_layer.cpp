#include "caffe/layers/ogn_down_conv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
void OGNDownConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	_num_output_channels = this->layer_param_.ogn_down_conv_param().output_channels();
	_num_input_channels = bottom[0]->shape(1);

	this->blobs_.resize(2);

	_done_initial_reshape = false;

	_filter_size = 2;

        _weight_shape.push_back(_num_output_channels);
        _weight_shape.push_back(_num_input_channels);
        _weight_shape.push_back(_filter_size);
        _weight_shape.push_back(_filter_size);
        _weight_shape.push_back(_filter_size);
	
	_bias_shape.push_back(_num_output_channels);

    	this->blobs_[0].reset(new Blob<Dtype>(_weight_shape));
    	this->blobs_[1].reset(new Blob<Dtype>(_bias_shape));

    	shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        	this->layer_param_.ogn_down_conv_param().weight_filler()));
    	weight_filler->Fill(this->blobs_[0].get());

    	shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        	this->layer_param_.ogn_down_conv_param().bias_filler()));
    	bias_filler->Fill(this->blobs_[1].get());
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
	_batch_size = bottom[0]->shape(0);
	_num_input_pixels = bottom[0]->shape(2);

	if(!_done_initial_reshape) {
		vector<int> features_shape;
		features_shape.push_back(_batch_size);
		features_shape.push_back(_num_output_channels);
		features_shape.push_back(1);
		top[0]->Reshape(features_shape);

		_done_initial_reshape = true;
	} else {
		std::string key_layer_name = this->layer_param_.ogn_down_conv_param().key_layer();
		boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
		boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

		_num_output_pixels = 0;

		_next_level_keys.clear();
		for (int n = 0; n < _batch_size; ++n) {
			set<KeyType> next_level_keys;
			GeneralOctree<int> cur_key_octree = l_ptr->get_keys_octree(n);
			for(typename GeneralOctree<int>::iterator it=cur_key_octree.begin(); it!=cur_key_octree.end(); it++) {
				KeyType key = it->first;
				if (key != GeneralOctree<int>::INVALID_KEY()) {
					key >>= 3;
					next_level_keys.insert(key);
				}
			}
			_next_level_keys.push_back(next_level_keys);
			if (next_level_keys.size() > _num_output_pixels) {
				_num_output_pixels = next_level_keys.size();
			}
		}

		vector<int> features_shape;
		features_shape.push_back(_batch_size);
		features_shape.push_back(_num_output_channels);
		features_shape.push_back(_num_output_pixels);
		top[0]->Reshape(features_shape);

		_col_buffer_shape.clear();
		_col_buffer_shape.push_back(_num_input_channels * _filter_size * _filter_size * _filter_size);
		_col_buffer_shape.push_back(_num_output_pixels);

		_col_buffer.Reshape(_col_buffer_shape);
	}
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::propagate_keys_cpu() {
	this->_octree_keys.clear();
	this->_octree_prop.clear();

	for (int n = 0; n < _batch_size; ++n) {
		int output_counter = 0;

		GeneralOctree<int> octree_keys;
		GeneralOctree<int> octree_prop;
		
		set<KeyType>* cur_next_level_keys = &(this->_next_level_keys[n]);
		for (set<KeyType>::iterator it = cur_next_level_keys->begin(); it != cur_next_level_keys->end(); ++it) {
			KeyType key = *it;
			octree_keys.add_element(key, output_counter);
			octree_prop.add_element(key, PROP_TRUE);
			++ output_counter;
		}

		this->_octree_keys.push_back(octree_keys);
		this->_octree_prop.push_back(octree_prop);
	}
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::resize_computation_buffers_cpu(int batch_num_pixels)
{
    vector<int> bias_multiplier_shape;
    bias_multiplier_shape.push_back(_num_output_pixels); bias_multiplier_shape.push_back(1);
    _bias_multiplier.Reshape(bias_multiplier_shape);
    caffe_set(_num_output_pixels, Dtype(0), _bias_multiplier.mutable_cpu_data());
    caffe_set(batch_num_pixels, Dtype(1), _bias_multiplier.mutable_cpu_data());
    memset(_col_buffer.mutable_cpu_data(), 0, sizeof(Dtype)*_col_buffer_shape[0]*_col_buffer_shape[1]);
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "OGNDownConvLayer<Dtype>::Forward_cpu";

	propagate_keys_cpu();

	for (int n=0; n<_batch_size; n++) {
		int num_elements = this->_next_level_keys[n].size();
		resize_computation_buffers_cpu(num_elements);
		im2col_octree_cpu(n, bottom, top);
		forward_cpu_gemm(this->blobs_[0]->cpu_data(), _col_buffer.cpu_data(),
				top[0]->mutable_cpu_data() + n * _num_output_channels * _num_output_pixels);
		forward_cpu_bias(top[0]->mutable_cpu_data() + 
				n * _num_output_channels * _num_output_pixels, this->blobs_[1]->cpu_data());
	}
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "OGNDownConvLayer<Dtype>::Backward_cpu";

	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
	const Dtype* top_diff = top[0]->cpu_diff();

	memset(weight_diff, 0, sizeof(Dtype)*_num_output_channels*_num_input_channels*_filter_size*_filter_size*_filter_size);
	memset(bias_diff, 0, sizeof(Dtype)*_num_output_channels);
	
	for (int n = 0; n < _batch_size; ++n) {
		resize_computation_buffers_cpu(this->_octree_keys[n].num_elements());
		im2col_octree_cpu(n, bottom, top);
		weight_cpu_gemm(_col_buffer.cpu_data(),
			top_diff + n * _num_output_channels * _num_output_pixels, weight_diff);
		backward_cpu_bias(bias_diff, top_diff + n * _num_output_channels * _num_output_pixels);
		backward_cpu_gemm(top_diff + n * _num_output_channels * _num_output_pixels,
			this->blobs_[0]->cpu_data(), _col_buffer.mutable_cpu_data());
		col2im_octree_cpu(n, bottom, top);
    	}
}


template <typename Dtype>
void OGNDownConvLayer<Dtype>::backward_cpu_gemm(const Dtype* top_diff, const Dtype* weights, Dtype* col_buff)
{
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
        _col_buffer_shape[1], _weight_shape[0],
        (Dtype)1., weights, top_diff,
        (Dtype)0., col_buff);
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::forward_cpu_gemm(const Dtype* weights, const Dtype* input, Dtype* output)
{
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
        _col_buffer_shape[1], _col_buffer_shape[0],
        (Dtype)1., weights, input,
        (Dtype)0., output);
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias)
{
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _bias_shape[0], _num_output_pixels, 1,
                          (Dtype)1., bias, _bias_multiplier.cpu_data(),
                          (Dtype)1., output);
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input)
{
    caffe_cpu_gemv<Dtype>(CblasNoTrans, _num_output_channels, _num_output_pixels, 1.,
      input, _bias_multiplier.cpu_data(), (Dtype)1., bias);
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights)
{
    const Dtype* col_buff = input;
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                          _col_buffer_shape[0], _col_buffer_shape[1],
                          (Dtype)1., output, col_buff, (Dtype)1., weights);
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::col2im_octree_cpu(int batch_ind, const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	Dtype* output_arr = bottom[0]->mutable_cpu_diff();
	int output_rows = _num_input_channels;
	int output_cols = _num_input_pixels;

	if(!batch_ind) memset(output_arr, 0, sizeof(Dtype) * _batch_size * output_rows * output_cols);

	std::string key_layer_name = this->layer_param_.ogn_down_conv_param().key_layer();
	boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

	set<KeyType>* cur_next_level_keys = &(this->_next_level_keys[batch_ind]);
	for(set<KeyType>::iterator it = cur_next_level_keys->begin(); it != cur_next_level_keys->end(); ++it) {
		KeyType key = *it;
		key <<= 3;
		
		GeneralOctree<int>* parent_keys_octree = &(l_ptr->get_keys_octree(batch_ind));
		std::vector<KeyType> neighbors = parent_keys_octree->get_neighbor_keys(key, _filter_size);

		for(int ch = 0; ch < _num_input_channels; ++ch) {
			for(int el = 0; el < neighbors.size(); ++el) {
				int col_buff_ind = ch * neighbors.size() * _col_buffer_shape[1] + 
					el * _col_buffer_shape[1] + this->_octree_keys[batch_ind].get_value(*it);

				if (neighbors[el] != GeneralOctree<int>::INVALID_KEY()) {
					KeyType nbh_key = neighbors[el];
					int feature_ind = batch_ind * _num_input_channels * _num_input_pixels + 
						ch * _num_input_pixels + parent_keys_octree->get_value(nbh_key);
					output_arr[feature_ind] += _col_buffer.mutable_cpu_data()[col_buff_ind];
				}
			}
		}
	}

    memset(_col_buffer.mutable_cpu_data(), 0, sizeof(Dtype)*_col_buffer_shape[0]*_col_buffer_shape[1]);
}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::im2col_octree_cpu(int batch_ind, const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	Dtype* col_buff = _col_buffer.mutable_cpu_data();

	std::string key_layer_name = this->layer_param_.ogn_down_conv_param().key_layer();
	boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
	boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

	set<KeyType>* cur_next_level_keys = &(this->_next_level_keys[batch_ind]);
	GeneralOctree<int>* parent_keys_octree = &(l_ptr->get_keys_octree(batch_ind));
	for(set<KeyType>::iterator it = cur_next_level_keys->begin(); it != cur_next_level_keys->end(); ++it) {
		KeyType key = *it;
		key <<= 3;
		
		std::vector<KeyType> neighbors = parent_keys_octree->get_neighbor_keys(key, _filter_size);
		for(int ch = 0; ch < _num_input_channels; ++ch) {
			for(int el = 0; el < neighbors.size(); ++el) {
				int col_buff_ind = ch * neighbors.size() * _col_buffer_shape[1] + 
					el * _col_buffer_shape[1] + this->_octree_keys[batch_ind].get_value(*it);

				if (neighbors[el] != GeneralOctree<int>::INVALID_KEY()) {
					KeyType nbh_key = neighbors[el];
					int feature_ind = batch_ind * _num_input_channels * _num_input_pixels + 
						ch * _num_input_pixels + parent_keys_octree->get_value(nbh_key);
					col_buff[col_buff_ind] = bottom[0]->cpu_data()[feature_ind];
				} else {
					col_buff[col_buff_ind] = 0;
				}
			}
		}
	}
}


INSTANTIATE_CLASS(OGNDownConvLayer);
REGISTER_LAYER_CLASS(OGNDownConv);

}  // namespace caffe
