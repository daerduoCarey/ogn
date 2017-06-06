#include "caffe/layers/ogn_down_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void OGNDownConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "OGNDownConvLayer<Dtype>::Forward_gpu";

    propagate_keys_cpu();

    for (int n = 0; n < _batch_size; n++)
    {
        int num_elements = this->_octree_keys[n].num_elements();
    	resize_computation_buffers_cpu(num_elements);
    	im2col_octree_cpu(n, bottom, top);
    	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
			_col_buffer_shape[1], _col_buffer_shape[0],
	    		(Dtype)1., this->blobs_[0]->gpu_data(), _col_buffer.mutable_gpu_data(),
	    		(Dtype)0., top[0]->mutable_gpu_data() + n * _num_output_channels * _num_output_pixels);
    	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _bias_shape[0], _num_output_pixels, 1,
	    		(Dtype)1., this->blobs_[1]->gpu_data(), _bias_multiplier.gpu_data(), (Dtype)1., top[0]->mutable_gpu_data() +
			n * _num_output_channels * _num_output_pixels);
    }

}

template <typename Dtype>
void OGNDownConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "OGNDownConvLayer<Dtype>::Backward_gpu";

    for (int n = 0; n < _batch_size; ++n)
    {
            resize_computation_buffers_cpu(this->_octree_keys[n].num_elements());
            im2col_octree_cpu(n, bottom, top);

            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0], _col_buffer_shape[0], _col_buffer_shape[1],
                (Dtype)1., top[0]->gpu_diff() + n * _num_output_channels * _num_output_pixels, _col_buffer.mutable_gpu_data(), 
		(Dtype)1., this->blobs_[0]->mutable_gpu_diff());

            caffe_gpu_gemv<Dtype>(CblasNoTrans, _num_output_channels, _num_output_pixels, 1.,
              top[0]->gpu_diff() + n * _num_output_channels * _num_output_pixels, _bias_multiplier.gpu_data(), (Dtype)1., this->blobs_[1]->mutable_gpu_diff());

            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
                _col_buffer_shape[1], _weight_shape[0], (Dtype)1., this->blobs_[0]->gpu_data(),
                top[0]->gpu_diff() + n * _num_output_channels * _num_output_pixels, (Dtype)0., _col_buffer.mutable_gpu_data());

            col2im_octree_cpu(n, bottom, top);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(OGNDownConvLayer);

}  // namespace caffe
