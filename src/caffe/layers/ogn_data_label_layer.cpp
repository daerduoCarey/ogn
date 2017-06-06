#include "caffe/layers/ogn_data_label_layer.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void OGNDataLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    _model_counter = 0;
    _current_model_ind = 0;
    _done_initial_reshape = false;
    load_data_from_disk();
}

template <typename Dtype>
void OGNDataLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const int batch_size = this->layer_param_.ogn_data_label_param().batch_size();

    vector<int> values_shape;
    vector<int> labels_shape;
    labels_shape.push_back(batch_size);

    if(!_done_initial_reshape)
    {
        values_shape.push_back(batch_size); values_shape.push_back(1);
        _done_initial_reshape = true;
    }
    else
    {
	bool data_shuffle = this->layer_param_.ogn_data_label_param().shuffle();
        vector<int> batch_elements;
        for(int bt=0; bt<batch_size; bt++)
        {
		if(data_shuffle) {
			int random_ind = rand() % _model_counter;
                	batch_elements.push_back(random_ind);
		} else {
			batch_elements.push_back(_current_model_ind);
			++ _current_model_ind;
			if(_current_model_ind == _file_names.size()) {
				_current_model_ind = 0;
			}
		}
        }
        int num_elements = select_next_batch_models(batch_elements);
        values_shape.push_back(batch_size); values_shape.push_back(num_elements);
    }

    top[0]->Reshape(values_shape);
    top[1]->Reshape(labels_shape);
}

template <typename Dtype>
void OGNDataLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    this->_octree_keys.clear();
    const int batch_size = this->layer_param_.ogn_data_label_param().batch_size();
    int num_elements = top[0]->shape(1);

    Dtype* top_values = top[0]->mutable_cpu_data();
    Dtype* top_labels = top[1]->mutable_cpu_data();

    memset(top_values, 0, sizeof(Dtype) * top[0]->count());
    memset(top_labels, 0, sizeof(Dtype) * top[1]->count());

    for(int bt=0; bt<batch_size; bt++)
    {
        GeneralOctree<int> octree_keys;
        int counter = 0;
        for(Octree::iterator it=_batch_octrees[bt].begin(); it!=_batch_octrees[bt].end(); it++)
        {
            int top_index = bt * num_elements + counter;
            top_values[top_index] = (Dtype)(it->second);
            octree_keys.add_element(it->first, counter);
            counter++;
        }
        top_labels[bt] = _batch_labels[bt];
        this->_octree_keys.push_back(octree_keys);
    }
}

template <typename Dtype>
void OGNDataLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}

template <typename Dtype>
int OGNDataLabelLayer<Dtype>::select_next_batch_models(vector<int> labels)
{
    const bool preload_data = this->layer_param_.ogn_data_label_param().preload_data();
    int num_elements = 0;
    _batch_octrees.clear();
    _batch_labels.clear();

    for(int bt=0; bt<labels.size(); bt++)
    {
        int len = 0;
        _batch_labels.push_back(_model_labels[labels[bt]]);
        if(preload_data)
        {
            _batch_octrees.push_back(_octrees[labels[bt]]);
            len = _octrees[labels[bt]].num_elements();
        }
        else
        {
            Octree tree;
            tree.from_file(_file_names[labels[bt]]);
            _batch_octrees.push_back(tree);
            len = tree.num_elements();
        }
        if(len > num_elements) num_elements = len;
    }
    return num_elements;
}

template <typename Dtype>
void OGNDataLabelLayer<Dtype>::load_data_from_disk()
{
    cout << "Loading training data from disk..." << endl;
    const string model_source = this->layer_param_.ogn_data_label_param().model_source();
    const string label_source = this->layer_param_.ogn_data_label_param().label_source();
    const bool preload_data = this->layer_param_.ogn_data_label_param().preload_data();

    ifstream infile(model_source.c_str());
    ifstream infile_label(label_source.c_str());
    string name;
    int gt_label;
    while(infile >> name)
    {
	infile_label >> gt_label;
	_model_labels.push_back(gt_label);
	_file_names.push_back(name);
        if(preload_data)
        {
            Octree tree;
            tree.from_file(name);
            _octrees.push_back(tree);
            cout << name << endl;
        }
        _model_counter++;
    }

    if (_model_labels.size() != _file_names.size()) {
    	CHECK(false) << "OGNDataLabelLayer<Dtype>::load_data_from_disk: _model_labels.size() != _file_names.size()" << std::endl;
    }

    infile.close();
    infile_label.close();

    std::cout << "Done." << std::endl;
}

INSTANTIATE_CLASS(OGNDataLabelLayer);
REGISTER_LAYER_CLASS(OGNDataLabel);

}  // namespace caffe
