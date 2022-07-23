#include "../includes/data_handler.hpp"

//?  Data instantiation

data_instance::data_instance()
{
    _feature_vector = new std::vector<uint8_t>;
}

data_instance::~data_instance()
{
    delete _feature_vector;
}

void data_instance::set_label(uint8_t label)
{
    _label = label;
}

void data_instance::set_label_enumeration(int label_enumeration)
{
    _label_enumeration = label_enumeration;
}

void data_instance::set_feature_vector(std::vector<uint8_t> *feature_vector)
{
    _feature_vector = feature_vector;
}

void data_instance::set_distance(double distance)
{
    _distance = distance;
}

uint8_t data_instance::get_label(void) const
{
    return _label;
}
uint8_t data_instance::get_label_enumeration(void) const
{
    return _label_enumeration;
}

int data_instance::get_feature_vector_size(void) const
{
    return _feature_vector->size();
}

const std::vector<uint8_t> *data_instance::get_feature_vector(void) const
{
    return _feature_vector;
}

double data_instance::get_distance(void) const
{
    return _distance;
}

void data_instance::append_to_feature_vector(const uint8_t value)
{
    _feature_vector->push_back(value);
}

//?  Data handling

data_handler::data_handler(void)
{
    std::cout << "Start data handling" << std::endl;
    _data_array = new std::vector<data_instance *>;
    _training_data = new std::vector<data_instance *>;
    _test_data = new std::vector<data_instance *>;
    _validation_data = new std::vector<data_instance *>;
}

data_handler::~data_handler(void)
{
    std::cout << "Stop data handling" << std::endl;
    for (auto entry = _data_array->begin(); entry != _data_array->end(); entry++)
        delete *entry;
    delete _data_array;
    delete _training_data;
    delete _test_data;
    delete _validation_data;
}

void data_handler::run(int ac, char **av)
{
    switch (ac) {
        case 3: if (av[1] != nullptr && av[2] != nullptr) {
            read_feature_vector(av[1]);
            read_feature_labels(av[2]);
            split_data();
            number_classes();
        } else throw error("Error : no file specified"); break;
        default: throw error("Error : invalid arguments"); break;
    }
}

void data_handler::read_feature_vector(const std::string &content)
{
    std::fstream file(content);
    if (!file) throw error("Error : cannot open file");
    std::array<uint32_t, 4> header;
    std::array<unsigned char, 4> bytes;
    for (std::size_t y = 0; y < header.size(); y++)
        if (file.read(reinterpret_cast<char *>(&bytes), sizeof(bytes)))
            header.at(y) = little_endian_converter(bytes);
        else throw error("Error : Failed to read " + content + " header");
    std::cout << "Done getting feature file header" << std::endl;
    const int image_size = header.at(2) * header.at(3);
    for (std::size_t y = 0; y < header.at(1); y++) {
        data_instance *instance = new data_instance();
        std::array<uint8_t, 1> element;
        for (int x = 0; x < image_size; x++) {
            if (file.read(reinterpret_cast<char *>(&element), sizeof(element)))
                instance->append_to_feature_vector(element.at(0));
            else throw error("Error : failed to read " + content + " element");
        }
        _data_array->push_back(instance);
    }
    std::cout << "Read and stored " << _data_array->size() << " feature vectors" << std::endl;
}

void data_handler::read_feature_labels(const std::string &content)
{
    std::fstream file(content);
    if (!file) throw error("Error : cannot open file");
    std::array<uint32_t, 2> header;
    std::array<unsigned char, 4> bytes;
    for (std::size_t y = 0; y < header.size(); y++)
        if (file.read(reinterpret_cast<char *>(&bytes), sizeof(bytes)))
            header.at(y) = little_endian_converter(bytes);
        else throw error("Error : Failed to read " + content + " header");
    std::cout << "Done getting label file header" << std::endl;
    for (std::size_t y = 0; y < header.at(1); y++) {
        std::array<uint8_t, 1> element;
        if (file.read(reinterpret_cast<char *>(&element), sizeof(element)))
            _data_array->at(y)->set_label(element.at(0));
        else throw error("Error : failed to read " + content + " element");
    }
    std::cout << "Read and stored label vectors" << std::endl;
}

void data_handler::split_data(void)
{
    std::unordered_set<int> indexes_path;
    int train_size = _data_array->size() * TRAIN_SET_PERCENT;
    int test_size = _data_array->size() * TEST_SET_PERCENT;
    int validation_size = _data_array->size() * VALIDATION_PERCENT;
    map_indexes(indexes_path, _training_data, train_size);
    map_indexes(indexes_path, _test_data, test_size);
    map_indexes(indexes_path, _validation_data, validation_size);
    std::cout << "Training data size : " << _training_data->size() << std::endl;
    std::cout << "Test data size : " << _test_data->size() << std::endl;
    std::cout << "Validation data size : " << _validation_data->size() << std::endl;
}

void data_handler::map_indexes(std::unordered_set<int> &set, std::vector<data_instance *> *instance, int data_size)
{
    for (int count = 0; count < data_size; count++) {
        const int random_index = rand() % _data_array->size();
        if (set.find(random_index) == set.end()) {
            instance->push_back(_data_array->at(random_index));
            set.insert(random_index);
        }
    }
}

void data_handler::number_classes(void)
{
    int count = 0;
    for (std::size_t y = 0; y < _data_array->size(); y++)
        if (_class_map.find(_data_array->at(y)->get_label()) == _class_map.end()) {
            _class_map[_data_array->at(y)->get_label()] = count;
            _data_array->at(y)->set_label_enumeration(count);
            count++;
        }
    _number_classes = count;
    std::cout << "Extracted " << _number_classes << " unique classes" << std::endl;
}

uint32_t data_handler::little_endian_converter(std::array<unsigned char, 4> bytes)
{
    return (static_cast<uint32_t>(bytes.at(0) << 24 | bytes.at(1) << 16 | bytes.at(2) << 8 | bytes.at(3)));
}

std::vector<data_instance *> *data_handler::get_training_data(void) const
{
    return _training_data;
}

std::vector<data_instance *> *data_handler::get_test_data(void) const
{
    return _test_data;
}

std::vector<data_instance *> *data_handler::get_validation_data(void) const
{
    return _validation_data;
}
