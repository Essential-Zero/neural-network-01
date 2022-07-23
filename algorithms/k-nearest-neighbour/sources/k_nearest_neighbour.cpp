#include "../includes/k_nearest_neighbour.hpp"

k_nearest_neighbour::k_nearest_neighbour(int k) : _k(k) {}

k_nearest_neighbour::k_nearest_neighbour() {}

k_nearest_neighbour::~k_nearest_neighbour() { delete _neighbours; }

void k_nearest_neighbour::find_k_nearest(data_instance *query_point)
{
    _neighbours = new std::vector<data_instance *>;
    double minima = std::numeric_limits<double>::max();
    double previous_minima = minima;
    int index = 0;
    for (int y = 0; y < _k; y++) {
        if (y == 0) {
            for (std::size_t x = 0; x < _training_data->size(); x++) {
                double distance = calculate_distances(query_point, _training_data->at(x));
                _training_data->at(x)->set_distance(distance);
                if (distance < minima) {
                    minima = distance;
                    index = x;
                }
            }
            _neighbours->push_back(_training_data->at(index));
            previous_minima = minima;
            minima = std::numeric_limits<double>::max();
        } else {
            for (std::size_t x = 0; x < _training_data->size(); x++) {
                double distance = _training_data->at(x)->get_distance();
                _training_data->at(x)->set_distance(distance);
                if (distance >  previous_minima && distance < minima) {
                    minima = distance;
                    index = x;
                }
            }
            _neighbours->push_back(_training_data->at(index));
            previous_minima = minima;
            minima = std::numeric_limits<double>::max();
        }
    }
}

void k_nearest_neighbour::set_training_data(std::vector<data_instance *> *training_data)
{
    _training_data = training_data;
}

void k_nearest_neighbour::set_test_data(std::vector<data_instance *> *test_data)
{
    _test_data = test_data;
}

void k_nearest_neighbour::set_validation_data(std::vector<data_instance *> *validation_data)
{
    _validation_data = validation_data;
}

void k_nearest_neighbour::set_k(int k)
{
    _k = k;
}

int k_nearest_neighbour::predict(void)
{
    std::map<uint8_t, int> class_frequency;
    for (std::size_t y = 0; y < _neighbours->size(); y++) {
        if (class_frequency.find(_neighbours->at(y)->get_label()) == class_frequency.end()) {
            class_frequency[_neighbours->at(y)->get_label()] = 1;
        } else {
            class_frequency[_neighbours->at(y)->get_label()] += 1;
        }
    }
    int best_class = 0;
    int maxima = 0;
    for (const auto &entry : class_frequency) {
        if (entry.second > maxima) {
            maxima = entry.second;
            best_class = entry.first;
        }
    }
    _neighbours->clear();
    return best_class;
}

double k_nearest_neighbour::calculate_distances(data_instance *query_point, data_instance *input)
{
    if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
        throw error("Error : vector size mismatch");
    double distance = 0.0;
    #ifdef EUCLID
        for (int y = 0; y < query_point->get_feature_vector_size(); y++)
            distance += pow(query_point->get_feature_vector()->at(y) - input->get_feature_vector()->at(y), 2);
        distance = sqrt(distance);
    #elif defined MANHATTAN
        // Manhattan implementation
    #endif
    return distance;
}

double k_nearest_neighbour::validate_performances(void)
{
    double current_performance = 0;
    int count = 0;
    int data_index = 0;
    for (data_instance *query_point : *_validation_data) {
        find_k_nearest(query_point);
        int prediction = predict();
        std::cout << "Prediction = " << prediction << "\t\tLabel = " << +query_point->get_label();
        prediction == query_point->get_label() && (count += 1);
        data_index++;
        std::cout << "\tCurrent performance = " << std::fixed << std::setprecision(3) <<
        (double)((double)count * 100 / (double)data_index) << " %" << std::endl;
    }
    current_performance = (double)((double)count * 100 / (double)_validation_data->size());
    std::cout << "Validation performance = " << std::fixed << std::setprecision(3) <<
    current_performance << " %" << std::endl;
    return current_performance;
}

double k_nearest_neighbour::test_performances(void)
{
    double current_performance = 0;
    int count = 0;
    for (data_instance *query_point : *_test_data) {
        find_k_nearest(query_point);
        int prediction = predict();
        prediction == query_point->get_label() && (count += 1);
    }
    current_performance = (double)((double)count * 100 / (double)_test_data->size());
    std::cout << "Tested performance = " << std::fixed << std::setprecision(3) <<
    current_performance << " %" << std::endl;
    return current_performance;
}
