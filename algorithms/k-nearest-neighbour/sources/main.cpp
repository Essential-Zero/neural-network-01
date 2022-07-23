#include "../includes/k_nearest_neighbour.hpp"

int main(int ac, char **av)
{
    try {
        data_handler *handler_instance = new data_handler();
        handler_instance->run(ac, av);

        k_nearest_neighbour *neighbour_instance = new k_nearest_neighbour();
        neighbour_instance->set_training_data(handler_instance->get_training_data());
        neighbour_instance->set_test_data(handler_instance->get_test_data());
        neighbour_instance->set_validation_data(handler_instance->get_validation_data());

        double performance, best_performance = 0;
        int best_k = 1;

        for (int y = 1; y <= 4; y++) {
            if (y == 1) {
                neighbour_instance->set_k(y);
                performance = neighbour_instance->validate_performances();
                best_performance = performance;
            } else {
                neighbour_instance->set_k(y);
                performance = neighbour_instance->validate_performances();
                performance > best_performance && ((best_performance = performance), (best_k = y));
            }
        }
        neighbour_instance->set_k(best_k);
        neighbour_instance->test_performances();

        delete handler_instance;
        delete neighbour_instance;
    } catch(const std::exception &error) {
        std::cerr << error.what() << std::endl;
    }
}
