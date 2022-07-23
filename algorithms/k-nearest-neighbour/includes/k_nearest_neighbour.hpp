#ifndef K_NEAREST_NEIGHBOUR
#define K_NEAREST_NEIGHBOUR

#include "../../../data_handler/includes/data_handler.hpp"

#include <cmath>
#include <limits>
#include <iomanip>

class k_nearest_neighbour {
    public:
        k_nearest_neighbour(const k_nearest_neighbour &) = delete;
        k_nearest_neighbour(int);
        k_nearest_neighbour();
        ~k_nearest_neighbour();

        void find_k_nearest(data_instance *);
        
        void set_training_data(std::vector<data_instance *> *);
        void set_test_data(std::vector<data_instance *> *);
        void set_validation_data(std::vector<data_instance *> *);
        void set_k(int);

        int predict(void);
        double calculate_distances(data_instance *, data_instance *);
        double validate_performances(void);
        double test_performances(void);

    protected:
    private:
        int _k;
        std::vector<data_instance *> *_neighbours;
        std::vector<data_instance *> *_training_data;
        std::vector<data_instance *> *_test_data;
        std::vector<data_instance *> *_validation_data;
};

#endif /* K_NEAREST_NEIGHBOUR */
