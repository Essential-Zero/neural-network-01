#ifndef DATA_HANDLER
#define DATA_HANDLER

#include <unordered_set>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <map>

#define TRAIN_SET_PERCENT 0.75
#define TEST_SET_PERCENT 0.20
#define VALIDATION_PERCENT 0.05

class error : public std::runtime_error {
    public:
        error(const std::string &message) : std::runtime_error(message) {}
};

class data_instance {
    public:
        data_instance(const data_instance &) = delete;
        explicit data_instance();
        ~data_instance();

        void set_label(uint8_t);
        void set_label_enumeration(int);
        void set_feature_vector(std::vector<uint8_t> *);
        void set_distance(double);

        uint8_t get_label(void) const;
        uint8_t get_label_enumeration(void) const;
        int get_feature_vector_size(void) const;
        const std::vector<uint8_t> *get_feature_vector(void) const;
        double get_distance(void) const;

        void append_to_feature_vector(const uint8_t);

    protected:
    private:
        uint8_t _label;
        int _label_enumeration;
        std::vector<uint8_t> *_feature_vector;
        double _distance;
};

class data_handler {
    public:
        data_handler(const data_handler &) = delete;
        explicit data_handler();
        ~data_handler();

        void run(int, char **);

        void read_feature_vector(const std::string &);
        void read_feature_labels(const std::string &);

        void split_data(void);
        void map_indexes(std::unordered_set<int> &, std::vector<data_instance *> *, int);

        void number_classes(void);

        uint32_t little_endian_converter(std::array<unsigned char, 4>);

        std::vector<data_instance *> *get_training_data(void) const;
        std::vector<data_instance *> *get_test_data(void) const;
        std::vector<data_instance *> *get_validation_data(void) const;

    protected:
    private:
        std::vector<data_instance *> *_data_array;
        std::vector<data_instance *> *_training_data;
        std::vector<data_instance *> *_test_data;
        std::vector<data_instance *> *_validation_data;

        int _number_classes;
        int _feature_vector_size;
        std::map<uint8_t, int> _class_map;

};

#endif /* DATA_HANDLER */
