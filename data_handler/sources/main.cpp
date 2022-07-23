#include "../includes/data_handler.hpp"

int main(int ac, char **av)
{
    try {
        data_handler instance;
        instance.run(ac, av);
    } catch(const std::exception &error) {
        std::cerr << error.what() << std::endl;
    }
}
