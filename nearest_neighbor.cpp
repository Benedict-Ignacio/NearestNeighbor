#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

// Name: Benedict Ignacio (bigna003)
// Datasets: Small 48, Large 70

double accuracy(const std::vector<std::vector<double>>& database, const std::vector<unsigned>& feature_list) {
    // If we have no features in the feature list, return default rate (number of most common-class instances/total instances)
    if (feature_list.empty()) {
        double class_1 = 0.0;
        double class_2 = 0.0;
        for (unsigned k = 0; k < database.size(); k++) {
            if (database.at(k).at(0) == 1.0) {
                class_1 += 1.0;
            }
            else {
                class_2 += 1.0;
            }
        }

        if (class_1 < class_2) {
            return (class_2/database.size())*100;
        }

        return (class_1/database.size())*100;
    }

    // Find Nearest Neighbor using Euclidean distance and Leave-One-Out cross validation
    double success = 0.0;
    for (unsigned k = 0; k < database.size(); k++) {
        std::vector<double> test = database.at(k);
        std::vector<double> closest;
        double closest_distance = std::numeric_limits<double>::max();
        for (unsigned i = 0; i < database.size(); i++) {
            if (i != k) {
                std::vector<double> curr = database.at(i);
                double distance = 0;
                for (unsigned j = 0; j < feature_list.size(); j++) {
                    unsigned index = feature_list.at(j);
                    distance += pow(test.at(index)-curr.at(index), 2);
                }
                distance = sqrt(distance);
                if (closest_distance > distance) {
                    closest_distance = distance;
                    closest = curr;
                }
            }
        }

        if (test.at(0) == closest.at(0)) {
            success += 1.0;
        } 
    }

    return (success/database.size())*100;
}

double print_features(const std::vector<std::vector<double>>& database, const std::vector<unsigned>& feature_list) {
    if (!feature_list.empty()) {
        std::cout << "{";
        for (unsigned i = 0; i < feature_list.size()-1; i++) {
            std::cout << feature_list.at(i) << ",";
        }
        std::cout << feature_list.at(feature_list.size()-1) << "}";
    }
    else {
        std::cout << "{}";
    }

    double success = accuracy(database, feature_list);
    return success;
}

int main(int arg, char* argv[]) {
    std::string file = argv[1];
    int algorithm = std::stoi(argv[2]);
    std::ifstream data;

    data.open(argv[1]);

    if (!data.is_open()) {
        std::cout << "Error reading " << file << std::endl;
        return 1;
    }
    else {
        std::cout << "Successfully read " << file << std::endl;
    }

    std::string curr_line;
    std::vector<std::vector<double>> database;
    while (getline(data, curr_line, '\n')) {
        std::vector<double> line_data;
        std::stringstream in(curr_line);
        double input;
        
        while (in >> input) {
            line_data.push_back(input);
        }

        database.push_back(line_data);
    }

    // These vectors essentially flag when a feature is used so that it may not be repeated
    // For forward selection, Forward vectors flag to 1 until the entire vector is the backward vector (all the features are used)
    // For backward elimination, Backward vectors flag to 0 until the entire vector is the forward vector (all features were eliminated)
    std::vector<bool> forward(database.at(0).size(), 0);
    std::vector<bool> backward(database.at(0).size(), 1);
    
    std::vector<unsigned> feature_list;
    std::vector<unsigned> best_list;

    int features = database.at(0).size()-1;
    int instances = database.size();

    std::vector<unsigned> all_features;
    for (unsigned i = 1; i < database.at(0).size(); i++) {
        all_features.push_back(i);
    }
    double all_features_accuracy = accuracy(database, all_features);

    std::cout << "This dataset has " << features << " features (not including the class attribute), with " << instances << " instances." << std::endl << std::endl;
    std::cout << "Running nearest neighbor with all " << features << " features, using \"leaving-one-out\" evaluation, I get an accuracy of " << all_features_accuracy << "%" << std::endl << std::endl;

    // Forward Selection
    if (algorithm == 1) {
        std::vector<unsigned> curr_list = {};
        feature_list = curr_list;
        best_list = curr_list;
        forward.at(0) = 1;

        std::cout << "Beginning search." << std::endl << std::endl;

        while (forward != backward) {
            double best_accuracy = 0;
            unsigned best_index = 0;
            for (unsigned i = 1; i < database.at(0).size(); i++) {
                if (!forward.at(i)) {
                    std::vector<unsigned> temp_list = curr_list;
                    temp_list.push_back(i);
                    std::cout << "\tUsing feature(s) ";
                    double curr = print_features(database, temp_list);
                    std::cout << " accuracy is " << curr << "%" << std::endl;
                    if (best_accuracy < curr) {
                        best_accuracy = curr;
                        best_index = i;
                    }
                }
            }

            curr_list.push_back(best_index);
            feature_list = curr_list;
            if (accuracy(database, best_list) < accuracy(database, feature_list) || best_list.empty()) {
                best_list = feature_list;
            }
            else {
                std::cout << std::endl << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)";
            }
            std::cout << std::endl << "Feature set ";
            print_features(database, feature_list);
            std::cout << " was best, accuracy is " << best_accuracy << "%" << std::endl << std::endl;
            forward.at(best_index) = 1;
        }
    }
    // Backward Elimination
    else {
        std::vector<unsigned> curr_list;
        for (unsigned i = 1; i < database.at(0).size(); i++) {
            curr_list.push_back(i);
        }
        feature_list = curr_list;
        best_list = curr_list;
        backward.at(0) = 0;

        std::cout << "Beginning search." << std::endl << std::endl;

        while (forward != backward) {
            double best_accuracy = 0;
            unsigned best_index;
            for (unsigned i = 0; i < curr_list.size(); i++) {
                if (backward.at(curr_list.at(i))) {
                    std::vector<unsigned> temp_list = curr_list;
                    temp_list.erase(temp_list.begin() + i);
                    std::cout << "\tUsing feature(s) ";
                    double curr = print_features(database, temp_list);
                    std::cout << " accuracy is " << curr << "%" << std::endl;
                    if (best_accuracy < curr) {
                        best_accuracy = curr;
                        best_index = curr_list.at(i);
                    }
                }
            }

            unsigned k, j;
            for (k = 0; k < curr_list.size(); k++) {
                if (curr_list.at(k) == best_index) {
                    j = k;
                }
            }

            curr_list.erase(curr_list.begin() + j);
            feature_list = curr_list;
            if (accuracy(database, best_list) < accuracy(database, feature_list)) {
                best_list = feature_list;
            }
            else {
                std::cout << std::endl << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)";
            }
            std::cout << std::endl << "Feature set ";
            print_features(database, feature_list);
            std::cout << " was best, accuracy is " << best_accuracy << "%" << std::endl << std::endl;
            backward.at(best_index) = 0;
        }
    }

    std::cout << "Finished search!! The best feature subset is ";
    double best_accuracy = print_features(database, best_list);
    std::cout << ", which has an accuracy of " << best_accuracy << "%" << std::endl;

    return 0;
}