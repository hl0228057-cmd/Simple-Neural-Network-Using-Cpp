#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

// references to vector instead of copying them, they cannot be changed here because it is a const
double dotProduct(const vector<double>& a, const vector<double>& b) {
    double total = 0;
    // assumes a and b are same size
    for (int i = 0; i < a.size(); i++) {
        total += a[i] * b[i];
    }
    return total;
}

class neuralNetwork {
    private:
        vector<vector<double>> hiddenWeights;
        vector<double> hiddenBiases;

        vector<double> outputWeights;
        double outputBias = 0;

        vector<double> featureMins;
        vector<double> featureMaxes;
        
        vector<vector<double>> normalizeTrainingInputs(const vector<vector<double>>& unnormalizedInputs) {
            vector<vector<double>> normalizedInputs = unnormalizedInputs;

            int n = normalizedInputs.size();
            int numFeatures = normalizedInputs[0].size();

            featureMins.resize(numFeatures);
            featureMaxes.resize(numFeatures);

            for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                double minValue = 1e9;
                double maxValue = -1e9;

                // find min/max
                for (int dataPointIdx = 0; dataPointIdx < n; dataPointIdx++) {
                    double value = normalizedInputs[dataPointIdx][featureIdx];

                    if (value < minValue) {
                        minValue = value;
                    }

                    if (value > maxValue) {
                        maxValue = value;
                    }
                }

                featureMins[featureIdx] = minValue;
                featureMaxes[featureIdx] = maxValue;

                // normalize column
                for (int dataPointIdx = 0; dataPointIdx < n; dataPointIdx++) {
                    normalizedInputs[dataPointIdx][featureIdx] = (normalizedInputs[dataPointIdx][featureIdx] - minValue) / (maxValue - minValue);
                }
            }

            return normalizedInputs;
        }

        vector<double> normalizeOneInput(const vector<double>& unnormalizedInput) {
            vector<double> normalizedInput = unnormalizedInput;

            for (int featureIdx = 0; featureIdx < normalizedInput.size(); featureIdx++) {
                normalizedInput[featureIdx] = (normalizedInput[featureIdx] - featureMins[featureIdx]) / (featureMaxes[featureIdx] - featureMins[featureIdx]);
            }

            return normalizedInput;
        }

        double sigmoidFunction(double z) {
            // 1 / (1 + e^(-z))
            return 1 / (1 + exp(-z));
        }

    public:
        void fit(const vector<vector<double>>& unnormalizedInputs, const vector<double>& expectedOutputs) {
            vector<vector<double>> normalizedInputs = normalizeTrainingInputs(unnormalizedInputs);
            int n = normalizedInputs.size();

            int numFeatures = normalizedInputs[0].size();
            int hiddenNeurons = 3;

            hiddenWeights = {
                {0.0, 0.0, -12.0, -12.0},
                {0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0}
            };

            hiddenBiases = {8.0, 0.0, 0.0};

            outputWeights = {12.0, 0.0, 0.0};
            outputBias = -6.0;
        }

        double predict(const vector<double>& features) {
            vector<double> normalizedInput = normalizeOneInput(features);

            vector<double> hiddenOutputs;
            double finalOutput;

            for (int hiddenIdx = 0; hiddenIdx < hiddenWeights.size(); hiddenIdx++) {
                double z = dotProduct(hiddenWeights[hiddenIdx], normalizedInput) + hiddenBiases[hiddenIdx];

                hiddenOutputs.push_back(sigmoidFunction(z));
            }

            double outputZ = dotProduct(outputWeights, hiddenOutputs) + outputBias;
            finalOutput = sigmoidFunction(outputZ);
            return finalOutput;
        }
};

int main() {
    vector<vector<double>> inputs = {
        {5.1, 3.5, 1.4, 0.2}, // setosa
        {4.9, 3.0, 1.4, 0.2}, // setosa
        {5.0, 3.6, 1.4, 0.2}, // setosa
        {5.4, 3.9, 1.7, 0.4}, // setosa
        {4.6, 3.4, 1.4, 0.3}, // setosa

        {7.0, 3.2, 4.7, 1.4}, // not setosa
        {6.4, 3.2, 4.5, 1.5}, // not setosa
        {6.9, 3.1, 4.9, 1.5}, // not setosa
        {5.5, 2.3, 4.0, 1.3}, // not setosa
        {6.5, 2.8, 4.6, 1.5}  // not setosa
    };

    vector<double> expectedOutputs = {
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0
    };

    neuralNetwork model;

    model.fit(inputs, expectedOutputs);

    double pred = model.predict({5.1, 3.5, 1.4, 0.2});
    cout << pred << "\n";
    if (pred >= 0.5) cout << "setosa\n";
    else cout << "not setosa\n";

    double pred2 = model.predict({6.4, 3.2, 4.5, 1.5});
    cout << pred2 << "\n";
    if (pred2 >= 0.5) cout << "setosa\n";
    else cout << "not setosa\n";

    return 0;
}