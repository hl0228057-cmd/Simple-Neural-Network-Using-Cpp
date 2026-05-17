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

class NeuralNetwork {
    private:
        int iterations;
        double learningRate;

        vector<vector<double>> hiddenWeights;
        vector<double> hiddenBiases;

        vector<vector<double>> outputWeights;
        vector<double> outputBiases;

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

        vector<double> softmax(const vector<double>& values) {
            vector<double> result(values.size());

            double maxValue = *max_element(values.begin(), values.end());

            double sum = 0.0;
            for (int i = 0; i < values.size(); i++) {
                result[i] = exp(values[i] - maxValue);
                sum += result[i];
            }

            for (int i = 0; i < result.size(); i++) {
                result[i] /= sum;
            }

            return result;
        }

        double relu(double x) {
            return std::max(0.0, x);
        }

        double reluDerivative(double x) {
            return x > 0 ? 1 : 0;
        }

        double hiddenActivation(double x) {
            return relu(x);
        }

        double hiddenActivationDerivative(double x) {
            return reluDerivative(x);
        }

        double crossEntropyLoss(const vector<double>& pred, const vector<double>& expected) {
            double loss = 0.0;

            for (int i = 0; i < pred.size(); i++) {
                double safePred = max(1e-9, min(pred[i], 1 - 1e-9));
                loss += -expected[i] * log(safePred);
            }

            return loss;
        }

    public:
        // default values: 3000 iterations, 4e-3 learning rate
        NeuralNetwork(int iterationsInput = 3000, double learningRateInput = 4e-3) {
            iterations = iterationsInput;
            learningRate = learningRateInput;
        }

        void fit(const vector<vector<double>>& unnormalizedInputs, const vector<vector<double>>& expectedOutputs) {
            vector<vector<double>> normalizedInputs = normalizeTrainingInputs(unnormalizedInputs);
            int n = normalizedInputs.size();

            // Initialize hidden/output weights and biases, small non-zero values break symmetry
            hiddenWeights = {
                {0.2, -0.1, 0.4, 0.3},
                {-0.3, 0.5, 0.1, -0.2},
                {0.4, 0.2, -0.5, 0.1}
            };

            hiddenBiases = {0.1, -0.1, 0.05};

            outputWeights = {
                {0.3, -0.4, 0.2},
                {-0.2, 0.1, 0.4},
                {0.5, -0.3, 0.1}
            };

            outputBiases = {0.1, -0.1, 0.05};

            double currentLoss = 0;

            for (int epoch = 0; epoch < iterations; epoch++) {
                currentLoss = 0;
                for (int dataPointIdx = 0; dataPointIdx < n; dataPointIdx++) {
                    // Forward pass
                    vector<double> normalizedInput = normalizeOneInput(unnormalizedInputs[dataPointIdx]);

                    vector<double> hiddenOutputs;
                    for (int hiddenIdx = 0; hiddenIdx < hiddenWeights.size(); hiddenIdx++) {
                        double z = dotProduct(hiddenWeights[hiddenIdx], normalizedInput) + hiddenBiases[hiddenIdx];

                        hiddenOutputs.push_back(hiddenActivation(z));
                    }

                    vector<double> outputZs;
                    for (int outputIdx = 0; outputIdx < outputWeights.size(); outputIdx++) {
                        double outputZ = dotProduct(outputWeights[outputIdx], hiddenOutputs) + outputBiases[outputIdx];

                        outputZs.push_back(outputZ);
                    }

                    vector<double> finalOutputs = softmax(outputZs);
                    vector<double> outputErrors(finalOutputs.size(), 0.0);
                    for (int outputIdx = 0; outputIdx < finalOutputs.size(); outputIdx++) {
                        outputErrors[outputIdx] = finalOutputs[outputIdx] - expectedOutputs[dataPointIdx][outputIdx];
                    }

                    // compute ONE multiclass loss
                    currentLoss += crossEntropyLoss(
                        finalOutputs,
                        expectedOutputs[dataPointIdx]
                    );

                    // store hidden errors BEFORE changing output weights
                    vector<double> hiddenErrors(hiddenOutputs.size(), 0.0);

                    for (int hiddenIdx = 0; hiddenIdx < hiddenOutputs.size(); hiddenIdx++) {
                        double totalHiddenError = 0;

                        for (int outputIdx = 0; outputIdx < finalOutputs.size(); outputIdx++) {
                            totalHiddenError += outputErrors[outputIdx] * outputWeights[outputIdx][hiddenIdx];
                        }

                        hiddenErrors[hiddenIdx] = totalHiddenError * hiddenActivationDerivative(hiddenOutputs[hiddenIdx]);
                    }

                    // update output layer
                    for (int outputIdx = 0; outputIdx < outputWeights.size(); outputIdx++) {
                        for (int hiddenIdx = 0; hiddenIdx < hiddenOutputs.size(); hiddenIdx++) {
                            outputWeights[outputIdx][hiddenIdx] -=
                                learningRate * outputErrors[outputIdx] * hiddenOutputs[hiddenIdx];
                        }

                        outputBiases[outputIdx] -= learningRate * outputErrors[outputIdx];
                    }

                    // apply hiddenErrors to hiddenWeights
                    for (int hiddenIdx = 0; hiddenIdx < hiddenWeights.size(); hiddenIdx++) {
                        for (int inputIdx = 0; inputIdx < normalizedInput.size(); inputIdx++) {
                            hiddenWeights[hiddenIdx][inputIdx] -= learningRate * hiddenErrors[hiddenIdx] * normalizedInput[inputIdx];
                        }

                        hiddenBiases[hiddenIdx] -= learningRate * hiddenErrors[hiddenIdx];
                    }
                }
                currentLoss /= n;
                if (epoch % 500 == 0 || epoch == iterations - 1) {
                    cout << "Epoch " << epoch
                        << " Loss: " << currentLoss
                        << "\n";
                }
            }
        }

        vector<double> predict(const vector<double>& features) {
            vector<double> normalizedInput = normalizeOneInput(features);

            vector<double> hiddenOutputs;

            for (int hiddenIdx = 0; hiddenIdx < hiddenWeights.size(); hiddenIdx++) {
                double z = dotProduct(hiddenWeights[hiddenIdx], normalizedInput) + hiddenBiases[hiddenIdx];

                hiddenOutputs.push_back(hiddenActivation(z));
            }

            vector<double> outputZs;

            for (int outputIdx = 0; outputIdx < outputWeights.size(); outputIdx++) {
                double outputZ = dotProduct(outputWeights[outputIdx], hiddenOutputs) + outputBiases[outputIdx];
                outputZs.push_back(outputZ);
            }

            vector<double> finalOutputs = softmax(outputZs);
            return finalOutputs;
        }
};

int main() {
    vector<vector<double>> inputs = {
        // setosa
        {5.1, 3.5, 1.4, 0.2},
        {4.9, 3.0, 1.4, 0.2},
        {5.0, 3.6, 1.4, 0.2},
        {5.4, 3.9, 1.7, 0.4},

        // versicolor
        {7.0, 3.2, 4.7, 1.4},
        {6.4, 3.2, 4.5, 1.5},
        {6.9, 3.1, 4.9, 1.5},
        {5.5, 2.3, 4.0, 1.3},

        // virginica
        {6.3, 3.3, 6.0, 2.5},
        {5.8, 2.7, 5.1, 1.9},
        {7.1, 3.0, 5.9, 2.1},
        {6.5, 3.0, 5.8, 2.2}
    };

    vector<vector<double>> expectedOutputs = {
        // setosa
        {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},

        // versicolor
        {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0},

        // virginica
        {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}
    };

    NeuralNetwork model;

    model.fit(inputs, expectedOutputs);
    vector<vector<double>> tests = {
        {5.0, 3.4, 1.5, 0.2}, // expected: setosa
        {6.1, 2.9, 4.7, 1.4}, // expected: versicolor
        {6.7, 3.1, 5.6, 2.4}, // expected: virginica
        {5.9, 3.0, 5.1, 1.8}  // expected: virginica
    };

    for (int test = 0; test < 4; test++) {
        vector<double> pred = model.predict(tests[test]);
        cout << pred[0] << " "
            << pred[1] << " "
            << pred[2] << "\n";

        int predictedClass = 0;

        for (int i = 1; i < pred.size(); i++) {
            if (pred[i] > pred[predictedClass]) {
                predictedClass = i;
            }
        }

        if (predictedClass == 0) cout << "setosa\n";
        else if (predictedClass == 1) cout << "versicolor\n";
        else cout << "virginica\n";
    }

    return 0;
}