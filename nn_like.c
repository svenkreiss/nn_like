#include "nn_like.h"

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int n_layers;
int* layer_units;
double** states;
double* states2_sum;
double** deltas;
double*** weights;
double*** vars;


void nn_like(int _n_layers, int* _layer_units) {
    n_layers = _n_layers;
    printf("Layers = %d\n", n_layers);

    // reserve memory
    layer_units = (int*)malloc(n_layers * sizeof(int));
    states = (double**)malloc(n_layers * sizeof(double*));
    states2_sum = (double*)malloc(n_layers * sizeof(double));
    deltas = (double**)malloc(n_layers * sizeof(double*));
    weights = (double***)malloc((n_layers-1) * sizeof(double**));
    vars = (double***)malloc((n_layers-1) * sizeof(double**));

    // initialize
    for (int l=0; l < n_layers; l++) {
        layer_units[l] = _layer_units[l];
        printf("Layer units[%d] = %d\n", l, layer_units[l]);
    }
    for (int l=0; l < n_layers; l++) {
        states[l] = (double*)malloc((layer_units[l]+1) * sizeof(double));
        deltas[l] = (double*)malloc(layer_units[l] * sizeof(double));
        for (int i=0; i < layer_units[l]; i++) {
            states[l][i] = 0.0;
            deltas[l][i] = 0.0;
        }
        states[l][layer_units[l]] = 1.0;  // bias
    }
    for (int l=0; l < n_layers-1; l++) {
        weights[l] = (double**)malloc((layer_units[l]+1) * sizeof(double*));
        vars[l] = (double**)malloc((layer_units[l]+1) * sizeof(double*));
        for (int i=0; i < (layer_units[l]+1); i++) {
            weights[l][i] = (double*)malloc(layer_units[l+1] * sizeof(double));
            vars[l][i] = (double*)malloc(layer_units[l+1] * sizeof(double));
        }
    }
    random_weights();
}


void random_weights(void) {
    for (int l=0; l < n_layers-1; l++) {
        for (int i=0; i < (layer_units[l]+1); i++) {
            for (int o=0; o < layer_units[l+1]; o++) {
                weights[l][i][o] = -0.4 + 0.8*((double)rand()/RAND_MAX);
                vars[l][i][o] = 1.0;
            }
        }
    }
}


void fixed_weights(double weight, double variance) {
    for (int l=0; l < n_layers-1; l++) {
        for (int i=0; i < (layer_units[l]+1); i++) {
            for (int o=0; o < layer_units[l+1]; o++) {
                weights[l][i][o] = weight;
                vars[l][i][o] = variance;
            }
        }
    }
}


double softplus(double i) {
    return log(1.0 + exp(i));
}

double relu(double i) {
    return i < 0.0 ? 0.0 : i;
}

double logistic_function(double i) {
    return 1.0 / (1.0 + exp(-i));
}
double logistic_function_prime(double o) {
    return o * (1.0-o);
}
double logistic_function_inverse(double i) {
    if (i > 0.999) return 10.0;
    if (i < 0.001) return -10.0;
    return -log((1.0-i) / i);
}
double logistic_function_backprop(double t, double o) {
    // printf("%f -> %f, %f -> %f\n", t, logistic_function_inverse(t), o, logistic_function_inverse(o));
    double delta = logistic_function_inverse(t) - logistic_function_inverse(o);
    // for numerical stability, constrain delta
    if (delta > 10.0) delta = 10.0;
    if (delta < -10.0) delta = -10.0;
    return delta;
}

// double tanh(double i); // already in math library
double tanh_prime(double o) {
    return 1 - o*o;
}
double tanh_inverse(double i) {
    return atanh(i);
}
double tanh_backprop(double t, double o) {
    if (t >= 0.999) t = 0.999;
    if (t < -0.999) t = -0.999;
    if (o > 0.999) o = 0.999;
    if (o < -0.999) o = -0.999;
    // printf("%f -> %f, %f -> %f\n", t, tanh_inverse(t), o, tanh_inverse(o));
    double delta = tanh_inverse(t) - tanh_inverse(o);
    // for numerical stability, constrain delta
    if (delta > 10.0) delta = 10.0;
    if (delta < -10.0) delta = -10.0;
    return delta;
}


double (*f)(double) = &tanh;  // activation function
double (*f_inv)(double) = &tanh_inverse;  // inverse
double (*f_prime)(double) = &tanh_prime;  // derivative
double (*f_backprop)(double, double) = &tanh_backprop; // backprop


void forward_deterministic(double* input, double* output) {
    // copy input to initial state
    memcpy(states[0], input, sizeof(double)*layer_units[0]);
    states2_sum[0] = 0.0;
    for (int i=0; i < layer_units[0]; i++) {
        states2_sum[0] += states[0][i]*states[0][i];
    }

    // process layers
    for (int l=0; l < n_layers-1; l++) {
        for (int o=0; o < layer_units[l+1]; o++) {
            states[l+1][o] = 0.0;
            for (int i=0; i < layer_units[l]; i++) {
                states[l+1][o] += states[l][i] * weights[l][i][o];
            }
            states[l+1][o] = (*f)(states[l+1][o]);
        }

        // update sum of states squared
        states2_sum[l+1] = 0.0;
        for (int o=0; o < layer_units[l+1]; o++) {
            states2_sum[l+1] += states[l+1][o]*states[l+1][o];
        }
    }

    // return output
    memcpy(output, states[n_layers-1], sizeof(double)*layer_units[n_layers-1]);
}


void forward(double* input, double* output) {
    // copy input to initial state
    memcpy(states[0], input, sizeof(double)*layer_units[0]);
    states2_sum[0] = 0.0;
    for (int i=0; i < layer_units[0]; i++) {
        states2_sum[0] += states[0][i]*states[0][i];
    }

    // process layers
    for (int l=0; l < n_layers-1; l++) {
        for (int o=0; o < layer_units[l+1]; o++) {
            states[l+1][o] = 0.0;
            for (int i=0; i < layer_units[l]; i++) {
                // Box-Muller for Gauss rand number
                double U = (double)rand() / RAND_MAX;
                double V = (double)rand() / RAND_MAX;
                double r = sqrt(-2.0 * log(U)) * cos(2.0*M_PI*V);
                double w_obs = sqrt(vars[l][i][o])*r + weights[l][i][o];
                states[l+1][o] += states[l][i] * w_obs;
            }
            states[l+1][o] = (*f)(states[l+1][o]);
        }

        // update sum of states squared
        states2_sum[l+1] = 0.0;
        for (int o=0; o < layer_units[l+1]; o++) {
            states2_sum[l+1] += states[l+1][o]*states[l+1][o];
        }
    }

    // return output
    memcpy(output, states[n_layers-1], sizeof(double)*layer_units[n_layers-1]);
}


void backprop_deterministic_traditional(double* output, double* target_output, double eta) {
    /*
     * eta is the learning rate.
     */

    // init output deltas
    for (int i=0; i < layer_units[n_layers-1]; i++) {
        deltas[n_layers-1][i] = (target_output[i] - states[n_layers-1][i]);

        // multiply by f'(net) [standard way]
        deltas[n_layers-1][i] *= (*f_prime)(states[n_layers-1][i]);
    }

    // backpropagate hidden deltas
    for (int l=n_layers-2; l > 0; l--) {
        for (int i=0; i < layer_units[l]; i++) {
            deltas[l][i] = 0.0;
            for (int o=0; o < layer_units[l+1]; o++) {
                deltas[l][i] += deltas[l+1][o] * weights[l][i][o];
            }

            // multiply by f'(net)
            deltas[l][i] *= (*f_prime)(states[l][i]);
        }
    }

    // update weights
    for (int l=0; l <= n_layers-2; l++) {
        for (int i=0; i < layer_units[l]; i++) {
            for (int o=0; o < layer_units[l+1]; o++) {
                weights[l][i][o] += eta * deltas[l+1][o] * states[l][i];
            }
        }
    }
}


void backprop_deterministic(double* output, double* target_output, double eta) {
    /*
     * eta is the learning rate.
     */

    // init output deltas
    for (int i=0; i < layer_units[n_layers-1]; i++) {
        deltas[n_layers-1][i] = (target_output[i] - states[n_layers-1][i]);

        // [non-traditional] use exact expression to compute how delta
        // propagates through the activation function
        deltas[n_layers-1][i] = (*f_backprop)(
            states[n_layers-1][i] + deltas[n_layers-1][i],
            states[n_layers-1][i]
        );
    }

    // backpropagate hidden deltas
    for (int l=n_layers-2; l > 0; l--) {
        for (int i=0; i < layer_units[l]; i++) {
            deltas[l][i] = 0.0;
            for (int o=0; o < layer_units[l+1]; o++) {
                double w2_sum = 0.0;
                for (int ii=0; ii < layer_units[l]; ii++) {
                    w2_sum += weights[l][ii][o]*weights[l][ii][o];
                }

                // if (w2_sum == 0.0) continue;
                // if (weights[l][i][o]/w2_sum > 1000.0) continue;

                // [non-traditional] backprop deltas weighted by the
                // normalized weight squared.
                // The weight is normalized with respect to all the input
                // units connecting to this particular output.
                deltas[l][i] += deltas[l+1][o] * weights[l][i][o] / w2_sum;
            }

            deltas[l][i] = (*f_backprop)(
                states[l][i] + deltas[l][i],
                states[l][i]
            );
        }
    }

    // update weights
    // for (int l=0; l <= n_layers-2; l++) {
    int l = rand() % (n_layers-1);
    // printf("update layer %i\n", l);
        // [non-traditional] weight update proportional to the square of
        // the input state.

        for (int i=0; i < layer_units[l]; i++) {
            for (int o=0; o < layer_units[l+1]; o++) {
                weights[l][i][o] += eta * deltas[l+1][o] * states[l][i] / states2_sum[l];
            }
        }
    // }
}


int output_size(void) {
    return layer_units[n_layers-1];
}


void print_states(void) {
    for (int l=0; l < n_layers; l++) {
        printf("States(deltas) layer %i: ", l);
        for (int i=0; i < layer_units[l]; i++) {
            printf("%.2f(%.4f) ", states[l][i], deltas[l][i]);
        }
        printf("\n");
    }
}


void print_connections(void) {
    for (int l=0; l < n_layers-1; l++) {
        printf("Weights layer %i:\n", l);
        for (int i=0; i < layer_units[l]; i++) {
            for (int o=0; o < layer_units[l+1]; o++) {
                printf("%.2f+/-%.2f ", weights[l][i][o], sqrt(vars[l][i][o]));
            }
            printf("\n");
        }
        printf("\n\n");
    }
}
