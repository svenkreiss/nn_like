#include "nn_like.h"

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int n_layers;
int* layer_units;
double** states;
double** deltas;
double*** weights;
double*** vars;


void nn_like(int _n_layers, int* _layer_units) {
    n_layers = _n_layers;
    printf("Layers = %d\n", n_layers);

    // reserve memory
    layer_units = (int*)malloc(n_layers * sizeof(int));
    states = (double**)malloc(n_layers * sizeof(double*));
    deltas = (double**)malloc(n_layers * sizeof(double*));
    weights = (double***)malloc((n_layers-1) * sizeof(double**));
    vars = (double***)malloc((n_layers-1) * sizeof(double**));

    // initialize
    for (int n=0; n < n_layers; n++) {
        layer_units[n] = _layer_units[n];
        printf("Layer units[%d] = %d\n", n, layer_units[n]);
    }
    for (int n=0; n < n_layers; n++) {
        states[n] = (double*)malloc(layer_units[n] * sizeof(double));
        deltas[n] = (double*)malloc(layer_units[n] * sizeof(double));
    }
    for (int n=0; n < n_layers-1; n++) {
        weights[n] = (double**)malloc(layer_units[n] * sizeof(double*));
        vars[n] = (double**)malloc(layer_units[n] * sizeof(double*));
        for (int r=0; r < layer_units[n]; r++) {
            weights[n][r] = (double*)malloc(layer_units[n+1] * sizeof(double));
            vars[n][r] = (double*)malloc(layer_units[n+1] * sizeof(double));
            for (int c=0; c < layer_units[n+1]; c++) {
                weights[n][r][c] = 0.5;
                vars[n][r][c] = 1.0;
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

double logistic_function_prime(double i) {
    return logistic_function(i)*(1.0 - logistic_function(i));
}

double logistic_function_inverse(double i) {
    return -log((1.0-i) / i);
}

double (*f)(double) = &logistic_function;  // activation function
double (*f_inv)(double) = &logistic_function_inverse;  // inverse
double (*f_prime)(double) = &logistic_function_prime;  // derivative


void forward_deterministic(double* input, double* output) {
    // copy input to initial state
    memcpy(states[0], input, sizeof(double)*layer_units[0]);

    // process layers
    for (int l=0; l < n_layers-1; l++) {
        for (int uo=0; uo < layer_units[l+1]; uo++) {
            states[l+1][uo] = 0.0;
            for (int ui=0; ui < layer_units[l]; ui++) {
                states[l+1][uo] += states[l][ui] * weights[l][ui][uo];
            }
            states[l+1][uo] = (*f)(states[l+1][uo]);
        }
    }

    // return output
    memcpy(output, states[n_layers-1], sizeof(double)*layer_units[n_layers-1]);
}

void forward(double* input, double* output) {
    // copy input to initial state
    memcpy(states[0], input, sizeof(double)*layer_units[0]);

    // process layers
    for (int l=0; l < n_layers-1; l++) {
        for (int uo=0; uo < layer_units[l+1]; uo++) {
            states[l+1][uo] = 0.0;
            for (int ui=0; ui < layer_units[l]; ui++) {
                // Box-Muller for Gauss rand number
                double U = (double)rand() / RAND_MAX;
                double V = (double)rand() / RAND_MAX;
                double r = sqrt(-2.0 * log(U)) * cos(2.0*M_PI*V);
                double w_obs = sqrt(vars[l][ui][uo])*r + weights[l][ui][uo];
                states[l+1][uo] += states[l][ui] * w_obs;
            }
            states[l+1][uo] = (*f)(states[l+1][uo]);
        }
    }

    // return output
    memcpy(output, states[n_layers-1], sizeof(double)*layer_units[n_layers-1]);
}

void backprop_deterministic(double* output, double* target_output, double eta) {
    /*
     * eta is the learning rate.
     */

    // init output deltas
    for (int i=0; i < layer_units[n_layers-1]; i++) {
        deltas[n_layers-1][i] = (target_output[i] - states[n_layers-1][i]);

        // multiply by f'(net)
        double net = (*f_inv)(states[n_layers-1][i]);
        deltas[n_layers-1][i] *= (*f_prime)(net);
    }

    // backpropagate hidden deltas
    for (int l=n_layers-2; l >= 0; l--) {
        for (int i=0; i < layer_units[l]; i++) {
            deltas[l][i] = 0.0;
            for (int o=0; o < layer_units[l+1]; o++) {
                deltas[l][i] += deltas[l+1][o] * weights[l][i][o];
            }

            // multiply by f'(net)
            double net = (*f_inv)(states[l+1][i]);
            deltas[l][i] *= (*f_prime)(net);
        }
    }

    // update weights
    for (int l=0; l <= n_layers-2; l++) {
        for (int i=0; i < layer_units[l]; i++) {
            for (int o=0; o < layer_units[l+1]; o++) {
                weights[l][i][o] += eta * deltas[l+1][o] * states[l+1][o];
            }
        }
    }
}

int output_size(void) {
    return layer_units[n_layers-1];
}


void print_states(void) {
    for (int n=0; n < n_layers; n++) {
        for (int u=0; u < layer_units[n]; u++) {
            printf("%.2f ", states[n][u]);
        }
        printf("\n");
    }
}

void print_connections(void) {
    for (int n=0; n < n_layers-1; n++) {
        for (int ui=0; ui < layer_units[n]; ui++) {
            for (int uo=0; uo < layer_units[n+1]; uo++) {
                printf("%.2f+/-%.2f ", weights[n][ui][uo], sqrt(vars[n][ui][uo]));
            }
            printf("\n");
        }
        printf("\n\n");
    }
}
