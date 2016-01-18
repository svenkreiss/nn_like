void nn_like(int _layers, int* _layer_units);

void random_weights(void);
void fixed_weights(double weight, double variance);

void print_states(void);
void print_connections(void);

int output_size(void);
void forward_deterministic(double* input, double* output);
void forward(double* input, double* output);
void backprop_deterministic(double* output, double* target_output, double eta);
