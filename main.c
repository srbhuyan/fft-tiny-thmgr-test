#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <thpool.h>

#define PI 3.14159265358979323846
#define MAX_SIZE 8192
#define THRESHOLD 0.001

typedef double complex cplx;

// Structure for 2D complex array
typedef struct {
    cplx** data;
    int rows;
    int cols;
} Complex2D;

// Thread data structures
typedef struct {
    Complex2D* data;
    int start_row;
    int end_row;
    int cols;
    int inverse;
} RowThreadData;

typedef struct {
    Complex2D* data;
    int start_col;
    int end_col;
    int rows;
    int inverse;
} ColThreadData;

typedef struct {
    Complex2D* data;
    int start_row;
    int end_row;
    int rows;
    int cols;
    double cutoff;
} FilterThreadData;

// Global variables for thread management
int num_threads;
threadpool thpool;

// Function to create 2D complex array
Complex2D* create_complex_2d(int rows, int cols) {
    Complex2D* arr = (Complex2D*)malloc(sizeof(Complex2D));
    arr->rows = rows;
    arr->cols = cols;
    arr->data = (cplx**)malloc(rows * sizeof(cplx*));
    for(int i = 0; i < rows; i++) {
        arr->data[i] = (cplx*)calloc(cols, sizeof(cplx));
    }
    return arr;
}

// Function to free 2D complex array
void free_complex_2d(Complex2D* arr) {
    for(int i = 0; i < arr->rows; i++) {
        free(arr->data[i]);
    }
    free(arr->data);
    free(arr);
}

// Bit reversal for FFT
unsigned int bit_reverse(unsigned int x, int log2n) {
    int n = 0;
    for(int i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Iterative 1D FFT implementation
void fft_1d(cplx* data, int n, int inverse) {
    // Bit reversal permutation
    int log2n = (int)log2(n);
    for(int i = 0; i < n; i++) {
        int j = bit_reverse(i, log2n);
        if(i < j) {
            cplx temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    // Cooley-Tukey FFT
    for(int len = 2; len <= n; len <<= 1) {
        double angle = 2 * PI / len * (inverse ? 1 : -1);
        cplx wlen = cexp(I * angle);

        for(int i = 0; i < n; i += len) {
            cplx w = 1;
            for(int j = 0; j < len/2; j++) {
                cplx u = data[i + j];
                cplx v = data[i + j + len/2] * w;
                data[i + j] = u + v;
                data[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }

    // Scale if inverse
    if(inverse) {
        for(int i = 0; i < n; i++) {
            data[i] /= n;
        }
    }
}

// Thread function for processing rows
void process_rows_thread(void* arg) {
    RowThreadData* data = (RowThreadData*)arg;
    
    for(int i = data->start_row; i < data->end_row; i++) {
        fft_1d(data->data->data[i], data->cols, data->inverse);
    }
}

// Thread function for processing columns
void process_cols_thread(void* arg) {
    ColThreadData* data = (ColThreadData*)arg;
    
    // Allocate temporary array for columns
    cplx* temp = (cplx*)malloc(data->rows * sizeof(cplx));
    
    for(int j = data->start_col; j < data->end_col; j++) {
        // Copy column to temporary array
        for(int i = 0; i < data->rows; i++) {
            temp[i] = data->data->data[i][j];
        }

        // Transform
        fft_1d(temp, data->rows, data->inverse);

        // Copy back
        for(int i = 0; i < data->rows; i++) {
            data->data->data[i][j] = temp[i];
        }
    }
    
    free(temp);
}

// Thread function for frequency domain filtering
void apply_filter_thread(void* arg) {
    FilterThreadData* data = (FilterThreadData*)arg;
    int center_row = data->rows / 2;
    int center_col = data->cols / 2;

    for(int i = data->start_row; i < data->end_row; i++) {
        for(int j = 0; j < data->cols; j++) {
            // Calculate distance from center
            double dist = sqrt(pow(i - center_row, 2) +
                             pow(j - center_col, 2));

            // Apply low-pass filter
            if(dist > data->cutoff) {
                data->data->data[i][j] *= exp(-pow(dist - data->cutoff, 2) / 100.0);
            }
        }
    }
}

// Parallel 2D FFT implementation
void fft_2d_parallel(Complex2D* data, int rows, int cols, int inverse) {
    
    // Transform rows in parallel
    RowThreadData* row_data = (RowThreadData*)malloc(num_threads * sizeof(RowThreadData));
    int rows_per_thread = rows / num_threads;
    int remaining_rows = rows % num_threads;
    
    for(int t = 0; t < num_threads; t++) {
        row_data[t].data = data;
        row_data[t].start_row = t * rows_per_thread;
        row_data[t].end_row = (t + 1) * rows_per_thread;
        if(t == num_threads - 1) {
            row_data[t].end_row += remaining_rows;
        }
        row_data[t].cols = cols;
        row_data[t].inverse = inverse;
        
        thpool_add_work(thpool, *process_rows_thread, &row_data[t]);
    }
    
    thpool_wait(thpool);

    // Transform columns in parallel
    ColThreadData* col_data = (ColThreadData*)malloc(num_threads * sizeof(ColThreadData));
    int cols_per_thread = cols / num_threads;
    int remaining_cols = cols % num_threads;
    
    for(int t = 0; t < num_threads; t++) {
        col_data[t].data = data;
        col_data[t].start_col = t * cols_per_thread;
        col_data[t].end_col = (t + 1) * cols_per_thread;
        if(t == num_threads - 1) {
            col_data[t].end_col += remaining_cols;
        }
        col_data[t].rows = rows;
        col_data[t].inverse = inverse;
        
        thpool_add_work(thpool, *process_cols_thread, &col_data[t]);
    }
    
    thpool_wait(thpool);

    free(row_data);
    free(col_data);
}

// Parallel frequency domain filtering
void apply_frequency_filter_parallel(Complex2D* data, int rows, int cols, double cutoff) {
    FilterThreadData* filter_data = (FilterThreadData*)malloc(num_threads * sizeof(FilterThreadData));
    
    int rows_per_thread = rows / num_threads;
    int remaining_rows = rows % num_threads;
    
    for(int t = 0; t < num_threads; t++) {
        filter_data[t].data = data;
        filter_data[t].start_row = t * rows_per_thread;
        filter_data[t].end_row = (t + 1) * rows_per_thread;
        if(t == num_threads - 1) {
            filter_data[t].end_row += remaining_rows;
        }
        filter_data[t].rows = rows;
        filter_data[t].cols = cols;
        filter_data[t].cutoff = cutoff;
        
        thpool_add_work(thpool, apply_filter_thread, &filter_data[t]);
    }
    
    thpool_wait(thpool);

    free(filter_data);
}

// Signal analysis functions
void analyze_frequency_components(cplx* data, int n) {
    printf("\nFrequency Components Analysis:\n");

    // Find dominant frequencies
    for(int i = 0; i < n/2; i++) {
        double magnitude = cabs(data[i]);
        if(magnitude > THRESHOLD) {
            double frequency = (double)i / n;
            printf("Frequency: %.3f, Magnitude: %.3f\n",
                   frequency, magnitude);
        }
    }
}

// Example of practical usage
void process_signal(double* input_signal, int signal_length) {
    // Ensure signal length is power of 2
    int n = 1;
    while(n < signal_length) n <<= 1;

    // Allocate and initialize complex array
    cplx* data = (cplx*)malloc(n * sizeof(cplx));
    for(int i = 0; i < signal_length; i++) {
        data[i] = input_signal[i] + 0.0 * I;
    }
    for(int i = signal_length; i < n; i++) {
        data[i] = 0;
    }

    // Forward FFT
    fft_1d(data, n, 0);

    // Analyze frequency components
    analyze_frequency_components(data, n);

    // Apply frequency domain filtering
    for(int i = n/4; i < 3*n/4; i++) {
        data[i] *= 0.5; // Attenuate middle frequencies
    }

    // Inverse FFT
    fft_1d(data, n, 1);

    // Copy result back
    for(int i = 0; i < signal_length; i++) {
        input_signal[i] = creal(data[i]);
    }

    free(data);
}

// Parallel 2D image processing
void process_image_parallel(Complex2D* image, int rows, int cols) {
    // Forward 2D FFT (parallel)
    fft_2d_parallel(image, rows, cols, 0);

    // Apply frequency domain filter (parallel)
    apply_frequency_filter_parallel(image, rows, cols, image->rows/4);

    // Inverse 2D FFT (parallel)
    fft_2d_parallel(image, rows, cols, 1);
}

int main(int argc, char * argv[]) {

    if(argc < 2) {
        printf("Usage: %s <size> [num_threads]\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    
    // Set number of threads (default to number of CPU cores)
    if(argc > 2) {
        num_threads = atoi(argv[2]);
    } else {
        num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    }

    // Init thread pool
    thpool = thpool_init(num_threads);

    // Example usage with 1D signal
    int signal_length = size;
    double* test_signal = (double*)malloc(signal_length * sizeof(double));

    // Generate test signal (sum of sinusoids)
    for(int i = 0; i < signal_length; i++) {
        double t = (double)i / signal_length;
        test_signal[i] = sin(2 * PI * 10 * t) +  // 10 Hz component
                        0.5 * sin(2 * PI * 40 * t) + // 40 Hz component
                        0.25 * sin(2 * PI * 90 * t); // 90 Hz component
    }

    // Process 1D signal (not parallelized as it's already fast)
    clock_t start = clock();
    process_signal(test_signal, signal_length);
    clock_t end = clock();

    printf("1D Signal processing time: %f seconds\n",
           ((double)(end - start)) / CLOCKS_PER_SEC);

    // Cleanup
    free(test_signal);

    // Example usage with 2D image - Serial version
    int image_size = size;

    // Example usage with 2D image - Parallel version
    Complex2D* test_image_parallel = create_complex_2d(image_size, image_size);

    // Generate test image (pattern) - same as serial
    for(int i = 0; i < image_size; i++) {
        for(int j = 0; j < image_size; j++) {
            double x = (double)i / image_size;
            double y = (double)j / image_size;
            test_image_parallel->data[i][j] = sin(2 * PI * 5 * x) *
                                             sin(2 * PI * 5 * y);
        }
    }

    // Process 2D image - Parallel
    start = clock();
    process_image_parallel(test_image_parallel, image_size, image_size);
    end = clock();

    printf("2D Image processing time (parallel): %f seconds\n",
           ((double)(end - start)) / CLOCKS_PER_SEC);

    // Cleanup
    thpool_destroy(thpool);
    free_complex_2d(test_image_parallel);

    return 0;
}

