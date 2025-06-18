//Implementation of a Fast Fourier Transform (FFT) calculator that includes both 1D and 2D transforms, along with some practical applications like frequency filtering and signal analysis

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>

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

// 2D FFT implementation
void fft_2d(Complex2D* data, int inverse) {
    int rows = data->rows;
    int cols = data->cols;

    // Allocate temporary array for columns
    cplx* temp = (cplx*)malloc(rows * sizeof(cplx));

    // Transform rows
    for(int i = 0; i < rows; i++) {
        fft_1d(data->data[i], cols, inverse);
    }

    // Transform columns
    for(int j = 0; j < cols; j++) {
        // Copy column to temporary array
        for(int i = 0; i < rows; i++) {
            temp[i] = data->data[i][j];
        }

        // Transform
        fft_1d(temp, rows, inverse);

        // Copy back
        for(int i = 0; i < rows; i++) {
            data->data[i][j] = temp[i];
        }
    }

    free(temp);
}

// Frequency domain filtering
void apply_frequency_filter(Complex2D* data, double cutoff) {
    int rows = data->rows;
    int cols = data->cols;
    int center_row = rows / 2;
    int center_col = cols / 2;

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            // Calculate distance from center
            double dist = sqrt(pow(i - center_row, 2) +
                             pow(j - center_col, 2));

            // Apply low-pass filter
            if(dist > cutoff) {
                data->data[i][j] *= exp(-pow(dist - cutoff, 2) / 100.0);
            }
        }
    }
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

// Example of 2D image processing
void process_image(Complex2D* image) {
    // Forward 2D FFT
    fft_2d(image, 0);

    // Apply frequency domain filter
    apply_frequency_filter(image, image->rows/4);

    // Inverse 2D FFT
    fft_2d(image, 1);
}

int main(int argc, char * argv[]) {

    int size = atoi(argv[1]);

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

    // Process 1D signal
    clock_t start = clock();
    process_signal(test_signal, signal_length);
    clock_t end = clock();

    printf("1D Signal processing time: %f seconds\n",
           ((double)(end - start)) / CLOCKS_PER_SEC);

    // Example usage with 2D image
    int image_size = size;
    Complex2D* test_image = create_complex_2d(image_size, image_size);

    // Generate test image (pattern)
    for(int i = 0; i < image_size; i++) {
        for(int j = 0; j < image_size; j++) {
            double x = (double)i / image_size;
            double y = (double)j / image_size;
            test_image->data[i][j] = sin(2 * PI * 5 * x) *
                                    sin(2 * PI * 5 * y);
        }
    }

    // Process 2D image
    start = clock();
    process_image(test_image);
    end = clock();

    printf("2D Image processing time: %f seconds\n",
           ((double)(end - start)) / CLOCKS_PER_SEC);

    // Cleanup
    free(test_signal);
    free_complex_2d(test_image);

    return 0;
}

