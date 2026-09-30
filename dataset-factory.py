import numpy as np
from scipy.special import softmax

# Configuration for synthetic data generation
NUM_SAMPLES = 1000         # Number of data points to generate
WINDOW_LENGTH = 30         # Length of each time series window (number of frames)
LOWER_BOUND = 0.0          # Lower bound for random initial values in flicker data
UPPER_BOUND = 1.0          # Upper bound for random initial values in flicker data
SALT_PEPPER_NOISE_LEVEL = 0.05  # Level of salt and pepper noise to add
#add some guassian noise 
#types of noises and adding those 
SHIFT_PROBABILITY = 0.2    # Probability of a random shift in value occurring in a frame
SHIFT_RANGE = 0.1  # Range of value shift

SINE_WAVE_AMPLITUDE = 0.2  # Amplitude of the sine wave component
SINE_WAVE_FREQUENCY = 2.0  # Frequency of the sine wave component (in radians per frame)
PHASE_NOISE_RANGE = 2 * np.pi  # Range for the random initial phase of the sine wave

OUTPUT_FILE = "synthetic_flicker_data.npy"  # Output path for generated .npy data

def generate_flicker_data(num_samples, window_length, lower_bound, upper_bound, noise_level,
                          shift_prob, shift_range, sine_amplitude, sine_frequency, phase_noise_range):
    data = []

    for _ in range(num_samples):
        # Step 1: Generate random base values for the flicker series
        flicker = np.random.uniform(lower_bound, upper_bound, window_length)
        
        # Step 2: Add salt and pepper noise
        noise_mask = np.random.choice([0, 1, -1], size=window_length, p=[1 - noise_level, noise_level / 2, noise_level / 2])
        flicker += noise_mask * (upper_bound - lower_bound) * 0.1  # Scaled to fit within bounds

        # Step 3: Introduce random shifts in values
        for i in range(window_length):
            if np.random.rand() < shift_prob:
                flicker[i] += np.random.uniform(-shift_range, shift_range)

        # Step 4: Add a sine wave with random phase
        phase_shift = np.random.uniform(0, phase_noise_range)
        sine_wave = sine_amplitude * np.sin(sine_frequency * np.arange(window_length) + phase_shift)
        flicker_with_sine = flicker + sine_wave

        # The deflickered target: same as flicker but without the sine wave
        deflicker = flicker.copy()
        flicker_with_sine = softmax(flicker_with_sine)
        deflicker = softmax(deflicker)

        # Store the flicker-deflicker pair
        data.append([flicker_with_sine, deflicker])

    return np.array(data, dtype=object)

def main():
    # Generate synthetic flicker and deflicker data
    synthetic_data = generate_flicker_data(
        num_samples=NUM_SAMPLES,
        window_length=WINDOW_LENGTH,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        noise_level=SALT_PEPPER_NOISE_LEVEL,
        shift_prob=SHIFT_PROBABILITY,
        shift_range=SHIFT_RANGE,
        sine_amplitude=SINE_WAVE_AMPLITUDE,
        sine_frequency=SINE_WAVE_FREQUENCY,
        phase_noise_range=PHASE_NOISE_RANGE
    )

    # Save the data to a .npy file for training
    np.save(OUTPUT_FILE, synthetic_data)
    print(f"Synthetic flicker data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
