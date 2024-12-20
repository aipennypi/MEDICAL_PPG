import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# read the Video，Convert to Numpy Array
def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # convert each frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    video.release()
    frames = np.array(frames, dtype=np.uint8)
    return frames

# Segennet the skin
def segment_face_region(data):
    lower = np.array([0, 140, 140])
    upper = np.array([255, 170, 255])
    face = data[0]
    face = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
    mask = cv2.inRange(face, lower, upper)
    mask = mask[:, :, np.newaxis]/255
    mask = mask > 0.5
    return mask

# scale final_dim =（高度, 寬度），例如（100,100）
def avg_pooling(data, final_dim):
    h, w = data.shape[1], data.shape[2]
    new_h, new_w = final_dim
    h_stride = h // new_h
    w_stride = w // new_w
    pooled_data = np.zeros((data.shape[0], new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            pooled_data[:, i, j] = np.mean(data[:, i*h_stride:(i+1)*h_stride, j*w_stride:(j+1)*w_stride], axis=(1, 2))
    return pooled_data

# Normalize the video
def normalize_along_time(data):
    data = data.astype(np.float32)
    mean_data = np.mean(data, axis=0)

    # Calculate standard deviation, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        std_data = np.std(data, axis=0)
        std_data = np.where(std_data == 0, 1, std_data)

    # Normalize the data
    normalized_data = (data - mean_data) / std_data

    # Replace inf and nan values with 0
    normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)

    return normalized_data


# Butterworth
def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)  # Apply filtering along the time axis

def butterworth_lowpass_filter(data, lowcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    b, a = butter(order, low, btype='low')
    return filtfilt(b, a, data, axis=0)

# 
def gaussian_amplitude_filter(data, std, window_size=5):
    epsilon = 1e-10  # Small value to prevent division by zero

    if data.ndim == 1:
        weights = np.zeros_like(data)
        for j in range(len(data)):
            start = int(max(0, j - window_size // 2))
            end = int(min(len(data), j + window_size // 2 + 1))
            weights[j] = np.exp(-max(data[start:end] ** 2) / (2 * std ** 2 + epsilon))
        out = data * weights
    else:
        shape = data.shape
        flatten_data = np.reshape(data, (data.shape[0], -1))
        transpose_data = np.transpose(flatten_data)
        out = np.zeros_like(transpose_data)
        for i in range(len(transpose_data)):
            out[i] = gaussian_amplitude_filter(transpose_data[i], std, window_size)
        out = np.transpose(out)
        out = np.reshape(out, shape)
    return out


# calculate the FFT
def plot_fft(data, fs):
    n = len(data)
    T = 1/fs
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(n, T)[:n//2]
    xf = xf * 60  # convert to bpm
    yf = 2.0/n * np.abs(yf[0:n//2])

    peak = np.argmax(yf)
    bpm = xf[peak]

    plt.plot(xf, yf)
    plt.title('FFT of PPG. Peak at {:.2f} BPM'.format(bpm))
    plt.ylabel('Magnitude')
    plt.xlabel('Heartrate (BPM)')
    plt.xlim(0, 400)
    plt.show()
path1='/content/drive/MyDrive/medicalsingnal/0902 生理訊號量測/PPG/fingerPPG.mp4'

# 1. read the ideo
fingerPPG = read_video(path3)
print(f"RGB): {fingerPPG.shape}")

# 2. read the mean of the Video
# Hint: np.mean(video, axis=(1, 2))
fingerPPG = np.mean(fingerPPG, axis=(1, 2))

# 3. Draw the Singnal
plt.figure(figsize=(15, 5))
plt.plot(fingerPPG[:, 0], label='red', color='red')
plt.plot(fingerPPG[:, 1], label='green', color='green')
plt.plot(fingerPPG[:, 2], label='blue', color='blue')
plt.legend()
plt.show()

# 4. 
fingerPPG1 = fingerPPG[:, 1]

# 5. 
fingerPPG2 = normalize_along_time(fingerPPG1)

# 6. 
fingerPPG3 = butterworth_bandpass_filter(fingerPPG2, lowcut, highcut, fs)

# 7. 
fingerPPG4 = normalize_along_time(fingerPPG3)

fingerPPG = fingerPPG4

# 8. 
plt.figure(figsize=(15, 5), constrained_layout=True)
plt.subplot(4, 1, 1)
plt.plot(fingerPPG1)
plt.title('Original Signal')
plt.subplot(4, 1, 2)
plt.plot(fingerPPG2)
plt.title('Normalized Signal')
plt.subplot(4, 1, 3)
plt.plot(fingerPPG3)
plt.title('Bandpass Filtered Signal')
plt.subplot(4, 1, 4)
plt.plot(fingerPPG4)
plt.title('Normalized Signal after Bandpass Filtering')
plt.show()

# Find the High peaks
peaks 位置（index）
from scipy.signal import find_peaks

peaks, _ = find_peaks(facePPG_v2,distance=65)

plt.figure(figsize=(15,2))
plt.plot(facePPG_v2)
plt.plot(peaks, facePPG_v2[peaks], "ro")
plt.show()

# PPG Index
from scipy.signal import find_peaks

peaks, _ = find_peaks(fingerPPG,distance=65)

plt.figure(figsize=(15,2))
plt.plot(fingerPPG)
plt.plot(peaks, fingerPPG[peaks], "ro")
plt.show()
# Calculate the Heart beat
print("heart beat:",len(peaks)/90*60)
