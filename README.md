<h1>Overview</h1>
This project for extracting and analyzing heart rate data from video recordings. It combines techniques from computer vision, signal processing, and data visualization to provide insights into physiological signals. Future improvements may include optimizing the filtering process, enhancing peak detection algorithms, or implementing real-time processing capabilities.

<h1>How to run</h1>

Run this script after modifying the path1 variable with the path to your video file containing fingertip PPG signals. Ensure to adjust the filter parameters (e.g., lowcut, highcut, fs) based on your specific signal characteristics.

<h1>Features</h1>
<h3>Load the Video</h3>

    The video is loaded, and the frames are captured and averaged across the height and width dimensions to reduce noise.
<h3>Mean Calculation</h3>

    The average color value across the frames is calculated for each color channel (red, green, blue).
<h3>Signal Visualization</h3>

    The RGB channels of the averaged frames are plotted for visual inspection.

<h3>Signal Filtering</h3>

    The green channel of the averaged data is extracted, and normalization and bandpass filtering are applied to isolate the PPG signal.

<h3>Peak Detection</h3>

    Peaks in the filtered signal are detected using the find_peaks function from scipy.signal.

<h3>Heart Rate Calculation</h3>

    The heart rate is computed based on the number of detected peaks over the video duration.
