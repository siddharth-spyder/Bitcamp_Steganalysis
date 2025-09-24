import os
import numpy as np
import cv2
import joblib

def debug_image_reading(image_path):
    print(f"Attempting to read image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File does not exist at {image_path}")
        return False
    
    # Try reading with OpenCV
    try:
        image = cv2.imread(image_path)
        
        if image is None:
            print("cv2.imread() returned None")
            return False
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        
        return True
    except Exception as e:
        print(f"Error reading image: {e}")
        return False

class SteganalysisInference:
    def __init__(self, model_path):
        """
        Initialize steganalysis inference system
        
        Args:
            model_path (str): Path to saved model joblib file
        """
        # Load saved model pipeline
        self.model = joblib.load(model_path)
    
    def _extract_features(self, image):
        """
        Feature extraction method matching the training process
        
        Args:
            image (numpy.ndarray): Input image
        
        Returns:
            numpy.ndarray: Extracted features
        """
        if image is None:
            return np.zeros(150)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float32) / 255.0
        
        # 1. Basic statistical features
        stats = [
            np.mean(gray), np.std(gray), 
            np.percentile(gray, 25), np.percentile(gray, 75),
            np.mean(np.abs(gray - np.mean(gray))),
        ]
        
        # 2. Advanced filter responses
        filters = [
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # Sobel X
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # Sobel Y
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),  # Laplacian
            np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])  # Edge enhancement
        ]
        
        filter_stats = []
        for kernel in filters:
            filtered = cv2.filter2D(gray, -1, kernel)
            filter_stats.extend([
                np.mean(filtered),
                np.std(filtered),
                np.percentile(filtered, 10),
                np.percentile(filtered, 90)
            ])
        
        # 3. Gradient features
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        
        gradient_stats = [
            np.mean(np.abs(gx)),
            np.mean(np.abs(gy)),
            np.mean(np.sqrt(gx**2 + gy**2)),
            np.std(np.sqrt(gx**2 + gy**2))
        ]
        
        # 4. LSB analysis
        if len(image.shape) == 3:
            lsb_features = []
            for channel in range(3):
                lsb_plane = (image[:,:,channel] & 1).flatten()
                lsb_features.extend([
                    np.mean(lsb_plane),
                    np.std(lsb_plane),
                    len(np.where(np.diff(lsb_plane) != 0)[0]) / len(lsb_plane)
                ])
        else:
            lsb_features = [0] * 9
        
        # Combine all features
        full_features = stats + filter_stats + gradient_stats + lsb_features
        
        return np.array(full_features)
    
    def predict_steganography(self, image_path, threshold=0.5):
        """
        Predict steganography probability for an image
        
        Args:
            image_path (str): Path to image file
            threshold (float): Probability threshold for classification
        
        Returns:
            dict: Prediction results
        """
        # Read image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None
        
        # Extract features
        features = self._extract_features(image)
        
        # Predict probability
        probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
        
        # Return cleaned-up dictionary with Python-native types
        return {
            'image': os.path.basename(image_path),
            'steganography_probability': float(probability),
            'is_suspicious': bool(probability > threshold),
            'status': 'Stego Image' if probability > threshold else 'Clean Image'
        }
    
    def batch_predict(self, image_directory, threshold=0.45):
        """
        Predict steganography for multiple images in a directory
        
        Args:
            image_directory (str): Directory containing images
            threshold (float): Probability threshold for classification
        
        Returns:
            list: Prediction results for all images
        """
        # Find all image files
        image_files = [
            os.path.join(image_directory, f) 
            for f in os.listdir(image_directory) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        
        # Predict for each image
        results = []
        for single_image_path in image_files:
            try:
                result = self.predict_steganography(single_image_path, threshold)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {single_image_path}: {e}")
        
        return results

# Optional: Debugging code (safe to remove in production)
if __name__ == '__main__':
    image_path = "C:\\Users\\Aniruddh Rajagopal\\Downloads\\ALASKA_50774_QF75.jpg"
    debug_image_reading(image_path)

    inferencer = SteganalysisInference("C:\\Users\\Aniruddh Rajagopal\\Downloads\\best_steganalysis_model\\best_steganalysis_model.joblib")
    result = inferencer.predict_steganography(image_path)

    print(result["status"])
