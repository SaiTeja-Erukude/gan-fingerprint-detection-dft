import os
import cv2
import numpy as np
from   tqdm  import tqdm
 

######################
#
######################
def apply_fourier_transform(src_dir: str, out_dir: str) -> bool: 
    try:
        # Check if the source and destination directory exists
        if not os.path.isdir(src_dir):
            print(f"Directory '{src_dir}' does not exist.")
            return False
        
        if not os.path.isdir(out_dir):
            print(f"Directory '{out_dir}' does not exist. Creating it.")
            os.makedirs(out_dir)
        
        # Get all images in the specified directory
        images = []
        for img in os.listdir(src_dir):
            if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".JPEG") or img.endswith(".jpeg"):
                images.append(img)

        for img in tqdm(images):

            try:
                src_img_path = os.path.join(src_dir, img)
                out_img_path = os.path.join(out_dir, img)

                image = cv2.imread(src_img_path)
                gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Compute the discrete Fourier Transform of the image
                fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
                
                # Shift the zero-frequency component to the center of the spectrum
                fourier_shift = np.fft.fftshift(fourier)

                # calculate the magnitude of the Fourier Transform
                magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
                
                # Scale the magnitude for display
                magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

                cv2.imwrite(out_img_path, magnitude)
            
            except Exception as ex:
                print(f"Error for img: {img}. Skipping it!")
                continue


    except Exception as fourier_ex:
        print(f"Error occurred while applying FT: {str(fourier_ex)}.")
        return False



if __name__ == "__main__":

    src_dir = "D:/Projects/GAN Fingerprint Detection/data/raw/real"
    out_dir = "D:/Projects/GAN Fingerprint Detection/data/fourier/real"

    apply_fourier_transform(src_dir, out_dir)