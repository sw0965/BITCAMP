from PIL import Image
from autocrop import Cropper

cropper = Cropper()

cropped_array = cropper.crop('ms_net/ms')

cropper_image = Image.fromarray(cropped_array)
cropped_image.save('ms/cropped.png')