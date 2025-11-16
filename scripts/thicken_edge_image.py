from scipy.ndimage import binary_dilation
import numpy as np
import PIL.Image as Image


input_image = Image.open("resources/San_Francisco_Bay_Edges_Thin.jpg")
input_image_array = np.array(input_image)[:, :, 0]
edge_image = input_image_array == 0
thickened_edge_image = binary_dilation(edge_image)
thickened_edge_image_output = np.where(thickened_edge_image, 0, 255).astype(np.uint8)
thickned_image_rgb = np.stack([thickened_edge_image_output, thickened_edge_image_output, thickened_edge_image_output], axis=-1)
output_image = Image.fromarray(thickned_image_rgb)
output_image.save("resources/San_Francisco_Bay_Edges.jpg")