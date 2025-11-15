from PIL import Image


def downsize_image(input_filename: str, output_filename: str, downsize_factor: float):
    """Load a .jpg image, downsize it by the specified factor, and save it."""
    with Image.open(input_filename) as img:
        # Calculate new dimensions
        new_width = int(img.width / downsize_factor)
        new_height = int(img.height / downsize_factor)

        # Downsize the image
        downsized_img = img.resize((new_width, new_height))

        # Convert image to black and white
        bw_img = downsized_img.convert("L")
        # Replace downsized image with the black and white version
        downsized_img = bw_img

        # Save the downsized image
        downsized_img.save(output_filename)

# Example usage:
# downsize_image("input.jpg", "output.jpg", 2.0)  # This will reduce dimensions by half


if __name__ == "__main__":
    downsize_image("San_Francisco_Bay.jpg", "San_Francisco_Bay_downsized.jpg", 2.0)