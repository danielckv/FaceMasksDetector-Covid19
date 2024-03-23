# utils.py

classes = ['with_mask', 'without_mask']


def sliding_window(_image, window_size, step_size):
    """
    This function returns a patch of the input 'image' of size
    equal to 'window_size'. The first image returned top-left
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    """
    for y in range(0, _image.shape[0], step_size[1]):
        for x in range(0, _image.shape[1], step_size[0]):
            yield x, y, _image[y: y + window_size[1], x: x + window_size[0]]


def visualize_results(image, predicted_mask):
    # Visualize the image and the predicted mask
    # ...
    pass
