"""Small OpenCV display helpers."""


def show_image(name, image) -> None:
    """Display an image, importing OpenCV only when the GUI helper is used."""
    import cv2

    cv2.imshow(name, image)
