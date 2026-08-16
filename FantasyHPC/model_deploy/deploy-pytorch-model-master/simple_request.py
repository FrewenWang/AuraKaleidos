import argparse

import requests

# Initialize the PyTorch REST API endpoint URL.
PYTORCH_REST_API_URL = "http://127.0.0.1:5000/predict"


def predict_result(image_path):
    # Initialize image path
    with open(image_path, "rb") as image_file:
        payload = {"image": image_file.read()}

    # Submit the request.
    response = requests.post(PYTORCH_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if response["success"]:
        # Loop over the predictions and display them.
        for i, result in enumerate(response["predictions"]):
            print(
                "{}. {}: {:.4f}".format(
                    i + 1, result["label"], result["probability"]
                )
            )
    # Otherwise, the request failed.
    else:
        print("Request failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification demo")
    parser.add_argument("--file", type=str, help="test image file")

    args = parser.parse_args()
    predict_result(args.file)
