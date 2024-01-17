from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata
import argparse

ObjectDetectorWriter = object_detector.MetadataWriter

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate TFLite object detector metadata")
parser.add_argument("--model_path", required=True, help="Path to the TFLite model file")
parser.add_argument("--label_file", required=True, help="Path to the labelmap file")
parser.add_argument("--save_to_path", required=True, help="Path to save the metadata TFLite file")
args = parser.parse_args()

_MODEL_PATH = args.model_path
_LABEL_FILE = args.label_file
_SAVE_TO_PATH = args.save_to_path

# Create ObjectDetectorWriter and save metadata
writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [127.5], [127.5], [_LABEL_FILE])
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

# Verify the populated metadata and associated files.
displayer = metadata.MetadataDisplayer.with_model_file(_SAVE_TO_PATH)
print("Metadata populated:")
print(displayer.get_metadata_json())
print("Associated file(s) populated:")
print(displayer.get_packed_associated_file_list())
