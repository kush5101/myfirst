from deepface import DeepFace
import sys
import traceback

try:
    print("Testing DeepFace build model...")
    DeepFace.build_model("Emotion")
    print("Success building Emotion model")
except Exception as e:
    with open("error.log", "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
    print("Error written to error.log")
    sys.exit(1)
