from nnunet_serve.nnunet_serve import get_model_dictionary
from nnunet_serve.seg_writers import SegWriter
from nnunet_serve.logging_utils import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    model_dicts, _ = get_model_dictionary()

    for k, v in model_dicts.items():
        if "metadata" not in v:
            continue
        logger.info(f"Validating metadata for {k}")
        SegWriter.init_from_metadata_dict(v["metadata"], validate=True)
