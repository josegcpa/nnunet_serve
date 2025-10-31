from nnunet_serve.logging_utils import get_logger
from nnunet_serve.nnunet_api import get_model_dictionary
from nnunet_serve.seg_writers import SegWriter

logger = get_logger(__name__)


def main():
    model_dicts, _ = get_model_dictionary()

    for k, v in model_dicts.items():
        if "metadata" not in v:
            continue
        logger.info(f"Validating metadata for {k}")
        SegWriter.init_from_metadata_dict(v["metadata"], validate=True)


if __name__ == "__main__":
    main()
