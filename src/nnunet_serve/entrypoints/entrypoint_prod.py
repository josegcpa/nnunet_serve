import json
import os
import time

import requests

from nnunet_serve.entrypoints.entrypoint import (
    main_with_args as main_with_args_entrypoint,
)
from nnunet_serve.utils import make_parser
from nnunet_serve.logging_utils import get_logger

logger = get_logger(__name__)


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.count = 0

    def print_time(self, extra_str=None):
        elapsed_time = time.time() - self.start_time
        if extra_str is None:
            logger.info(f"Elapsed time {self.count}: {elapsed_time} seconds")
        else:
            logger.info(
                f"Elapsed time {self.count} ({extra_str}): \
                  {elapsed_time} seconds"
            )
        self.count += 1


def main_with_args(args):
    timer = Timer()
    # easier to adapt to docker
    if "SERIES_PATHS" in os.environ:
        args.series_paths = os.environ["SERIES_PATHS"].split(" ")

    main_with_args_entrypoint(args)

    timer.print_time()

    return args.success_message


if __name__ == "__main__":
    parser = make_parser()

    parser.add_argument(
        "--job_id",
        default=None,
        help="Job ID that will be used to post job status/create log file",
        type=str,
    )
    parser.add_argument(
        "--update_url",
        default=None,
        help="URL to be used to post job status",
        type=str,
    )
    parser.add_argument(
        "--success_message",
        default="done",
        help="Message to be posted in case of success",
        type=str,
    )
    parser.add_argument(
        "--failure_message",
        default="failed",
        help="Message to be posted in case of failure",
        type=str,
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Path to log file (with job_id, and success/failure messages)",
        type=str,
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enters debug mode",
    )

    args = parser.parse_args()

    if args.debug is not True:
        try:
            log_out = main_with_args(args)
            status = args.success_message
            err = ""
        except KeyboardInterrupt:
            status = args.failure_message
            log_out = ""
            err = "User interrupted execution"
        except Exception as e:
            status = args.failure_message
            log_out = ""
            err = repr(e)
            logger.error(e)
    else:
        log_out = main_with_args(args)
        status = args.success_message
        err = ""

    if "empty" in str(log_out):
        status = status + "_no_output"
        log_txt = "No lesion detected"
    else:
        log_txt = ""
    data = {
        "job_id": args.job_id,
        "status": status,
        "output_log": log_txt,
        "output_error": err,
    }

    if (args.update_url is not None) and (args.update_url != "skip"):
        logger.info(
            f"Posting {status} to {args.update_url} for job {args.job_id}"
        )
        requests.post(
            f"{args.update_url}",
            data=data,
        )

    if args.log_file is not None:
        if os.path.exists(args.log_file):
            with open(args.log_file, "r") as f:
                log_out = json.load(f)
            if "job_id" not in log_out:
                log_out["job_id"] = args.job_id
            for k in ["status", "output_log", "output_error"]:
                log_out[k] = data[k]
        else:
            log_out = data
        with open(args.log_file, "w") as f:
            json.dump(log_out, f)

    if len(err) > 0:
        raise Exception(err)
