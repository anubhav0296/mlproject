import logging
import os
from datetime import datetime 

# When logging file is triggered, as a first step, this file.log is created
LOG_FILE = f"{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log"

# Here we are joining CWD + logs + LOG_FILE
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Even though there is a file inside this folder, keep on appending the file
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Here we are customizing the logging - CUSTOM_LOGGING
# The print statement which you will use, will add the details in below filepath
# and in below format

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")