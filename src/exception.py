import sys
# from src.logger import logging -- This is giving error when running python exception.py

# Creating a function to show this message as my own custom message
def error_message_detail(error, error_detail:sys):

    # This variable gives all ṭhe information about which file the error has occured, which line , etc
    _,_,exc_tb = error_detail.exc_info()

    # Basically we are fetching the file name inside "exc_tb"
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}], line [{1}], error message: [{2}]".format(
                            file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

# Creating my own exception method, overriding the __init__
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("DivideByZero Error")
        raise CustomException(e,sys)