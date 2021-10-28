from utils.all_utils import save_model, save_plot
from utils.models import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s : %(levelname)s : %(module)s] %(message)s"
logging_dir = "logs" 
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir,"running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def main(data, eta, epochs, filename, plotfileName):

    df_OR = pd.DataFrame(data)
    logging.info(f"This is the actual dataframe{df_OR}")

    X,y = prepare_data(df_OR)

    model_OR = Perceptron(eta=eta, epochs=epochs)
    model_OR.fit(X,y)

    _ = model_OR.total_loss()

    save_model(model_OR, filename=filename)
    save_plot(df_OR, plotfileName, model_OR)


if __name__ == '__main__': # entry point

    OR = {
        "x1" : [0,0,1,1],
        "x2" : [0,1,0,1],
         "y" : [0,1,1,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    try :
        logging.info(">>>>>>>> Starting the training <<<<<<<")
        main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotfileName="or.png")
        logging.info(">>>>>>>> training completed successfully <<<<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e