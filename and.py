from utils.all_utils import save_model, save_plot
from utils.models import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import logging

def main(data, eta, epochs, filename, plotfileName):
    
    df = pd.DataFrame(data)
    logging.info(f"This is the actual dataframe{df}")
    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plotfileName, model)

if __name__ == '__main__': # entry point

    AND = {
            "x1" : [0,0,1,1],
            "x2" : [0,1,0,1],
            "y" : [0,0,0,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try :
        logging.info(">>>>>>>> Starting the training <<<<<<<")
        main(data=AND, eta=ETA, epochs=EPOCHS, filename="and.model", plotfileName="and.png")
        logging.info(">>>>>>>> training completed successfully <<<<<<<<")
    except Exception as e:
        logging.exception(e)