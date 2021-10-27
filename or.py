from utils.all_utils import save_model, save_plot
from utils.models import Perceptron
from utils.all_utils import prepare_data
import pandas as pd

def main(data, eta, epochs, filename, plotfileName):

    df_OR = pd.DataFrame(data)
    print(df_OR)

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

    main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotfileName="or.png")